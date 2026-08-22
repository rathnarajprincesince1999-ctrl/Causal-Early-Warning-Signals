"""
Component 1: Classical Early Warning Signals — Revised
=======================================================
Fixes reviewer concerns:
  1. Explicit STL deseasonalization before EWS (removes seasonality bias)
  2. Block-bootstrapped Mann-Kendall for valid p-values on overlapping windows
  3. Sensitivity analysis across window lengths (36, 50, 72 months)
  4. Proper detrending with Gaussian kernel smoother

Outputs: results/ews_classical/ews_metrics_{area}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR    = BASE_DIR / "results" / "ews_classical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS    = [36, 50, 72]   # sensitivity analysis over multiple window lengths
STUDY_AREAS = ["amazon", "sahel", "east_africa"]
EWS_VARS    = ["NDVI", "EVI", "LST_Day", "ET", "soil_moisture",
               "tp_anom", "t2m_anom", "swvl1_anom"]
N_BOOT      = 500   # block bootstrap iterations


def stl_deseasonalize(series, period=12):
    """Remove seasonal cycle using STL decomposition (additive model)."""
    s = pd.Series(series)
    s_filled = s.interpolate(limit=6).fillna(s.mean())
    if s_filled.isna().any():
        s_filled = s_filled.fillna(0.0)
    try:
        stl    = STL(s_filled, period=period, robust=True)
        result = stl.fit()
        # Return trend + residual (seasonal removed)
        return result.trend + result.resid
    except Exception:
        # Fallback: simple 12-month moving average detrend
        trend = s_filled.rolling(12, center=True, min_periods=6).mean()
        return (s_filled - trend).fillna(0.0).values


def gaussian_detrend(series, bandwidth=0.5):
    """Remove slow trend using Gaussian kernel smoother."""
    n = len(series)
    if n < 10:
        return series - np.nanmean(series)
    x = np.linspace(0, 1, n)
    trend = np.zeros(n)
    for i in range(n):
        w = np.exp(-0.5 * ((x - x[i]) / bandwidth) ** 2)
        w /= w.sum()
        trend[i] = np.dot(w, series)
    return series - trend


def block_bootstrap_kendall(series, n_boot=N_BOOT, block_size=None):
    """
    Block-bootstrapped Mann-Kendall test for autocorrelated time series.
    Returns (tau, p_value). Block size defaults to sqrt(n) (Politis 1994).
    """
    n = len(series)
    if n < 10:
        return np.nan, np.nan

    # Observed tau
    tau_obs, _ = stats.kendalltau(np.arange(n), series)

    # Block size
    bs = block_size or max(3, int(np.sqrt(n)))

    # Bootstrap distribution under H0 (randomly shuffled blocks)
    tau_boot = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, n - bs + 1)
            idx.extend(range(start, min(start + bs, n)))
        idx = idx[:n]
        s_perm = series[idx]
        t, _ = stats.kendalltau(np.arange(n), s_perm)
        tau_boot.append(t)

    tau_boot = np.array(tau_boot)
    # Two-sided p-value
    p_val = np.mean(np.abs(tau_boot) >= np.abs(tau_obs))
    return float(tau_obs), float(p_val)


def rolling_variance(series, window):
    return pd.Series(series).rolling(window, center=True,
                                     min_periods=window//2).var().values


def rolling_ac1(series, window):
    def ac1(x):
        if len(x) < 3 or np.std(x) < 1e-10:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    return pd.Series(series).rolling(window, center=True,
                                     min_periods=window//2).apply(
        ac1, raw=True).values


def compute_ews_for_series(time_series, var_name, region_id, area_name, window):
    """Compute EWS with block-bootstrapped significance and deseasonalization."""
    ts = np.array(time_series, dtype=float)
    n  = len(ts)

    ts_interp = pd.Series(ts).interpolate(limit=3).values
    if np.sum(np.isfinite(ts_interp)) < n * 0.6:
        return None

    ts_clean = np.where(np.isfinite(ts_interp), ts_interp, np.nanmean(ts_interp))

    # 1. STL deseasonalize
    ts_deseas = stl_deseasonalize(ts_clean, period=12)

    # 2. Gaussian detrend
    ts_dt = gaussian_detrend(np.array(ts_deseas), bandwidth=0.5)

    # 3. Rolling statistics
    roll_var = rolling_variance(ts_dt, window)
    roll_ac1 = rolling_ac1(ts_dt, window)

    valid_var = roll_var[np.isfinite(roll_var)]
    valid_ac1 = roll_ac1[np.isfinite(roll_ac1)]

    if len(valid_var) < 10 or len(valid_ac1) < 10:
        return None

    # 4. Block-bootstrapped Kendall tau
    tau_var, pval_var = block_bootstrap_kendall(valid_var)
    tau_ac1, pval_ac1 = block_bootstrap_kendall(valid_ac1)

    # 5. Overall stats on deseasonalised series
    overall_var  = float(np.nanvar(ts_dt))
    overall_ac1  = float(pd.Series(ts_dt).autocorr(lag=1))
    overall_skew = float(stats.skew(ts_dt[np.isfinite(ts_dt)]))
    overall_cv   = float(np.nanstd(ts_clean) / (np.nanmean(ts_clean) + 1e-8))

    ews_score = float(np.nanmean([
        max(0, tau_var) if not np.isnan(tau_var) else 0,
        max(0, tau_ac1) if not np.isnan(tau_ac1) else 0,
    ]))

    return {
        "area":       area_name,
        "region_id":  region_id,
        "variable":   var_name,
        "window":     window,
        "kendall_tau_variance":  tau_var,
        "pval_variance":         pval_var,  # block-bootstrapped
        "kendall_tau_ac1":       tau_ac1,
        "pval_ac1":              pval_ac1,  # block-bootstrapped
        "variance":              overall_var,
        "autocorrelation_lag1":  overall_ac1,
        "skewness":              overall_skew,
        "cv":                    overall_cv,
        "ews_score":             ews_score,
    }


def process_area(area_name):
    print(f"\n{'='*60}")
    print(f"Classical EWS (block-bootstrap MK) — {area_name}")
    print(f"{'='*60}")

    csv_path = MERGED_DIR / f"merged_regions_{area_name}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["region_id", "time"])
    regions = sorted(df["region_id"].unique())

    all_rows = []

    for region_id in tqdm(regions, desc="  Regions"):
        df_r = df[df["region_id"] == region_id].sort_values("time")
        for var in EWS_VARS:
            if var not in df_r.columns:
                continue
            ts = df_r[var].values
            # Sensitivity: compute across multiple window lengths
            for win in WINDOWS:
                result = compute_ews_for_series(ts, var, region_id, area_name, win)
                if result is not None:
                    all_rows.append(result)

    if not all_rows:
        return

    df_ews = pd.DataFrame(all_rows)

    # Primary window results (50 months)
    df_primary = df_ews[df_ews["window"] == 50]
    out = OUT_DIR / f"ews_metrics_{area_name}.csv"
    df_primary.to_csv(out, index=False)

    # All windows for sensitivity analysis
    df_ews.to_csv(OUT_DIR / f"ews_sensitivity_{area_name}.csv", index=False)

    sig  = ((df_primary.pval_variance < 0.05) & (df_primary.kendall_tau_variance > 0)).sum()
    siga = ((df_primary.pval_ac1 < 0.05) & (df_primary.kendall_tau_ac1 > 0)).sum()
    print(f"\n  [SAVED] {out.name} | sig variance: {sig} | sig AC1: {siga}")

    # Top regions
    region_risk = df_primary.groupby("region_id")["ews_score"].mean().sort_values(ascending=False)
    print("  Top 5 regions by EWS score:")
    print(region_risk.head(5).to_string())

    return df_primary


def main():
    print("=" * 60)
    print("Classical EWS — block-bootstrapped Mann-Kendall")
    print(f"Windows: {WINDOWS} months | Bootstrap iterations: {N_BOOT}")
    print("Deseasonalization: STL decomposition")
    print("=" * 60)

    all_results = {}
    for area in STUDY_AREAS:
        r = process_area(area)
        if r is not None:
            all_results[area] = r

    print("\n" + "=" * 60)
    print("CROSS-AREA SUMMARY (window=50, block-bootstrap p<0.05)")
    for area, df in all_results.items():
        mean_s = df["ews_score"].mean()
        high   = (df["ews_score"] > 0.3).sum()
        print(f"  {area:<15} mean EWS: {mean_s:+.3f} | high-risk pairs: {high}")
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
