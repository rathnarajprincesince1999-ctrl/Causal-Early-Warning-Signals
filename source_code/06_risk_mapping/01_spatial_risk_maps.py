"""
Component 4: Spatial Collapse Risk Mapping
==========================================
Applies the trained TCN model pixel-by-pixel to generate spatially
explicit collapse risk probability maps over time.

Also computes:
  - Trend in risk score (2000-2023)
  - Year of peak risk per pixel
  - Spatial clustering of high-risk zones

Outputs:
  results/risk_maps/
    risk_map_{area}_YYYY.nc     — annual risk maps
    risk_trend_{area}.nc        — Kendall tau trend in risk
    risk_peak_year_{area}.nc    — year of maximum risk
    high_risk_zones_{area}.csv  — contiguous high-risk clusters
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy import stats
from scipy.ndimage import label as ndimage_label
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch

BASE_DIR    = Path(__file__).resolve().parents[1]
MERGED_DIR  = BASE_DIR / "data" / "processed" / "merged"
DL_DIR      = BASE_DIR / "results" / "deep_learning"
EWS_DIR     = BASE_DIR / "results" / "ews_classical"
OUT_DIR     = BASE_DIR / "results" / "risk_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_AREAS = ["amazon", "sahel", "east_africa"]

FEATURE_COLS = ["NDVI", "EVI", "LST_Day", "ET", "soil_moisture",
                "swvl1", "swvl1_anom", "t2m", "t2m_anom"]
SEQ_LEN = 60

HIGH_RISK_THRESHOLD = 0.6   # risk score above which a pixel is "high risk"
MIN_CLUSTER_SIZE    = 10    # minimum pixels for a risk cluster


def load_model(device):
    """Load trained TCN model."""
    import json
    import sys
    sys.path.insert(0, str(BASE_DIR / "04_deep_learning"))
    from train_ews_detector import TCNEWSDetector  # noqa

    config_path = DL_DIR / "model_config.json"
    weights_path = DL_DIR / "model_weights.pt"

    if not weights_path.exists():
        print("  [WARN] No trained model found. Using classical EWS for risk maps.")
        return None

    with open(config_path) as f:
        config = json.load(f)

    model = TCNEWSDetector(
        n_features=config["n_features"],
        seq_len=config["seq_len"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def compute_classical_risk(pixel_ts):
    """
    Fallback: compute composite risk from classical EWS.
    Returns risk score in [0, 1].
    """
    from scipy.stats import kendalltau

    if len(pixel_ts) < 30 or np.sum(np.isfinite(pixel_ts)) < 20:
        return np.nan

    ts = pixel_ts.copy()
    ts = pd.Series(ts).interpolate(limit=3).values
    ts = np.where(np.isfinite(ts), ts, np.nanmean(ts))

    # Detrend
    x = np.arange(len(ts))
    slope, intercept, _, _, _ = stats.linregress(x, ts)
    ts_dt = ts - (slope * x + intercept)

    # Rolling variance trend
    window = 36
    roll_var = pd.Series(ts_dt).rolling(window, min_periods=window//2).var().dropna().values
    if len(roll_var) < 10:
        return np.nan

    tau_var, pval_var = kendalltau(np.arange(len(roll_var)), roll_var)

    # Rolling AC1 trend
    roll_ac1 = pd.Series(ts_dt).rolling(window, min_periods=window//2).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan,
        raw=True
    ).dropna().values
    if len(roll_ac1) < 10:
        return np.nan

    tau_ac1, pval_ac1 = kendalltau(np.arange(len(roll_ac1)), roll_ac1)

    # Composite: normalize tau to [0, 1] and weight by significance
    def weighted_tau(tau, pval):
        if np.isnan(tau) or np.isnan(pval):
            return 0.0
        sig_weight = max(0, 1 - pval / 0.1)  # higher weight for lower p-value
        return max(0, tau) * sig_weight

    score = 0.5 * weighted_tau(tau_var, pval_var) + 0.5 * weighted_tau(tau_ac1, pval_ac1)
    return float(np.clip(score, 0, 1))


def build_risk_map_classical(area_name):
    """
    Build pixel-level risk maps using vectorized classical EWS.
    Processes all pixels simultaneously using xarray/numpy operations.
    """
    print(f"  Building vectorized EWS risk map for {area_name}...")

    pixel_path = MERGED_DIR / f"merged_pixel_{area_name}.nc"
    ds = xr.open_dataset(pixel_path)

    lats = ds.lat.values
    lons = ds.lon.values
    n_lat, n_lon = len(lats), len(lons)
    times = pd.to_datetime(ds.time.values)
    years = np.array([t.year for t in times])
    n_t   = len(times)

    # Use NDVI as primary collapse indicator
    ndvi = ds["NDVI"].values.astype(float)   # (time, lat, lon)
    ds.close()

    # Replace fill values / extreme values
    ndvi = np.where(np.abs(ndvi) > 100, np.nan, ndvi)

    print(f"    Grid: {n_lat}×{n_lon} | Valid pixels: {np.sum(~np.all(np.isnan(ndvi), axis=0)):,}")

    # ── Vectorized detrending ──────────────────────────────────────────────
    # Linear detrend across time axis per pixel
    x      = np.arange(n_t, dtype=float)
    x_mean = x.mean()
    x_std  = x.std()
    x_norm = (x - x_mean) / (x_std + 1e-8)

    # Replace NaN with pixel mean for computation
    ndvi_filled = ndvi.copy()
    pixel_means = np.nanmean(ndvi, axis=0, keepdims=True)
    nan_mask    = np.isnan(ndvi)
    ndvi_filled[nan_mask] = np.broadcast_to(pixel_means, ndvi.shape)[nan_mask]

    # Compute slope: cov(x, y) / var(x)  — shape (lat, lon)
    x_3d   = x_norm[:, np.newaxis, np.newaxis]
    slopes = np.mean(x_3d * ndvi_filled, axis=0) / (np.var(x_norm) + 1e-8)

    # Detrended
    ndvi_dt = ndvi_filled - slopes[np.newaxis, :, :] * x_3d

    # ── Rolling variance & AC1 using strided windows ───────────────────────
    WINDOW = 36  # 3 years

    def rolling_kendall_vectorized(arr_3d, window):
        """
        Compute Kendall tau trend on rolling statistics for all pixels.
        Returns tau array (lat, lon).
        """
        n = arr_3d.shape[0]
        n_windows = n - window + 1

        # Compute rolling stat at each window end
        roll_stats = np.full((n_windows, arr_3d.shape[1], arr_3d.shape[2]), np.nan)

        for i in range(n_windows):
            chunk = arr_3d[i:i+window]
            roll_stats[i] = np.nanvar(chunk, axis=0)

        # Kendall tau per pixel (vectorized using correlation as approximation)
        # Spearman correlation between index and rolling stat ≈ Kendall tau
        t_idx = np.arange(n_windows, dtype=float)
        t_idx = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-8)

        t_3d  = t_idx[:, np.newaxis, np.newaxis]
        stats_mean = np.nanmean(roll_stats, axis=0, keepdims=True)
        stats_std  = np.nanstd(roll_stats,  axis=0, keepdims=True) + 1e-8
        stats_norm = (roll_stats - stats_mean) / stats_std

        corr = np.nanmean(t_3d * stats_norm, axis=0)
        return corr  # Pearson ≈ Spearman ≈ Kendall tau for monotone trends

    print("    Computing rolling variance trend...")
    tau_var = rolling_kendall_vectorized(ndvi_dt, WINDOW)

    # AC1 rolling
    def rolling_ac1_vectorized(arr_3d, window):
        n = arr_3d.shape[0]
        n_windows = n - window + 1
        roll_ac1 = np.full((n_windows, arr_3d.shape[1], arr_3d.shape[2]), np.nan)

        for i in range(n_windows):
            chunk = arr_3d[i:i+window]
            # Lag-1 correlation
            c1 = chunk[:-1]
            c2 = chunk[1:]
            mn = np.nanmean(chunk, axis=0, keepdims=True)
            c1c = c1 - mn
            c2c = c2 - mn
            num = np.nanmean(c1c * c2c, axis=0)
            den = np.nanvar(chunk, axis=0) + 1e-8
            roll_ac1[i] = num / den

        t_idx  = np.arange(n_windows, dtype=float)
        t_idx  = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-8)
        t_3d   = t_idx[:, np.newaxis, np.newaxis]
        ac_mean = np.nanmean(roll_ac1, axis=0, keepdims=True)
        ac_std  = np.nanstd(roll_ac1,  axis=0, keepdims=True) + 1e-8
        ac_norm = (roll_ac1 - ac_mean) / ac_std
        corr    = np.nanmean(t_3d * ac_norm, axis=0)
        return corr

    print("    Computing rolling AC1 trend...")
    tau_ac1 = rolling_ac1_vectorized(ndvi_dt, WINDOW)

    # ── Composite risk score ───────────────────────────────────────────────
    # Normalize tau to [0,1] — positive tau = rising = higher risk
    risk_var = np.clip((tau_var + 1) / 2, 0, 1)
    risk_ac1 = np.clip((tau_ac1 + 1) / 2, 0, 1)
    risk_grid = 0.5 * risk_var + 0.5 * risk_ac1

    # Mask pixels with no valid NDVI data
    no_data = np.all(np.isnan(ndvi), axis=0)
    risk_grid[no_data] = np.nan

    # NDVI trend
    trend_grid = slopes.copy()
    trend_grid[no_data] = np.nan

    # Year of minimum NDVI
    ndvi_filled_masked = np.where(no_data[np.newaxis, :, :], np.inf, ndvi_filled)
    peak_idx   = np.argmin(ndvi_filled_masked, axis=0)
    peak_grid  = years[peak_idx].astype(float)
    peak_grid[no_data] = np.nan

    print(f"    Risk range: {np.nanmin(risk_grid):.3f} – {np.nanmax(risk_grid):.3f}")
    return risk_grid, trend_grid, peak_grid, lats, lons


def save_risk_maps(area_name, risk_grid, trend_grid, peak_grid, lats, lons):
    """Save risk maps as NetCDF."""
    ds_risk = xr.Dataset(
        {
            "collapse_risk":  (["lat", "lon"], risk_grid.astype(np.float32)),
            "ndvi_trend":     (["lat", "lon"], trend_grid.astype(np.float32)),
            "peak_stress_yr": (["lat", "lon"], peak_grid.astype(np.float32)),
        },
        coords={"lat": lats, "lon": lons},
        attrs={
            "study_area":    area_name,
            "description":   "Ecosystem collapse risk scores from EWS analysis",
            "risk_metric":   "Composite EWS score (variance + AC1 Kendall tau)",
            "ndvi_trend":    "Monthly NDVI trend slope (negative = greeness decline)",
            "peak_stress_yr": "Year of minimum NDVI (maximum stress)",
        }
    )

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
                for v in ds_risk.data_vars}
    out_path = OUT_DIR / f"risk_map_{area_name}.nc"
    ds_risk.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {out_path.name} ({size_mb:.1f} MB)")
    return ds_risk


def identify_high_risk_clusters(risk_grid, lats, lons, area_name):
    """Find contiguous spatial clusters of high-risk pixels."""
    binary = (risk_grid >= HIGH_RISK_THRESHOLD).astype(int)
    binary = np.where(np.isfinite(risk_grid), binary, 0)

    labeled, n_clusters = ndimage_label(binary)

    clusters = []
    for cid in range(1, n_clusters + 1):
        mask = labeled == cid
        size = mask.sum()
        if size < MIN_CLUSTER_SIZE:
            continue

        lat_idx, lon_idx = np.where(mask)
        clusters.append({
            "area":          area_name,
            "cluster_id":    cid,
            "n_pixels":      size,
            "mean_risk":     float(np.nanmean(risk_grid[mask])),
            "max_risk":      float(np.nanmax(risk_grid[mask])),
            "lat_center":    float(lats[lat_idx].mean()),
            "lon_center":    float(lons[lon_idx].mean()),
            "lat_min":       float(lats[lat_idx].min()),
            "lat_max":       float(lats[lat_idx].max()),
            "lon_min":       float(lons[lon_idx].min()),
            "lon_max":       float(lons[lon_idx].max()),
            "area_deg2":     float(size * 0.05 * 0.05),
        })

    df_clusters = pd.DataFrame(clusters).sort_values("mean_risk", ascending=False)
    out_csv = OUT_DIR / f"high_risk_clusters_{area_name}.csv"
    df_clusters.to_csv(out_csv, index=False)
    print(f"  [SAVED] {out_csv.name} — {len(df_clusters)} clusters found")
    return df_clusters


def process_area(area_name):
    print(f"\n{'='*60}")
    print(f"Spatial Risk Mapping — {area_name}")
    print(f"{'='*60}")

    # Check if deep learning predictions exist
    pred_path = DL_DIR / f"predictions_{area_name}.csv"

    if pred_path.exists():
        print("  Using deep learning predictions...")
        df_pred = pd.read_csv(pred_path, parse_dates=["time"])
        # Aggregate to spatial grid via region metadata
        region_meta_path = BASE_DIR / "data" / "processed" / "merged" / \
                           f"region_metadata_{area_name}.csv"
        if region_meta_path.exists():
            df_meta = pd.read_csv(region_meta_path)
            # Get max risk per region over last 12 months
            df_recent = df_pred[df_pred["time"] >= df_pred["time"].max() - pd.DateOffset(months=12)]
            region_risk = df_recent.groupby("region_id")["risk_score"].max().reset_index()
            df_meta = df_meta.merge(region_risk, on="region_id", how="left")
            print(f"  Region-level risk scores computed ({len(df_meta)} regions)")
            df_meta.to_csv(OUT_DIR / f"region_risk_{area_name}.csv", index=False)

    # Build pixel-level risk map (classical EWS — works without trained model)
    risk_grid, trend_grid, peak_grid, lats, lons = build_risk_map_classical(area_name)

    # Save maps
    ds_risk = save_risk_maps(area_name, risk_grid, trend_grid, peak_grid, lats, lons)

    # Identify high-risk clusters
    df_clusters = identify_high_risk_clusters(risk_grid, lats, lons, area_name)

    # Summary statistics
    valid_risk = risk_grid[np.isfinite(risk_grid)]
    print(f"\n  Risk statistics for {area_name}:")
    print(f"    Mean risk score:   {np.nanmean(valid_risk):.3f}")
    print(f"    High-risk pixels:  {(valid_risk >= HIGH_RISK_THRESHOLD).sum()} "
          f"({100*(valid_risk >= HIGH_RISK_THRESHOLD).mean():.1f}%)")
    print(f"    High-risk clusters: {len(df_clusters)}")
    if len(df_clusters) > 0:
        print(f"    Largest cluster:   {df_clusters['n_pixels'].max()} pixels "
              f"(risk={df_clusters.iloc[0]['mean_risk']:.3f})")

    return ds_risk


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if __import__("importlib").util.find_spec("torch") else None

    print("=" * 60)
    print("Spatial Collapse Risk Mapping")
    print(f"High-risk threshold: {HIGH_RISK_THRESHOLD}")
    print("=" * 60)

    for area in STUDY_AREAS:
        process_area(area)

    print(f"\nRisk mapping complete. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
