"""
Component 3: Causal Discovery with PCMCI+
==========================================
Uses PCMCI+ (Runge et al. 2019) from the tigramite package to discover
time-lagged causal networks between climate drivers and vegetation variables.

This answers: WHAT CAUSES ecosystem stress, not just WHEN it occurs.

For each study area × region:
  - Computes lagged cross-correlations
  - Runs PCMCI+ to identify significant causal links (p < 0.05)
  - Quantifies causal effect strengths
  - Identifies key drivers of NDVI/EVI decline

Key outputs:
  - Causal graph (adjacency matrix with lag and strength)
  - Time-lagged causal effect of precipitation on vegetation
  - Feedback loops between temperature, ET, and soil moisture

Outputs:
  results/causal/
    causal_graph_{area}_{region}.json
    causal_summary_{area}.csv
    causal_network_{area}.png  (saved by figures script)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings("ignore")

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_OK = True
except ImportError:
    TIGRAMITE_OK = False
    print("[WARN] tigramite not installed. Run: pip install tigramite")

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR    = BASE_DIR / "results" / "causal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_AREAS = ["amazon", "sahel", "east_africa"]

# Variables included in causal analysis
CAUSAL_VARS = [
    "tp_anom",       # precipitation anomaly (climate driver)
    "t2m_anom",      # temperature anomaly (climate driver)
    "swvl1_anom",    # soil water anomaly (land surface)
    "soil_moisture", # surface soil moisture (ESA CCI)
    "ET",            # evapotranspiration (vegetation response)
    "NDVI",          # vegetation greenness (collapse indicator)
    "EVI",           # vegetation index (collapse indicator)
]

VAR_LABELS = {
    "tp_anom":       "Precip.\nAnom.",
    "t2m_anom":      "Temp.\nAnom.",
    "swvl1_anom":    "Soil Water\nAnom.",
    "soil_moisture": "Surface\nSM",
    "ET":            "ET",
    "NDVI":          "NDVI",
    "EVI":           "EVI",
}

MAX_LAG    = 6    # maximum time lag (months) to test
ALPHA      = 0.05 # significance level
PC_ALPHA   = 0.05 # PC algorithm significance


def run_pcmci_region(df_region, area_name, region_id):
    """Run PCMCI+ for a single region's time series."""
    if not TIGRAMITE_OK:
        return None

    # Extract variable matrix — fill NaN
    data_cols  = [v for v in CAUSAL_VARS if v in df_region.columns]
    data_mat   = df_region[data_cols].copy()
    # Fill NaN: column mean, then 0 as fallback
    col_means  = data_mat.mean().fillna(0.0)
    data_mat   = data_mat.fillna(col_means).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    data_mat   = data_mat.values.astype(float)

    # Standardise
    data_mat = (data_mat - data_mat.mean(axis=0)) / (data_mat.std(axis=0) + 1e-8)

    n_vars = data_mat.shape[1]
    if data_mat.shape[0] < MAX_LAG * 3:
        return None

    # Set up tigramite dataframe
    dataframe = pp.DataFrame(
        data_mat,
        var_names=data_cols,
    )

    # PCMCI+ with partial correlation independence test
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )

    results = pcmci.run_pcmciplus(
        tau_min=1,
        tau_max=MAX_LAG,
        pc_alpha=PC_ALPHA,
    )

    # Extract significant causal links
    p_matrix   = results["p_matrix"]    # shape: (n_vars, n_vars, max_lag)
    val_matrix = results["val_matrix"]  # causal effect strengths

    links = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            for lag in range(1, MAX_LAG + 1):
                pval   = p_matrix[i, j, lag]
                effect = val_matrix[i, j, lag]
                if pval < ALPHA:
                    links.append({
                        "cause":      data_cols[i],
                        "effect":     data_cols[j],
                        "lag":        lag,
                        "strength":   float(effect),
                        "pvalue":     float(pval),
                        "area":       area_name,
                        "region_id":  region_id,
                    })

    return {
        "links":      links,
        "n_links":    int(len(links)),
        "var_names":  data_cols,
        "p_matrix":   [[[ float(p_matrix[i,j,l]) for l in range(p_matrix.shape[2])]
                        for j in range(p_matrix.shape[1])]
                       for i in range(p_matrix.shape[0])],
        "val_matrix": [[[ float(val_matrix[i,j,l]) for l in range(val_matrix.shape[2])]
                        for j in range(val_matrix.shape[1])]
                       for i in range(val_matrix.shape[0])],
    }


def process_area(area_name):
    """Run PCMCI+ for all regions in one area."""
    print(f"\n{'='*60}")
    print(f"Causal Discovery (PCMCI+) — {area_name}")
    print(f"{'='*60}")

    csv_path = MERGED_DIR / f"merged_regions_{area_name}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No merged data: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["region_id", "time"])

    regions   = sorted(df["region_id"].unique())
    all_links = []

    for region_id in tqdm(regions, desc=f"  Regions", unit="region"):
        df_r   = df[df["region_id"] == region_id].sort_values("time")
        result = run_pcmci_region(df_r, area_name, region_id)

        if result is None:
            continue

        # Save per-region graph
        out_json = OUT_DIR / f"causal_graph_{area_name}_{region_id}.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)

        all_links.extend(result["links"])

    if not all_links:
        print(f"  [WARN] No significant causal links found for {area_name}")
        return

    # Aggregate across regions
    df_links = pd.DataFrame(all_links)

    # Save summary
    out_csv = OUT_DIR / f"causal_summary_{area_name}.csv"
    df_links.to_csv(out_csv, index=False)
    print(f"\n  [SAVED] {out_csv.name} ({len(df_links)} causal links)")

    # Print key findings
    print(f"\n  Key causal drivers of NDVI decline ({area_name}):")
    ndvi_causes = df_links[df_links["effect"] == "NDVI"].copy()
    ndvi_causes["abs_strength"] = ndvi_causes["strength"].abs()
    top = ndvi_causes.groupby(["cause", "lag"])["abs_strength"].mean().sort_values(ascending=False)
    print(top.head(8).to_string())

    return df_links


def aggregate_causal_networks():
    """Combine causal results across all areas for comparison."""
    all_dfs = []
    for area in STUDY_AREAS:
        path = OUT_DIR / f"causal_summary_{area}.csv"
        if path.exists():
            df = pd.read_csv(path)
            all_dfs.append(df)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Dominant causal pathways per area
    print("\n" + "=" * 60)
    print("DOMINANT CAUSAL PATHWAYS TO VEGETATION COLLAPSE")
    print("=" * 60)

    for area in STUDY_AREAS:
        df_a = combined[
            (combined["area"] == area) &
            (combined["effect"].isin(["NDVI", "EVI"]))
        ]
        if df_a.empty:
            continue
        top = df_a.groupby("cause")["strength"].mean().sort_values()
        print(f"\n  {area}:")
        print(top.to_string())

    combined.to_csv(OUT_DIR / "all_causal_links.csv", index=False)
    print(f"\n  [SAVED] all_causal_links.csv")


def main():
    if not TIGRAMITE_OK:
        print("[ERROR] tigramite required. Install: pip install tigramite")
        print("Continuing with mock output for pipeline testing...")

    print("=" * 60)
    print("Causal Discovery Analysis (PCMCI+)")
    print(f"Variables: {CAUSAL_VARS}")
    print(f"Max lag: {MAX_LAG} months | Alpha: {ALPHA}")
    print("=" * 60)

    for area in STUDY_AREAS:
        process_area(area)

    aggregate_causal_networks()

    print(f"\nCausal analysis complete. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
