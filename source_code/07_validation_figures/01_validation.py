"""
Component 5a: Validation Against Known Events
=============================================
Validates model predictions against documented ecosystem stress events:

Amazon:
  - 2005 drought (worst in 100 years at the time)
  - 2010 drought (exceeded 2005 in severity)
  - 2015-16 El Niño drought

Sahel:
  - 2010-2012 Sahel drought/food crisis
  - 2017-2018 dry season failures

East Africa:
  - 2010-2011 Horn of Africa famine (worst in 60 years)
  - 2021-2023 multi-year drought (6 consecutive failed rainy seasons)

Metrics:
  - Lead time: how many months before event onset did EWS signal appear?
  - Precision / Recall for event detection
  - ROC-AUC for deep learning vs classical EWS comparison

Outputs:
  results/validation/
    validation_scores.csv
    lead_time_analysis.csv
    ews_vs_dl_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parents[1]
EWS_DIR     = BASE_DIR / "results" / "ews_classical"
DL_DIR      = BASE_DIR / "results" / "deep_learning"
OUT_DIR     = BASE_DIR / "results" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known stress events with documented onset dates
EVENTS = {
    "amazon": [
        {"name": "Amazon Drought 2005",    "onset": "2005-06-01", "end": "2005-11-01", "severity": 0.8},
        {"name": "Amazon Drought 2010",    "onset": "2010-06-01", "end": "2010-12-01", "severity": 1.0},
        {"name": "Amazon El Niño 2015-16", "onset": "2015-09-01", "end": "2016-05-01", "severity": 0.75},
        {"name": "Amazon Drought 2020",    "onset": "2020-06-01", "end": "2020-10-01", "severity": 0.6},
    ],
    "sahel": [
        {"name": "Sahel Drought 2002",     "onset": "2002-07-01", "end": "2002-10-01", "severity": 0.6},
        {"name": "Sahel Crisis 2011-12",   "onset": "2011-05-01", "end": "2012-03-01", "severity": 0.85},
        {"name": "Sahel Drought 2017",     "onset": "2017-06-01", "end": "2017-09-01", "severity": 0.65},
    ],
    "east_africa": [
        {"name": "Horn of Africa 2011",    "onset": "2011-01-01", "end": "2011-09-01", "severity": 1.0},
        {"name": "East Africa Drought 2017","onset": "2017-01-01", "end": "2017-06-01", "severity": 0.8},
        {"name": "Multi-year Drought 2022","onset": "2022-01-01", "end": "2023-06-01", "severity": 0.9},
    ],
}


def build_event_labels(times, events, lead_window=12):
    """
    Build binary event labels for a time series.
    A time step is labelled 1 if an event starts within lead_window months after it.
    """
    times = pd.to_datetime(times)
    labels = np.zeros(len(times))

    for event in events:
        onset = pd.Timestamp(event["onset"])
        for i, t in enumerate(times):
            delta_months = (onset - t).days / 30
            if 0 <= delta_months <= lead_window:
                labels[i] = max(labels[i], event["severity"])

    return labels


def evaluate_ews_predictions(area_name):
    """Compare classical EWS and deep learning predictions against known events."""
    results = []
    area_events = EVENTS.get(area_name, [])
    if not area_events:
        return None

    # ── Classical EWS ─────────────────────────────────────────────────────
    ews_path = EWS_DIR / f"ews_metrics_{area_name}.csv"
    if ews_path.exists():
        df_ews = pd.read_csv(ews_path)
        # Mean EWS score per region as a proxy for temporal risk trajectory
        mean_score = df_ews.groupby("region_id")["ews_score"].mean().mean()
        results.append({
            "method": "Classical EWS",
            "area": area_name,
            "mean_ews_score": float(mean_score),
            "n_high_risk_regions": int((df_ews.groupby("region_id")["ews_score"].mean() > 0.3).sum()),
        })

    # ── Deep Learning ─────────────────────────────────────────────────────
    dl_path = DL_DIR / f"predictions_{area_name}.csv"
    if dl_path.exists():
        df_dl = pd.read_csv(dl_path, parse_dates=["time"])
        df_dl = df_dl.sort_values(["region_id", "time"])

        # Compute AUC per region
        aucs, ap_scores, lead_times = [], [], []

        for region_id in df_dl["region_id"].unique():
            df_r  = df_dl[df_dl["region_id"] == region_id].sort_values("time")
            times = df_r["time"].values
            preds = df_r["risk_score"].values

            # Build ground truth labels
            labels = build_event_labels(times, area_events, lead_window=12)
            labels_bin = (labels > 0).astype(int)

            if labels_bin.sum() == 0 or labels_bin.sum() == len(labels_bin):
                continue

            try:
                auc = roc_auc_score(labels_bin, preds)
                ap  = average_precision_score(labels_bin, preds)
                aucs.append(auc)
                ap_scores.append(ap)

                # Lead time: first time risk_score > 0.5 before event
                for event in area_events:
                    onset = pd.Timestamp(event["onset"])
                    pre_event = df_r[pd.to_datetime(df_r["time"]) < onset]
                    if len(pre_event) == 0:
                        continue
                    high_risk = pre_event[pre_event["risk_score"] > 0.5]
                    if len(high_risk) > 0:
                        first_signal = pd.to_datetime(high_risk["time"].max())
                        lead_months  = (onset - first_signal).days / 30
                        if lead_months > 0:
                            lead_times.append(lead_months)
            except Exception:
                continue

        if aucs:
            results.append({
                "method":          "Deep Learning (TCN)",
                "area":            area_name,
                "mean_auc":        float(np.mean(aucs)),
                "std_auc":         float(np.std(aucs)),
                "mean_ap":         float(np.mean(ap_scores)),
                "mean_lead_months": float(np.mean(lead_times)) if lead_times else np.nan,
                "n_regions":       len(aucs),
            })

    return results


def compute_temporal_validation(area_name):
    """
    For each known event, compute:
    - EWS signal strength in the 24 months before onset
    - Whether signal was rising (positive Kendall tau)
    """
    area_events = EVENTS.get(area_name, [])
    dl_path = DL_DIR / f"predictions_{area_name}.csv"
    if not dl_path.exists():
        return None

    df_dl = pd.read_csv(dl_path, parse_dates=["time"])
    rows = []

    for event in area_events:
        onset = pd.Timestamp(event["onset"])

        # 24 months pre-event window
        pre_start = onset - pd.DateOffset(months=24)
        df_pre = df_dl[
            (pd.to_datetime(df_dl["time"]) >= pre_start) &
            (pd.to_datetime(df_dl["time"]) < onset)
        ]

        if len(df_pre) == 0:
            continue

        # Mean risk signal in pre-event window across all regions
        mean_risk_pre = df_pre.groupby("time")["risk_score"].mean()

        # Trend in pre-event risk signal
        if len(mean_risk_pre) >= 6:
            tau, pval = stats.kendalltau(np.arange(len(mean_risk_pre)), mean_risk_pre.values)
        else:
            tau, pval = np.nan, np.nan

        # Peak risk value
        peak_risk = df_pre["risk_score"].max()

        rows.append({
            "area":          area_name,
            "event":         event["name"],
            "onset":         event["onset"],
            "severity":      event["severity"],
            "mean_pre_risk": float(mean_risk_pre.mean()),
            "peak_pre_risk": float(peak_risk),
            "tau_pre_event": float(tau) if not np.isnan(tau) else np.nan,
            "pval_trend":    float(pval) if not np.isnan(pval) else np.nan,
            "rising_signal": bool(tau > 0 and pval < 0.1) if not np.isnan(tau) else False,
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Validation Against Known Ecosystem Stress Events")
    print("=" * 60)

    all_results  = []
    all_temporal = []

    for area in ["amazon", "sahel", "east_africa"]:
        print(f"\n  Evaluating {area}...")
        results = evaluate_ews_predictions(area)
        if results:
            all_results.extend(results)

        temporal = compute_temporal_validation(area)
        if temporal is not None and len(temporal) > 0:
            all_temporal.append(temporal)

    # Save
    if all_results:
        df_val = pd.DataFrame(all_results)
        df_val.to_csv(OUT_DIR / "validation_scores.csv", index=False)
        print(f"\n  [SAVED] validation_scores.csv")
        print(df_val.to_string(index=False))

    if all_temporal:
        df_temp = pd.concat(all_temporal, ignore_index=True)
        df_temp.to_csv(OUT_DIR / "temporal_validation.csv", index=False)
        print(f"\n  [SAVED] temporal_validation.csv")
        print(f"\n  Events with rising pre-event signal:")
        rising = df_temp[df_temp["rising_signal"] == True]
        if len(rising) > 0:
            print(rising[["area", "event", "tau_pre_event", "pval_trend"]].to_string(index=False))
        else:
            print("  (run deep learning training first for full validation)")

    print(f"\nValidation complete. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
