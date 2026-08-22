"""
Baseline Models — LOEO Evaluation (Reviewer Fix)
=================================================
Leave-One-Event-Out (LOEO) cross-validation for:
  1. Logistic Regression on EWS features
  2. Gradient Boosting (XGBoost-style) on EWS features
  3. Simple LSTM baseline

Val set per fold = positive windows from held-out event + 3x random negatives.
Train set = all remaining windows.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR    = BASE_DIR / "results" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN      = 60
FEATURE_COLS = ["NDVI","EVI","LST_Day","ET","soil_moisture",
                "swvl1","swvl1_anom","t2m","t2m_anom"]

STRESS_EVENTS = {
    "amazon": [
        {"name":"Amazon 2005","onset":"2005-06-01","severity":0.8},
        {"name":"Amazon 2010","onset":"2010-06-01","severity":1.0},
        {"name":"Amazon 2015","onset":"2015-09-01","severity":0.75},
        {"name":"Amazon 2020","onset":"2020-06-01","severity":0.6},
    ],
    "sahel": [
        {"name":"Sahel 2011","onset":"2011-05-01","severity":0.85},
        {"name":"Sahel 2017","onset":"2017-06-01","severity":0.65},
    ],
    "east_africa": [
        {"name":"HoA 2011","onset":"2011-01-01","severity":1.0},
        {"name":"EA 2017","onset":"2017-01-01","severity":0.8},
        {"name":"EA 2022","onset":"2022-01-01","severity":0.9},
    ],
}


def make_labels(times, events, horizon=12):
    times  = pd.to_datetime(times)
    labels = np.zeros(len(times))
    for e in events:
        onset = pd.Timestamp(e["onset"])
        for i, t in enumerate(times):
            d = (onset - t).days / 30
            if 0 <= d <= horizon:
                labels[i] = max(labels[i], e["severity"])
    return labels


def extract_ews_features(series, window=36):
    ts  = pd.Series(series).fillna(pd.Series(series).mean()).values
    if len(ts) < window:
        return [0.0] * 5
    var  = float(np.nanvar(ts[-window:]))
    ac1  = float(np.corrcoef(ts[-window:-1], ts[-window+1:])[0,1]) if window > 1 else 0.0
    skew = float(stats.skew(ts[-window:]))
    tau, _ = stats.kendalltau(np.arange(window), ts[-window:])
    mean_v = float(np.nanmean(ts[-window:]))
    return [var, ac1, skew, float(tau), mean_v]


def load_data():
    seqs, ews_feats, y_bin, event_names = [], [], [], []
    for area in ["amazon","sahel","east_africa"]:
        csv = MERGED_DIR / f"merged_regions_{area}.csv"
        df  = pd.read_csv(csv, parse_dates=["time"])
        df  = df.sort_values(["region_id","time"])
        events = STRESS_EVENTS[area]
        for rid in df.region_id.unique():
            dr    = df[df.region_id==rid].sort_values("time")
            feats = dr[FEATURE_COLS].copy()
            means = feats.mean().fillna(0.0)
            fa    = feats.fillna(means).replace([np.inf,-np.inf],0.0).fillna(0.0).values.astype(np.float32)
            ts    = dr["time"].values
            lbls  = make_labels(ts, events)
            for i in range(SEQ_LEN, len(fa)):
                win = fa[i-SEQ_LEN:i]
                seqs.append(win)
                ef = []
                for fi in range(win.shape[1]):
                    ef.extend(extract_ews_features(win[:, fi]))
                ews_feats.append(ef)
                y_bin.append(int(lbls[i] >= 0.3))
                tag = "none"
                for e in events:
                    d = (pd.Timestamp(e["onset"]) - pd.Timestamp(ts[i])).days / 30
                    if 0 <= d <= 12:
                        tag = e["name"]
                        break
                event_names.append(tag)
    return (np.array(seqs, np.float32),
            np.array(ews_feats, np.float32),
            np.array(y_bin, np.float32),
            event_names)


class SimpleLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:,-1,:]).squeeze(1)


def train_lstm(X_tr, y_tr, device, epochs=60):
    sc  = StandardScaler()
    Xf  = np.nan_to_num(sc.fit_transform(X_tr.reshape(-1, X_tr.shape[-1]))).reshape(X_tr.shape).astype(np.float32)
    Xt  = torch.FloatTensor(Xf).to(device)
    yt  = torch.FloatTensor(y_tr).to(device)
    pw  = float((y_tr==0).sum() / max((y_tr==1).sum(), 1))
    sw  = np.where(y_tr==1, pw, 1.0)
    samp = torch.utils.data.WeightedRandomSampler(torch.FloatTensor(sw), len(y_tr), True)
    dl  = DataLoader(TensorDataset(Xt, yt), batch_size=64, sampler=samp)
    m   = SimpleLSTM(X_tr.shape[-1]).to(device)
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.BCELoss()
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m, sc


def evaluate_model(name, y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return None
    return {
        "model":          name,
        "AUC_ROC":        float(roc_auc_score(y_true, y_pred)),
        "Avg_Precision":  float(average_precision_score(y_true, y_pred)),
        "Brier":          float(brier_score_loss(y_true, np.clip(y_pred, 0, 1))),
    }


def loeo_evaluation(X_seq, X_ews, y, event_names):
    """
    LOEO: val = held-out event positives + 3x random negatives.
    Train = everything else.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    event_arr  = np.array(event_names)
    none_idx   = np.where(event_arr == "none")[0]
    rng        = np.random.default_rng(42)
    events_uniq = [e for e in sorted(set(event_names)) if e != "none"]

    all_rows = []

    for held_out in events_uniq:
        pos_idx = np.where(event_arr == held_out)[0]
        if len(pos_idx) < 5:
            print(f"  [SKIP] {held_out} — only {len(pos_idx)} positive windows")
            continue

        # Build val set: positives + 3x negatives
        n_neg   = min(len(pos_idx) * 3, len(none_idx))
        neg_val = rng.choice(none_idx, size=n_neg, replace=False)
        val_idx = np.concatenate([pos_idx, neg_val])

        # Train: everything not in val
        val_set    = set(val_idx.tolist())
        train_idx  = np.array([i for i in range(len(y)) if i not in val_set])

        y_va = y[val_idx]
        y_tr = y[train_idx]

        if len(np.unique(y_va)) < 2 or len(train_idx) < 100:
            print(f"  [SKIP] {held_out} — insufficient class balance")
            continue

        print(f"\n  LOEO [{held_out:<25}] "
              f"train:{len(train_idx)} val:{len(val_idx)} pos:{int(y_va.sum())}")

        sc = StandardScaler()
        Xe_tr = np.nan_to_num(sc.fit_transform(X_ews[train_idx]))
        Xe_va = np.nan_to_num(sc.transform(X_ews[val_idx]))

        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1)
        lr.fit(Xe_tr, y_tr)
        r1 = evaluate_model("LogReg", y_va, lr.predict_proba(Xe_va)[:, 1])

        # 2. Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                        learning_rate=0.05, subsample=0.8)
        gb.fit(Xe_tr, y_tr)
        r2 = evaluate_model("GradBoost", y_va, gb.predict_proba(Xe_va)[:, 1])

        # 3. LSTM
        lstm, sc_lstm = train_lstm(X_seq[train_idx], y_tr, device)
        lstm.eval()
        Xsf = np.nan_to_num(sc_lstm.transform(
            X_seq[val_idx].reshape(-1, len(FEATURE_COLS))
        ).reshape(X_seq[val_idx].shape).astype(np.float32))
        with torch.no_grad():
            pred_lstm = lstm(torch.FloatTensor(Xsf).to(device)).cpu().numpy()
        r3 = evaluate_model("LSTM", y_va, np.nan_to_num(pred_lstm, nan=0.5))

        for r in [r1, r2, r3]:
            if r:
                r["held_out_event"] = held_out
                all_rows.append(r)
                print(f"    {r['model']:<12} AUC={r['AUC_ROC']:.3f} "
                      f"AP={r['Avg_Precision']:.3f} Brier={r['Brier']:.3f}")

    return pd.DataFrame(all_rows)


def main():
    print("="*60)
    print("Baseline Models — Leave-One-Event-Out Evaluation")
    print("="*60)

    print("\nLoading data...")
    X_seq, X_ews, y, event_names = load_data()
    print(f"  Total: {len(y)} windows | Positive: {int(y.sum())} ({100*y.mean():.1f}%)")

    from collections import Counter
    ec = Counter(event_names)
    print("  Event counts:", {k: v for k, v in ec.items() if k != "none"})

    print("\nRunning LOEO evaluation...")
    df = loeo_evaluation(X_seq, X_ews, y, event_names)

    if len(df) > 0:
        df.to_csv(OUT_DIR / "loeo_baselines.csv", index=False)
        print(f"\n  [SAVED] loeo_baselines.csv")
        print("\n  Mean LOEO performance per model:")
        print(df.groupby("model")[["AUC_ROC","Avg_Precision","Brier"]]
               .agg(["mean","std"]).round(3).to_string())
    else:
        print("  [WARN] No valid LOEO folds produced.")

    print(f"\nDone. Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
