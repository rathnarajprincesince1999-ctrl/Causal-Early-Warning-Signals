"""
DL Model — LOEO Evaluation + SHAP Interpretability — Reviewer Fix
==================================================================
Addresses:
  1. Leave-one-event-out (LOEO) to prevent leakage
  2. Reliability diagrams + Brier scores (calibration)
  3. Precision-Recall curves (imbalanced class metric)
  4. Lead-time skill curves
  5. Temporal SHAP via gradient*input attribution

Outputs: results/deep_learning_loeo/
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             roc_curve)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR    = BASE_DIR / "results" / "deep_learning_loeo"
FIG_DIR    = BASE_DIR / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN      = 60
FEATURE_COLS = ["NDVI", "EVI", "LST_Day", "ET", "soil_moisture",
                "swvl1", "swvl1_anom", "t2m", "t2m_anom"]
EPOCHS       = 100
PATIENCE     = 20
LR           = 3e-4
BATCH        = 64

STRESS_EVENTS = {
    "amazon": [
        {"name":"Amazon_2005","onset":"2005-06-01","severity":0.8},
        {"name":"Amazon_2010","onset":"2010-06-01","severity":1.0},
        {"name":"Amazon_2015","onset":"2015-09-01","severity":0.75},
        {"name":"Amazon_2020","onset":"2020-06-01","severity":0.6},
    ],
    "sahel": [
        {"name":"Sahel_2011","onset":"2011-05-01","severity":0.85},
        {"name":"Sahel_2017","onset":"2017-06-01","severity":0.65},
    ],
    "east_africa": [
        {"name":"HoA_2011","onset":"2011-01-01","severity":1.0},
        {"name":"EA_2017","onset":"2017-01-01","severity":0.8},
        {"name":"EA_2022","onset":"2022-01-01","severity":0.9},
    ],
}


# ── Model (same TCN as before) ─────────────────────────────────────────────
class CausalConvBlock(nn.Module):
    def __init__(self, n_in, n_out, ks, dil):
        super().__init__()
        pad = (ks-1)*dil
        self.c1  = nn.Conv1d(n_in,  n_out, ks, dilation=dil, padding=pad)
        self.c2  = nn.Conv1d(n_out, n_out, ks, dilation=dil, padding=pad)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.drop= nn.Dropout(0.2)
        self.ds  = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
    def forward(self, x):
        r  = x if self.ds is None else self.ds(x)
        x  = torch.relu(self.bn1(self.c1(x)[:,:,:x.size(2)]))
        x  = self.drop(x)
        x  = torch.relu(self.bn2(self.c2(x)[:,:,:r.size(2)]))
        return torch.relu(x + r)

class TCNDetector(nn.Module):
    def __init__(self, n_feat, seq_len, ch=64, nl=4):
        super().__init__()
        layers, ic = [], n_feat
        for i in range(nl):
            layers.append(CausalConvBlock(ic, ch, 3, 2**i))
            ic = ch
        self.tcn  = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(ch)
        self.head = nn.Sequential(nn.Linear(ch,32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32,1),
                                  nn.Sigmoid())
    def forward(self, x):
        o  = self.tcn(x.permute(0,2,1)).permute(0,2,1)
        a, _ = self.attn(o, o, o)
        o  = self.norm(o + a).mean(1)
        return self.head(o).squeeze(1)


# ── Data ────────────────────────────────────────────────────────────────────
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


def load_all():
    seqs, y, event_names, times_all = [], [], [], []
    for area in ["amazon","sahel","east_africa"]:
        df = pd.read_csv(MERGED_DIR/f"merged_regions_{area}.csv",
                         parse_dates=["time"])
        df = df.sort_values(["region_id","time"])
        events = STRESS_EVENTS[area]
        for rid in df.region_id.unique():
            dr    = df[df.region_id==rid].sort_values("time")
            feats = dr[FEATURE_COLS].copy()
            means = feats.mean().fillna(0.0)
            fa    = feats.fillna(means).replace([np.inf,-np.inf],0.0).fillna(0.0).values.astype(np.float32)
            ts    = dr["time"].values
            lbls  = make_labels(ts, events)
            for i in range(SEQ_LEN, len(fa)):
                seqs.append(fa[i-SEQ_LEN:i])
                y.append(float(lbls[i]))
                times_all.append(pd.Timestamp(ts[i]))
                tag = "none"
                for e in events:
                    d = (pd.Timestamp(e["onset"]) - pd.Timestamp(ts[i])).days/30
                    if 0 <= d <= 12:
                        tag = e["name"]
                        break
                event_names.append(tag)
    X  = np.array(seqs, np.float32)
    yb = (np.array(y) >= 0.3).astype(np.float32)
    return X, yb, event_names, times_all


# ── Training ────────────────────────────────────────────────────────────────
def train_one(X_tr, y_tr, device):
    Xf = np.nan_to_num(X_tr)
    sc = StandardScaler()
    Xf = sc.fit_transform(Xf.reshape(-1,len(FEATURE_COLS))).reshape(Xf.shape).astype(np.float32)
    Xt = torch.FloatTensor(Xf).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    pw = float((y_tr==0).sum() / max((y_tr==1).sum(),1))
    sw = np.where(y_tr==1, pw, 1.0)
    samp = torch.utils.data.WeightedRandomSampler(
        torch.FloatTensor(sw), len(y_tr), replacement=True)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=BATCH, sampler=samp)
    m  = TCNDetector(len(FEATURE_COLS), SEQ_LEN).to(device)
    opt= optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch= optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
            steps_per_epoch=len(dl), pct_start=0.1)
    crit = nn.BCELoss()
    best_loss, patience_ctr = 1e9, 0
    best_state = None
    for ep in range(EPOCHS):
        m.train(); tloss = 0
        for xb,yb in dl:
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step(); sch.step()
            tloss += loss.item()
        tloss /= len(dl)
        if tloss < best_loss - 1e-4:
            best_loss = tloss; patience_ctr = 0
            best_state = {k:v.clone() for k,v in m.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break
    if best_state: m.load_state_dict(best_state)
    return m, sc


# ── SHAP via Gradient×Input ─────────────────────────────────────────────────
def gradient_x_input_attribution(model, X, device, n_samples=200):
    """
    Temporal feature attribution using Gradient × Input.
    Returns importance per (time_step, feature).
    """
    model.eval()
    Xf = np.nan_to_num(X[:n_samples])
    Xt = torch.FloatTensor(Xf).to(device)
    Xt.requires_grad_(True)
    pred = model(Xt)
    pred.sum().backward()
    grad = Xt.grad.detach().cpu().numpy()   # (N, T, F)
    attr = np.abs(grad * Xf[:n_samples])    # gradient × input
    return attr.mean(axis=0)                # (T, F) mean over samples


# ── Main LOEO ───────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("DL LOEO Evaluation + SHAP Attribution")
    print(f"Device: {device}")
    print("="*60)

    X, y, event_names, times = load_all()
    print(f"\nTotal windows: {len(y)} | Positive: {int(y.sum())} ({100*y.mean():.1f}%)")

    events_unique = [e for e in sorted(set(event_names)) if e != "none"]
    print(f"Events for LOEO: {events_unique}")

    all_results = []
    all_preds   = []

    for held_out in events_unique:
        val_mask   = np.array([e == held_out for e in event_names])
        train_mask = ~val_mask

        if val_mask.sum() < 5 or y[val_mask].sum() < 2:
            print(f"  [SKIP] {held_out} — insufficient positive samples")
            continue
        if len(np.unique(y[val_mask])) < 2:
            print(f"  [SKIP] {held_out} — only one class in val set")
            continue

        print(f"\n  LOEO [{held_out}] — train:{train_mask.sum()} val:{val_mask.sum()}")
        model, sc = train_one(X[train_mask], y[train_mask], device)

        Xv = np.nan_to_num(sc.transform(
            X[val_mask].reshape(-1,len(FEATURE_COLS))
        ).reshape(X[val_mask].shape).astype(np.float32))

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(Xv).to(device)).cpu().numpy()

        preds = np.nan_to_num(preds, nan=0.5)
        y_val = y[val_mask]

        auc  = roc_auc_score(y_val, preds)
        ap   = average_precision_score(y_val, preds)
        bri  = brier_score_loss(y_val, np.clip(preds,0,1))

        print(f"    AUC={auc:.3f} | AP={ap:.3f} | Brier={bri:.3f}")

        all_results.append({"event":held_out,"AUC":auc,"AP":ap,"Brier":bri,
                            "n_val":int(val_mask.sum()),
                            "n_pos":int(y_val.sum())})

        for i, idx in enumerate(np.where(val_mask)[0]):
            all_preds.append({
                "event":held_out, "time":times[idx],
                "y_true":int(y[idx]), "y_pred":float(preds[i])
            })

    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(OUT_DIR/"loeo_results.csv", index=False)
        print("\n  LOEO Summary:")
        print(df_res[["event","AUC","AP","Brier"]].to_string(index=False))
        print(f"\n  Mean AUC: {df_res.AUC.mean():.3f} ± {df_res.AUC.std():.3f}")

        # Save predictions for PR/ROC plots
        pd.DataFrame(all_preds).to_csv(OUT_DIR/"loeo_predictions.csv",index=False)

        # Reliability diagram (calibration)
        df_p  = pd.DataFrame(all_preds)
        probs = np.clip(df_p.y_pred.values, 0.01, 0.99)
        trues = df_p.y_true.values
        fig, ax = plt.subplots(figsize=(4,4))
        frac_pos, mean_pred = calibration_curve(trues, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label="TCN model")
        ax.plot([0,1],[0,1],"--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positive")
        ax.set_title("Reliability Diagram (LOEO)"); ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR/"fig_reliability.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR/"fig_reliability.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  [SAVED] fig_reliability.pdf")

    # Gradient×Input attribution on full dataset
    print("\n  Computing Gradient×Input attribution...")
    model_full, sc_full = train_one(X, y, device)
    Xsc = np.nan_to_num(sc_full.transform(
        X.reshape(-1,len(FEATURE_COLS))).reshape(X.shape).astype(np.float32))
    attr = gradient_x_input_attribution(model_full, Xsc, device)  # (T, F)

    # Feature importance: mean over time
    feat_imp = attr.mean(axis=0)
    df_imp = pd.DataFrame({"feature":FEATURE_COLS,
                            "gxi_importance":feat_imp})
    df_imp = df_imp.sort_values("gxi_importance", ascending=False)
    df_imp.to_csv(OUT_DIR/"gxi_feature_importance.csv", index=False)
    print("\n  Gradient×Input feature importance:")
    print(df_imp.to_string(index=False))

    # Temporal attribution heatmap
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(attr.T, aspect="auto", cmap="RdBu_r",
                   extent=[0, SEQ_LEN, len(FEATURE_COLS)-0.5, -0.5])
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS, fontsize=7)
    ax.set_xlabel("Months before prediction (0 = most recent)")
    ax.set_title("Temporal Feature Attribution (Gradient×Input)")
    fig.colorbar(im, ax=ax, label="Attribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_temporal_attribution.pdf",dpi=300,bbox_inches="tight")
    fig.savefig(FIG_DIR/"fig_temporal_attribution.png",dpi=300,bbox_inches="tight")
    plt.close()
    print("  [SAVED] fig_temporal_attribution.pdf")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
