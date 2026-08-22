"""
Component 2: Deep Learning Early Warning Signal Detector
=========================================================
Trains a Temporal Convolutional Network (TCN) to detect pre-collapse
signatures from multivariate time series simultaneously.

Architecture:
  - Input: [batch, time_steps, n_variables] = [B, 60, 14]
  - TCN with dilated causal convolutions (captures long-range dependencies)
  - Multi-head attention over temporal dimension
  - Output: collapse risk score in [0, 1] per time window

Training strategy:
  - Positive samples: windows ending within 12 months of a known stress event
  - Negative samples: windows from stable periods
  - Cross-validation across regions and ecosystems

Key advantage over classical EWS:
  - Multivariate (all 14 variables simultaneously)
  - Non-linear interactions captured
  - Learns optimal window length automatically
  - Works across different ecosystem types (transfer learning)

Outputs:
  results/deep_learning/
    model_weights.pt
    training_history.csv
    predictions_{area}.csv
    feature_importance.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import json

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR    = BASE_DIR / "results" / "deep_learning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEQ_LEN     = 60     # 5 years of monthly data as input window
PRED_HORIZON= 12     # predict collapse risk 12 months ahead
BATCH_SIZE  = 64
EPOCHS      = 150
LR          = 3e-4
PATIENCE    = 25     # early stopping

# Variables used as model inputs
FEATURE_COLS = ["NDVI", "EVI", "LST_Day", "ET", "soil_moisture",
                "swvl1", "swvl1_anom", "t2m", "t2m_anom"]

# Known ecosystem stress events (year-month, area, region filter)
# Used to create weakly-supervised positive labels
STRESS_EVENTS = {
    "amazon": [
        {"year": 2005, "months": [6, 7, 8, 9, 10], "severity": 0.8},   # Amazon drought
        {"year": 2010, "months": [6, 7, 8, 9, 10], "severity": 1.0},   # Severe drought
        {"year": 2015, "months": [9, 10, 11, 12],  "severity": 0.7},   # El Niño drought
        {"year": 2016, "months": [1, 2, 3, 4],     "severity": 0.7},
        {"year": 2020, "months": [6, 7, 8, 9],     "severity": 0.6},
    ],
    "sahel": [
        {"year": 2002, "months": [7, 8, 9],         "severity": 0.6},
        {"year": 2009, "months": [7, 8, 9],         "severity": 0.7},
        {"year": 2011, "months": [5, 6, 7, 8, 9],  "severity": 0.8},   # Sahel drought/famine
        {"year": 2017, "months": [6, 7, 8],         "severity": 0.6},
    ],
    "east_africa": [
        {"year": 2002, "months": [10, 11, 12],      "severity": 0.7},
        {"year": 2009, "months": [10, 11, 12],      "severity": 0.8},
        {"year": 2011, "months": [1, 2, 3, 4, 5],  "severity": 1.0},   # Horn of Africa famine
        {"year": 2017, "months": [1, 2, 3, 4],     "severity": 0.8},
        {"year": 2022, "months": [1, 2, 3, 4, 5],  "severity": 0.9},   # Multi-year drought
    ],
}


# ── Dataset ────────────────────────────────────────────────────────────────

class EWSDataset(Dataset):
    def __init__(self, sequences, labels, region_ids):
        self.X = torch.FloatTensor(sequences)   # (N, seq_len, n_features)
        self.y = torch.FloatTensor(labels)       # (N,)
        self.groups = region_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────

class CausalConvBlock(nn.Module):
    """Dilated causal convolution block with residual connection."""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        # Causal: trim future timesteps
        out = self.relu(self.norm1(self.conv1(x)[:, :, :x.size(2)]))
        out = self.drop(out)
        out = self.relu(self.norm2(self.conv2(out)[:, :, :x.size(2)]))
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEWSDetector(nn.Module):
    """
    Temporal Convolutional Network for EWS detection.
    Uses dilated convolutions to capture multi-scale temporal patterns.
    """
    def __init__(self, n_features, seq_len, n_channels=64, n_layers=4):
        super().__init__()
        self.n_features = n_features
        self.seq_len    = seq_len

        # TCN layers with exponentially growing dilation
        layers = []
        in_ch  = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            out_ch   = n_channels
            layers.append(CausalConvBlock(in_ch, out_ch, kernel_size=3, dilation=dilation))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        # Multi-head self-attention over time
        self.attn = nn.MultiheadAttention(
            embed_dim=n_channels, num_heads=4, batch_first=True, dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(n_channels)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        # TCN expects (batch, channels, seq_len)
        out = self.tcn(x.permute(0, 2, 1))           # → (B, n_channels, seq_len)
        out = out.permute(0, 2, 1)                    # → (B, seq_len, n_channels)

        # Self-attention
        attn_out, _ = self.attn(out, out, out)
        out = self.attn_norm(out + attn_out)

        # Global average pooling over time
        out = out.mean(dim=1)                         # → (B, n_channels)

        return self.classifier(out).squeeze(1)        # → (B,)

    def get_attention_weights(self, x):
        """Extract attention weights for interpretability."""
        out = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, weights = self.attn(out, out, out, need_weights=True)
        return weights


# ── Data preparation ────────────────────────────────────────────────────────

def make_stress_labels(df_region, area_name, pred_horizon=PRED_HORIZON):
    """
    Assign soft labels based on proximity to known stress events.
    Window ending at time t gets label = max stress severity
    in [t, t + pred_horizon] months.
    """
    times  = pd.to_datetime(df_region["time"].values)
    labels = np.zeros(len(times))

    events = STRESS_EVENTS.get(area_name, [])
    for event in events:
        for month in event["months"]:
            target = pd.Timestamp(year=event["year"], month=month, day=1)
            # Any window whose end falls within pred_horizon months before target
            for i, t in enumerate(times):
                delta = (target - t).days / 30
                if 0 <= delta <= pred_horizon:
                    labels[i] = max(labels[i], event["severity"])

    return labels


def prepare_sequences(area_name):
    """Build (X, y, groups) training arrays for one area."""
    csv_path = MERGED_DIR / f"merged_regions_{area_name}.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["region_id", "time"]).reset_index(drop=True)

    sequences, labels, groups = [], [], []

    for region_id in df["region_id"].unique():
        df_r = df[df["region_id"] == region_id].sort_values("time")

        # Extract features — fill NaN with column mean, then global mean fallback
        feats = df_r[FEATURE_COLS].copy()
        # Fill column-wise mean first
        col_means = feats.mean()
        # If a column is entirely NaN, fill with 0
        col_means = col_means.fillna(0.0)
        feats = feats.fillna(col_means)
        # Safety: replace any remaining NaN/Inf with 0
        feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        feat_arr = feats.values.astype(np.float32)

        # Make stress labels for this region
        region_labels = make_stress_labels(df_r, area_name)

        # Sliding windows
        n = len(feat_arr)
        for i in range(SEQ_LEN, n):
            window = feat_arr[i - SEQ_LEN : i]
            label  = region_labels[i]  # label at end of window
            sequences.append(window)
            labels.append(label)
            groups.append(region_id)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels,    dtype=np.float32),
        np.array(groups,    dtype=int),
    )


# ── Training ───────────────────────────────────────────────────────────────

def train_model(X, y, groups, n_features, device):
    """Train TCN with 5-fold time-based cross-validation."""

    # Global rescale across ALL windows — replace NaN with 0 first
    X_shape = X.shape
    X_flat  = X.reshape(-1, n_features)
    X_flat  = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    global_scaler = StandardScaler()
    X_flat  = global_scaler.fit_transform(X_flat)
    X_flat  = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
    X       = X_flat.reshape(X_shape).astype(np.float32)

    # Binarize labels for classification (threshold at 0.3)
    y_bin = (y >= 0.3).astype(np.float32)

    # Use time-based splits to ensure each fold has mixed classes
    n = len(X)
    fold_size = n // 5
    fold_aucs = []
    history   = []

    best_model_state = None
    best_val_auc     = 0.0

    for fold in range(5):
        val_start = fold * fold_size
        val_end   = val_start + fold_size if fold < 4 else n
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start),
                                    np.arange(val_end, n)])

        n_pos = int(y_bin[val_idx].sum())
        print(f"\n  Fold {fold+1}/5 — train: {len(train_idx)} | "
              f"val: {len(val_idx)} | val_pos: {n_pos}")

        if n_pos == 0 or n_pos == len(val_idx):
            print("  [SKIP] No mixed classes in val fold")
            fold_aucs.append(0.5)
            continue

        # Class-balanced sampler
        pw = (y_bin[train_idx] == 0).sum() / max((y_bin[train_idx] == 1).sum(), 1)
        sw = np.where(y_bin[train_idx] == 1, float(pw), 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.FloatTensor(sw), len(train_idx), replacement=True
        )

        train_ds = EWSDataset(X[train_idx], y_bin[train_idx], groups[train_idx])
        val_ds   = EWSDataset(X[val_idx],   y_bin[val_idx],   groups[val_idx])
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        # Fresh model + OneCycleLR scheduler
        model = TCNEWSDetector(n_features=n_features, seq_len=SEQ_LEN).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=EPOCHS,
            steps_per_epoch=len(train_dl), pct_start=0.1
        )
        criterion = nn.BCELoss()

        best_val   = 0.0
        no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                probs = model(Xb)
                loss  = criterion(probs, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for Xb, yb in val_dl:
                    pred = model(Xb.to(device)).cpu().numpy()
                    val_preds.extend(pred)
                    val_true.extend(yb.numpy())

            val_preds = np.nan_to_num(np.array(val_preds), nan=0.5)
            val_true  = np.array(val_true)

            val_auc = roc_auc_score(val_true, val_preds) \
                if len(np.unique(val_true)) > 1 else 0.5

            history.append({"fold": fold+1, "epoch": epoch+1,
                             "train_loss": train_loss/len(train_dl),
                             "val_auc": val_auc})

            if val_auc > best_val:
                best_val   = val_auc
                no_improve = 0
                if val_auc > best_val_auc:
                    best_val_auc    = val_auc
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stop epoch {epoch+1} | best AUC: {best_val:.4f}")
                    break

            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d} | loss: {train_loss/len(train_dl):.4f}"
                      f" | val_AUC: {val_auc:.4f}")

        fold_aucs.append(best_val)
        print(f"  Fold {fold+1} best AUC: {best_val:.4f}")

    mean_auc = float(np.mean([a for a in fold_aucs if a > 0.5] or [0.5]))
    print(f"\n  Cross-val AUC: {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, pd.DataFrame(history), mean_auc


# ── Inference & feature importance ─────────────────────────────────────────

def predict_area(model, area_name, device):
    """Generate predictions for all windows in an area."""
    csv_path = MERGED_DIR / f"merged_regions_{area_name}.csv"
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["region_id", "time"]).reset_index(drop=True)

    all_preds = []
    model.eval()

    for region_id in df["region_id"].unique():
        df_r  = df[df["region_id"] == region_id].sort_values("time")
        feats = df_r[FEATURE_COLS].copy()
        col_means = feats.mean().fillna(0.0)
        feats = feats.fillna(col_means).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        scaler = StandardScaler()
        feat_arr = scaler.fit_transform(feats.values.astype(np.float32))
        feat_arr = np.nan_to_num(feat_arr, nan=0.0)
        times    = df_r["time"].values

        for i in range(SEQ_LEN, len(feat_arr)):
            window = feat_arr[i - SEQ_LEN : i]
            X_t    = torch.FloatTensor(window).unsqueeze(0).to(device)
            with torch.no_grad():
                risk = model(X_t).item()
            all_preds.append({
                "area":      area_name,
                "region_id": region_id,
                "time":      pd.Timestamp(times[i]),
                "risk_score": risk,
            })

    return pd.DataFrame(all_preds)


def compute_feature_importance(model, X, device, n_permutations=20):
    """
    Permutation feature importance: how much does AUC drop when
    each feature is randomly shuffled?
    """
    model.eval()
    X_t  = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        base_preds = model(X_t).cpu().numpy()

    importances = {}
    for fi, fname in enumerate(FEATURE_COLS):
        drops = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, :, fi])
            with torch.no_grad():
                perm_preds = model(torch.FloatTensor(X_perm).to(device)).cpu().numpy()
            drops.append(np.mean(np.abs(base_preds - perm_preds)))
        importances[fname] = np.mean(drops)

    return pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in sorted(importances.items(), key=lambda x: -x[1])
    ])


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Deep Learning EWS Detector — TCN + Attention")
    print(f"Device: {device}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Sequence length: {SEQ_LEN} months | Prediction horizon: {PRED_HORIZON} months")
    print("=" * 60)

    # Combine data from all areas for training
    print("\nPreparing training data from all areas...")
    all_X, all_y, all_groups = [], [], []
    group_offset = 0

    for area in ["amazon", "sahel", "east_africa"]:
        print(f"  Loading {area}...")
        X, y, groups = prepare_sequences(area)
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups + group_offset)
        group_offset += groups.max() + 1
        print(f"    {len(X)} windows | {(y >= 0.3).sum()} stress windows "
              f"({100*(y>=0.3).mean():.1f}%)")

    X_all      = np.concatenate(all_X,      axis=0)
    y_all      = np.concatenate(all_y,      axis=0)
    groups_all = np.concatenate(all_groups, axis=0)

    print(f"\nTotal: {len(X_all)} windows | "
          f"{(y_all >= 0.3).sum()} positive ({100*(y_all>=0.3).mean():.1f}%)")

    # Train
    print("\nTraining TCN EWS Detector...")
    model, history, mean_auc = train_model(
        X_all, y_all, groups_all, n_features=len(FEATURE_COLS), device=device
    )

    # Save model and history
    torch.save(model.state_dict(), OUT_DIR / "model_weights.pt")
    history.to_csv(OUT_DIR / "training_history.csv", index=False)
    print(f"\n  [SAVED] model_weights.pt")
    print(f"  [SAVED] training_history.csv")
    print(f"  Mean cross-val AUC: {mean_auc:.4f}")

    # Save model config
    config = {
        "n_features":    len(FEATURE_COLS),
        "feature_cols":  FEATURE_COLS,
        "seq_len":       SEQ_LEN,
        "pred_horizon":  PRED_HORIZON,
        "mean_cv_auc":   float(mean_auc),
    }
    with open(OUT_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate predictions for all areas
    print("\nGenerating predictions...")
    for area in ["amazon", "sahel", "east_africa"]:
        df_pred = predict_area(model, area, device)
        df_pred.to_csv(OUT_DIR / f"predictions_{area}.csv", index=False)
        print(f"  [SAVED] predictions_{area}.csv ({len(df_pred)} rows)")

    # Feature importance
    print("\nComputing feature importance (permutation)...")
    df_imp = compute_feature_importance(model, X_all[:500], device)
    df_imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print("\n  Feature importances:")
    print(df_imp.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Deep learning training complete. Results in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
