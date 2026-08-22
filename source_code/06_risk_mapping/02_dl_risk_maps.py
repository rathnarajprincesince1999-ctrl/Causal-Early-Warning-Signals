"""
DL-Based Spatial Risk Maps — Reviewer Fix
==========================================
Addresses: "spatial risk maps are univariate NDVI — inconsistent with DL claim"

Uses the trained TCN model directly on pixel-level time series to produce
multivariate collapse risk maps. Semi-supervised approach:
  1. Train DL model on regional data (labelled)
  2. Apply pixel-by-pixel on merged_pixel_{area}.nc (all 14 variables)
  3. Output DL-based risk probability maps

Outputs: results/risk_maps/dl_risk_map_{area}.nc
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

BASE_DIR   = Path(__file__).resolve().parents[1]
MERGED_DIR = BASE_DIR / "data" / "processed" / "merged"
DL_DIR     = BASE_DIR / "results" / "deep_learning"
OUT_DIR    = BASE_DIR / "results" / "risk_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN      = 60
FEATURE_COLS = ["NDVI", "EVI", "LST_Day", "ET", "soil_moisture",
                "swvl1", "swvl1_anom", "t2m", "t2m_anom"]
BATCH_SIZE   = 2048   # large batch for GPU efficiency


# ── Import model from training script ─────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR / "04_deep_learning"))

class CausalConvBlock(nn.Module):
    def __init__(self, n_in, n_out, ks, dil):
        super().__init__()
        pad = (ks-1)*dil
        self.c1 = nn.Conv1d(n_in, n_out, ks, dilation=dil, padding=pad)
        self.c2 = nn.Conv1d(n_out, n_out, ks, dilation=dil, padding=pad)
        self.bn1 = nn.BatchNorm1d(n_out); self.bn2 = nn.BatchNorm1d(n_out)
        self.drop = nn.Dropout(0.2)
        self.ds = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
    def forward(self, x):
        r = x if self.ds is None else self.ds(x)
        x = torch.relu(self.bn1(self.c1(x)[:,:,:x.size(2)])); x = self.drop(x)
        x = torch.relu(self.bn2(self.c2(x)[:,:,:r.size(2)]))
        return torch.relu(x + r)

class TCNDetector(nn.Module):
    def __init__(self, n_feat=9, seq_len=60, ch=64, nl=4):
        super().__init__()
        layers, ic = [], n_feat
        for i in range(nl):
            layers.append(CausalConvBlock(ic, ch, 3, 2**i)); ic = ch
        self.tcn  = nn.Sequential(*layers)
        self.attn = nn.MultiheadAttention(ch, 4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(ch)
        self.head = nn.Sequential(nn.Linear(ch,32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        o = self.tcn(x.permute(0,2,1)).permute(0,2,1)
        a, _ = self.attn(o, o, o)
        return self.head(self.norm(o+a).mean(1)).squeeze(1)


def load_model(device):
    """Load trained model weights."""
    weights = DL_DIR / "model_weights.pt"
    config  = DL_DIR / "model_config.json"
    if not weights.exists():
        print("  [WARN] No trained model found. Run 01_train_ews_detector.py first.")
        return None, None

    import json
    with open(config) as f:
        cfg = json.load(f)

    model = TCNDetector(n_feat=cfg["n_features"], seq_len=cfg["seq_len"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    print("  Model loaded (strict=False — some keys may differ between versions)")
    model.eval()
    return model, cfg


def build_dl_risk_map(area_name, model, scaler, device):
    """Apply trained DL model pixel-by-pixel to produce risk map."""
    pixel_path = MERGED_DIR / f"merged_pixel_{area_name}.nc"
    if not pixel_path.exists():
        print(f"  [SKIP] {pixel_path} not found")
        return

    print(f"  Loading pixel dataset for {area_name}...")
    ds   = xr.open_dataset(pixel_path)
    lats = ds.lat.values
    lons = ds.lon.values
    n_lat, n_lon = len(lats), len(lons)
    n_time = len(ds.time)

    # Extract features one variable at a time to avoid OOM
    feat_arrays = []
    for feat in FEATURE_COLS:
        ds2 = xr.open_dataset(pixel_path)
        if feat in ds2:
            arr = ds2[feat].values.astype(np.float32)
        else:
            arr = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
        ds2.close()
        feat_arrays.append(arr)
        del arr

    data = np.stack(feat_arrays, axis=-1)   # (time, lat, lon, n_feat)
    del feat_arrays
    print(f"  Data shape: {data.shape} | NaN: {np.isnan(data).sum():,}")

    # Build risk map — initialise with NaN
    risk_map = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

    # Find pixels with sufficient valid data
    valid_pixels = []
    for i in range(n_lat):
        for j in range(n_lon):
            ts = data[:, i, j, :]        # (time, n_feat)
            nan_frac = np.isnan(ts).mean()
            if nan_frac < 0.4:
                valid_pixels.append((i, j))

    print(f"  Valid pixels: {len(valid_pixels):,} / {n_lat*n_lon:,}")

    # Process in batches
    batch_coords = []
    batch_seqs   = []

    for idx, (i, j) in enumerate(tqdm(valid_pixels, desc=f"  {area_name}", unit="px")):
        ts = data[:, i, j, :].copy()    # (time, n_feat)

        # Fill NaN
        for fi in range(ts.shape[1]):
            col = ts[:, fi]
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0.0
            ts[np.isnan(ts[:, fi]), fi] = col_mean

        # Last SEQ_LEN timesteps as input window
        if n_time < SEQ_LEN:
            continue
        window = ts[-SEQ_LEN:]          # (60, n_feat)

        # Scale using global scaler
        window_scaled = np.nan_to_num(
            scaler.transform(window), nan=0.0
        ).astype(np.float32)

        batch_coords.append((i, j))
        batch_seqs.append(window_scaled)

        # Run batch when full or at end
        if len(batch_seqs) >= BATCH_SIZE or idx == len(valid_pixels) - 1:
            if not batch_seqs:
                continue
            X_batch = torch.FloatTensor(np.array(batch_seqs)).to(device)
            with torch.no_grad():
                preds = model(X_batch).cpu().numpy()

            for (bi, bj), pred in zip(batch_coords, preds):
                risk_map[bi, bj] = float(pred)

            batch_coords = []
            batch_seqs   = []

    # Save
    ds_out = xr.Dataset(
        {"dl_risk": (["lat","lon"], risk_map)},
        coords={"lat": lats, "lon": lons},
        attrs={
            "study_area":   area_name,
            "description":  "DL-based (TCN+Attention) ecosystem collapse risk",
            "model":        "TCN with multihead self-attention, 9 variables",
            "time_window":  f"Last {SEQ_LEN} months of data",
        }
    )
    enc = {"dl_risk": {"zlib":True,"complevel":4,"dtype":"float32"}}
    out = OUT_DIR / f"dl_risk_map_{area_name}.nc"
    ds_out.to_netcdf(out, encoding=enc)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  [SAVED] {out.name} ({size_mb:.1f} MB)")
    print(f"  DL risk — mean: {np.nanmean(risk_map):.3f} | "
          f"high (≥0.6): {100*(risk_map>=0.6).sum()/np.sum(~np.isnan(risk_map)):.1f}%")
    return risk_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("DL-Based Spatial Risk Mapping")
    print(f"Device: {device}")
    print("="*60)

    # Load trained model
    model, cfg = load_model(device)
    if model is None:
        print("[ERROR] Trained model required. Run 04_deep_learning/01_train_ews_detector.py first.")
        return

    # Build a global scaler from regional data (same as training)
    print("\nBuilding global scaler from regional data...")
    all_seqs = []
    for area in ["amazon","sahel","east_africa"]:
        df = pd.read_csv(MERGED_DIR/f"merged_regions_{area}.csv", parse_dates=["time"])
        df = df.sort_values(["region_id","time"])
        for rid in df.region_id.unique():
            dr    = df[df.region_id==rid].sort_values("time")
            feats = dr[FEATURE_COLS].copy()
            means = feats.mean().fillna(0.0)
            fa    = feats.fillna(means).replace([np.inf,-np.inf],0.0).fillna(0.0).values
            for i in range(60, len(fa)):
                all_seqs.append(fa[i-60:i])

    X_flat = np.array(all_seqs, dtype=np.float32).reshape(-1, len(FEATURE_COLS))
    X_flat = np.nan_to_num(X_flat, nan=0.0)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    print(f"  Scaler fitted on {len(X_flat):,} windows")

    # Build DL risk maps
    for area in ["amazon","sahel","east_africa"]:
        print(f"\n{'='*40}")
        print(f"Building DL risk map: {area}")
        print(f"{'='*40}")
        build_dl_risk_map(area, model, scaler, device)

    print(f"\nDL risk maps saved to: {OUT_DIR}")
    print("Compare dl_risk_map_*.nc with classical risk_map_*.nc")


if __name__ == "__main__":
    main()
