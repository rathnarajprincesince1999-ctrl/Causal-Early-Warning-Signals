# Causal Early Warning Signals for Ecosystem Collapse
## Code Reproduction Guide

**Paper:** Munireddy R. et al. (2026). Causal Early Warning Signals for Ecosystem
Collapse Detected by a Multivariate Deep Learning Framework Across Three Tropical Biomes.
*Nature Machine Intelligence*.

---

## Environment

Python 3.10+. Install all dependencies:

```bash
pip install -r 00_setup/requirements.txt
```

GPU (CUDA) strongly recommended for steps 4–5.

---

## Step-by-step execution

### Step 1 — Data download
```bash
python 01_data_download/01_download_MODIS.py          # MODIS HDF files (~456 GB)
python 01_data_download/02_download_ERA5.py            # ERA5 NetCDF (NASA CDS API)
python 01_data_download/03_download_ESA_CCI_SoilMoisture.py  # ESA CCI SM daily
python 01_data_download/04_download_MOD16A2_missing.py # Fill any missing ET tiles
```

### Step 2 — Data processing
```bash
python 02_data_processing/01_process_MODIS.py          # Reprojects + aggregates MODIS
python 02_data_processing/02_process_ERA5.py           # Resamples ERA5 to 0.05°
python 02_data_processing/02b_process_ESA_CCI_SoilMoisture.py
python 02_data_processing/03_merge_all_variables.py    # Creates merged_regions_*.csv
```

### Step 3 — Classical EWS
```bash
python 03_ews_analysis/01_classical_ews.py
# Output: results/ews_classical/ews_metrics_*.csv
```

### Step 4 — Deep learning (TCN + LOEO + GxI attribution)
```bash
python 04_deep_learning/01_train_ews_detector.py       # Train TCN (GPU recommended)
python 04_deep_learning/02_baselines.py                # LogReg, GradBoost, LSTM
python 04_deep_learning/03_dl_loeo_shap.py             # LOEO eval + GxI heatmap
# Output: results/deep_learning/, results/deep_learning_loeo/
```

### Step 5 — Causal discovery
```bash
python 05_causal_discovery/01_pcmci_analysis.py
# Output: results/causal/causal_summary_*.csv
```

### Step 6 — Risk mapping
```bash
python 06_risk_mapping/01_spatial_risk_maps.py         # Classical EWS risk maps
python 06_risk_mapping/02_dl_risk_maps.py              # DL 90th-pctile risk maps
# Output: results/risk_maps/
```

### Step 7 — Validation & figures
```bash
python 07_validation_figures/01_validation.py          # Event validation
python 07_validation_figures/02_figures.py             # Generate Figs 1–6
# Output: results/validation/, results/figures/
```

### One-shot runners (run full pipeline)
```bash
python RUN_PIPELINE.py     # Steps 1–2 (data)
python RUN_MODELLING.py    # Steps 3–7 (modelling + figures)
```

---

## Key outputs

| File | Description |
|------|-------------|
| `results/deep_learning/model_weights.pt` | Trained TCN model weights |
| `results/deep_learning/training_history.csv` | Per-fold LOEO AUC |
| `results/baselines/loeo_baselines.csv` | Baseline LOEO results |
| `results/deep_learning_loeo/gxi_feature_importance.csv` | GxI attribution |
| `results/ews_classical/ews_metrics_*.csv` | Classical EWS Kendall τ |
| `results/causal/causal_summary_*.csv` | PCMCI+ causal links |
| `results/risk_maps/dl_risk_map_*.nc` | DL risk maps (NetCDF) |
| `results/risk_maps/risk_map_*.nc` | Classical risk maps (NetCDF) |
| `results/validation/temporal_validation.csv` | Event detection results |
| `results/figures/fig[1-6]_*.pdf` | Publication figures |

---

## Hardware used

- CPU: Intel Core i7 (12th Gen)
- GPU: NVIDIA RTX 4050 (6 GB VRAM)
- RAM: 32 GB
- Approximate wall-clock time: ~6 hours (full pipeline, GPU)

---

## Data access

MODIS: NASA Earthdata (https://earthdata.nasa.gov) — free account required.
ERA5: Copernicus CDS API (https://cds.climate.copernicus.eu) — free account required.
ESA CCI SM: CEDA Archive (https://data.ceda.ac.uk) — free account required.
See `00_setup/DATA_ACCESS_GUIDE.md` for credentials setup.
