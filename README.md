# Causal Early Warning Signals for Ecosystem Collapse
## Multivariate Deep Learning Framework Across Three Tropical Biomes

**Paper:** Munireddy R., Muthu R., Dhakshinamoorthy G.P., Kunasekaran K., Manimaran K. (2026).
*Causal Early Warning Signals for Ecosystem Collapse Detected by a Multivariate Deep Learning
Framework Across Three Tropical Biomes.* Nature Machine Intelligence.

---

## Repository Structure

```
├── manuscript/               Manuscript + cover letter + references
├── figures/                  Publication figures (Fig 1–6, PDF 300 dpi)
├── source_code/              Full analysis pipeline (Python)
│   ├── README_code.md        Step-by-step reproduction guide
│   ├── 00_setup/             Environment setup + data access guide
│   ├── 01_data_download/     MODIS / ERA5 / ESA CCI SM download scripts
│   ├── 02_data_processing/   Reprojection, harmonisation, merging
│   ├── 03_ews_analysis/      Classical EWS (STL + block-bootstrap MK)
│   ├── 04_deep_learning/     TCN detector + baselines + GxI attribution
│   ├── 05_causal_discovery/  PCMCI+ causal analysis
│   ├── 06_risk_mapping/      DL + classical spatial risk maps
│   ├── 07_validation_figures/Event validation + all publication figures
│   ├── RUN_PIPELINE.py       One-shot data pipeline runner
│   └── RUN_MODELLING.py      One-shot modelling + figures runner
└── supplementary_data/       Key result tables + trained model weights
    ├── model_weights.pt       Trained TCN model (PyTorch)
    ├── model_config.json      TCN architecture configuration
    ├── training_history.csv   Per-fold LOEO AUC-ROC
    ├── loeo_baselines.csv     Baseline model LOEO results
    ├── gxi_feature_importance.csv  Gradient×Input attribution scores
    ├── ews_metrics_*.csv      Classical EWS Kendall τ (3 areas)
    ├── causal_summary_*.csv   PCMCI+ significant causal links (3 areas)
    ├── temporal_validation.csv    Event pre-detection results (9 events)
    └── region_metadata_*.csv  Sub-region lat/lon metadata (3 areas)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r source_code/00_setup/requirements.txt

# 2. Run full data pipeline (requires NASA Earthdata + CDS + CEDA accounts)
python source_code/RUN_PIPELINE.py

# 3. Run full modelling pipeline + generate all figures
python source_code/RUN_MODELLING.py
```

See `source_code/README_code.md` for detailed step-by-step instructions
and `source_code/00_setup/DATA_ACCESS_GUIDE.md` for data access credentials.

---

## Key Results

| Model | LOEO AUC-ROC |
|-------|-------------|
| TCN (ours) | **0.883 ± 0.136** |
| LSTM | 0.686 ± 0.194 |
| Gradient Boosting | 0.575 ± 0.154 |
| Logistic Regression | 0.560 ± 0.148 |

- 7/9 documented ecosystem stress events detected (12–24 month lead time)
- 584 significant PCMCI+ causal links across 3 biomes
- DL risk maps: Amazon 77%, Sahel 73%, East Africa 74% pixels ≥ 0.6 risk

---

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| MODIS MOD13A3, MOD11A2, MOD16A2GF, MCD64A1 v6.1 | NASA Earthdata | https://earthdata.nasa.gov |
| ERA5 Monthly Reanalysis | Copernicus CDS | https://cds.climate.copernicus.eu |
| ESA CCI Soil Moisture v8.1 | CEDA Archive | https://data.ceda.ac.uk |

---

## Citation

```bibtex
@article{munireddy2026causal,
  author  = {Munireddy, Rajendraprasad and Muthu, Rajapriya and
             Dhakshinamoorthy, Gnana Prakash and Kunasekaran, Kadambanathan
             and Manimaran, Kumar},
  title   = {Causal Early Warning Signals for Ecosystem Collapse Detected
             by a Multivariate Deep Learning Framework Across Three
             Tropical Biomes},
  journal = {Nature Machine Intelligence},
  year    = {2026},
}
```

---

## Contact

Rajendraprasad Munireddy (Corresponding Author)
Email: rathnarajsince1999@gmail.com
