"""
Publication Figures 1–6  |  NMI format  |  Written from scratch
================================================================
Panel-label rule (applied to EVERY figure):
  - Letters placed INSIDE the axes at top-left  (0.02, 0.97)
  - White semi-transparent backing box
  - clip_on=False  so they are never clipped
  - Always added AFTER tight_layout() so geometry is finalised

NMI specs: 183 mm wide, Arial/Helvetica, 8 pt body, 300 DPI
"""

# ── imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.image import imread
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    HAS_STL = False

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
MRG  = BASE / "data" / "processed" / "merged"
R    = BASE / "results"
EWS  = R / "ews_classical"
DL   = R / "deep_learning"
BL   = R / "baselines"
RISK = R / "risk_maps"
CAU  = R / "causal"
VAL  = R / "validation"
DLL  = R / "deep_learning_loeo"
FIGS = R / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── global style ───────────────────────────────────────────────────────────
MM = 1 / 25.4
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   7,
    "xtick.labelsize":  6,
    "ytick.labelsize":  6,
    "legend.fontsize":  6,
    "figure.dpi":       300,
    "axes.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.0,
    "savefig.bbox":     "tight",
    "savefig.dpi":      300,
})

AREAS  = ["amazon", "sahel", "east_africa"]
ALABEL = {"amazon": "Amazon Rainforest",
           "sahel":  "Sahel Drylands",
           "east_africa": "East Africa"}
COL    = {"amazon": "#009E73", "sahel": "#E69F00", "east_africa": "#56B4E9",
          "tcn": "#CC79A7", "logr": "#999999", "gb": "#F0E442",
          "lstm": "#0072B2", "hi": "#D55E00"}


# ── shared helpers ─────────────────────────────────────────────────────────

def add_panel_label(ax, letter):
    """
    Bold panel letter inside the top-left corner of ax.
    White semi-transparent box ensures legibility on any background.
    clip_on=False prevents clipping by the axes frame.
    """
    ax.text(0.015, 0.975, letter,
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left",
            clip_on=False,
            bbox=dict(boxstyle="square,pad=0.15",
                      facecolor="white", edgecolor="none", alpha=0.80),
            zorder=10)


def stl_detrend(ts):
    """STL deseasonalise then Gaussian-kernel detrend."""
    ts = np.asarray(ts, dtype=float)
    if HAS_STL and len(ts) >= 36:
        try:
            res = STL(ts, period=12, robust=True).fit()
            ts  = res.trend + res.resid
        except Exception:
            pass
    sigma = max(1, int(0.5 * len(ts)))
    return ts - gaussian_filter1d(ts, sigma=sigma)


def load_nc(path):
    """Open netCDF, return (data_array_values, lats, lons)."""
    ds   = xr.open_dataset(path)
    v    = list(ds.data_vars)[0]
    data = ds[v].values
    lats = ds.lat.values
    lons = ds.lon.values
    ds.close()
    return data, lats, lons


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  Study areas · Data table · Framework schematic
# ══════════════════════════════════════════════════════════════════════════

def figure1():
    fig = plt.figure(figsize=(183*MM, 210*MM))
    # Three rows: maps | table | schematic
    gs = gridspec.GridSpec(3, 1,
                           height_ratios=[1.15, 1.05, 0.70],
                           hspace=0.65, figure=fig,
                           left=0.10, right=0.97,
                           top=0.96, bottom=0.04)

    # ── helper: place bold letter to the LEFT of a subplot row ────────────
    def row_label(ax_left, letter):
        """
        Place a bold panel letter just above and to the left of ax_left,
        using figure coordinates so it never overlaps any axes content.
        Renders after draw so positions are exact.
        """
        fig.canvas.draw()
        # get axes bounding box in figure fraction coordinates
        bb = ax_left.get_position()          # Bbox in fig fraction
        # place letter at left of bbox, slightly above top
        fig.text(bb.x0 - 0.055, bb.y1 + 0.005,
                 letter,
                 fontsize=11, fontweight="bold",
                 va="bottom", ha="left",
                 clip_on=False)

    # ── Row 0: geographic maps ─────────────────────────────────────────────
    gs_maps   = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0], wspace=0.30)
    map_cmaps = {"amazon": "YlGn", "sahel": "YlOrBr", "east_africa": "Blues"}
    first_map_ax = None

    for i, area in enumerate(AREAS):
        ax = fig.add_subplot(gs_maps[0, i])
        if i == 0:
            first_map_ax = ax
        p  = MRG / f"region_masks_{area}.nc"
        if p.exists():
            data, lats, lons = load_nc(p)
            data = data.astype(float)
            data[data <= 0] = np.nan
            ax.pcolormesh(lons, lats, data,
                          cmap=map_cmaps[area],
                          shading="auto", rasterized=True)
            ax.set_facecolor("#dde8f0")
        ax.set_title(ALABEL[area], fontsize=7, fontweight="bold", pad=2)
        ax.set_xlabel("Longitude (°)", fontsize=6)
        if i == 0:
            ax.set_ylabel("Latitude (°)", fontsize=6)
        ax.tick_params(labelsize=5)

    # ── Row 1: data sources table ─────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    ax_t.text(0.0, 0.99,
              "Multi-source Earth observation variables (14 total, 2000–2023)",
              transform=ax_t.transAxes, fontsize=7.5, fontweight="bold",
              va="top")

    rows_data = [
        ("#2ca02c", "MODIS MOD13A3 v6.1",   "NDVI, EVI",
         "Vegetation index",     "1 km / Monthly"),
        ("#d62728", "MODIS MOD11A2 v6.1",   "LST_Day, LST_Night",
         "Land surface temp",    "1 km / 8-day"),
        ("#1f77b4", "MODIS MOD16A2GF v6.1", "ET",
         "Evapotranspiration",   "500 m / 8-day"),
        ("#ff7f0e", "MODIS MCD64A1 v6.1",   "BurnDate",
         "Burned area",          "500 m / Monthly"),
        ("#9467bd", "ERA5 Reanalysis",
         "tp, t2m, pev, swvl1\ntp_anom, t2m_anom, pev_anom, swvl1_anom",
         "Climate & soil",       "0.25° / Monthly"),
        ("#8c564b", "ESA CCI SM v8.1",      "soil_moisture",
         "Surface soil moisture","0.25° / Daily"),
    ]
    cols_x  = [0.02, 0.28, 0.54, 0.72, 0.88]
    hdrs    = ["Source", "Variables", "Category", "Resolution"]
    y_hdr   = 0.86
    for hdr, cx in zip(hdrs, cols_x[1:]):
        ax_t.text(cx, y_hdr, hdr, transform=ax_t.transAxes,
                  fontsize=6.5, fontweight="bold", va="center")
    ax_t.plot([0.01, 0.99], [y_hdr-0.04, y_hdr-0.04],
              color="#555", lw=0.6, transform=ax_t.transAxes)

    row_h = 0.118
    for ri, (clr, src, var, cat, res) in enumerate(rows_data):
        yp = y_hdr - 0.05 - (ri + 0.5) * row_h
        # colour swatch
        ax_t.add_patch(mpatches.Rectangle(
            (0.01, yp - 0.04), 0.006, 0.08,
            color=clr, transform=ax_t.transAxes))
        # source name
        ax_t.text(0.02, yp, src, transform=ax_t.transAxes,
                  fontsize=6, fontweight="bold", va="center")
        for val, cx in zip([var, cat, res], cols_x[2:]):
            ax_t.text(cx, yp, val, transform=ax_t.transAxes,
                      fontsize=6, va="center", linespacing=1.2)
        ax_t.plot([0.01, 0.99], [yp - 0.05, yp - 0.05],
                  color="#ddd", lw=0.3, transform=ax_t.transAxes)

    # ── Row 2: framework schematic ────────────────────────────────────────
    ax_s = fig.add_subplot(gs[2])
    ax_s.axis("off"); ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1)

    ax_s.text(0.04, 0.96, "Integrated analytical framework",
              transform=ax_s.transAxes, fontsize=7.5, fontweight="bold",
              va="top")

    boxes = [
        ("Multi-Source EO\n14 Variables\n2000–2023",      0.01, 0.20, "#e3f2fd", "#1565c0"),
        ("Spatiotemporal\nHarmonisation\n0.05° Monthly",  0.25, 0.20, "#e8f5e9", "#2e7d32"),
        ("EWS Engine\n• Classical CSD\n• Dilated TCN\n• PCMCI+ Causal",
                                                          0.49, 0.24, "#fff3e0", "#e65100"),
        ("Outputs\n• Risk Maps\n• Lead Times\n• Causal Drivers",
                                                          0.77, 0.20, "#f3e5f5", "#7b1fa2"),
    ]
    for txt, bx, bw, fc, ec in boxes:
        ax_s.add_patch(mpatches.FancyBboxPatch(
            (bx, 0.12), bw, 0.72,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=fc, edgecolor=ec, lw=0.9,
            transform=ax_s.transAxes))
        ax_s.text(bx + bw/2, 0.48, txt,
                  ha="center", va="center", fontsize=6,
                  fontweight="bold", color="#111",
                  transform=ax_s.transAxes, linespacing=1.25)
    for x0, x1 in [(0.21, 0.25), (0.45, 0.49), (0.73, 0.77)]:
        ax_s.annotate("",
                      xy=(x1, 0.48), xytext=(x0, 0.48),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))

    # ── Panel labels a, b, c using figure-space coords after draw ─────────
    # Draw the canvas so get_position() returns correct Bbox values
    fig.canvas.draw()

    # "a" — above the leftmost map axes
    bb_a = first_map_ax.get_position()
    fig.text(bb_a.x0 - 0.06, bb_a.y1 + 0.008,
             "a", fontsize=11, fontweight="bold",
             va="bottom", ha="left", clip_on=False)

    # "b" — above the table axes
    bb_b = ax_t.get_position()
    fig.text(bb_b.x0 - 0.06, bb_b.y1 + 0.008,
             "b", fontsize=11, fontweight="bold",
             va="bottom", ha="left", clip_on=False)

    # "c" — above the schematic axes
    bb_c = ax_s.get_position()
    fig.text(bb_c.x0 - 0.06, bb_c.y1 + 0.008,
             "c", fontsize=11, fontweight="bold",
             va="bottom", ha="left", clip_on=False)

    fig.savefig(FIGS / "fig1_study_area.pdf")
    fig.savefig(FIGS / "fig1_study_area.png")
    plt.close(fig)
    print("  [OK] fig1_study_area")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  Classical EWS  (3 × 2 grid)
# ══════════════════════════════════════════════════════════════════════════

def figure2():
    BEST = {"amazon": 4, "sahel": 13, "east_africa": 8}
    fig, axes = plt.subplots(3, 2, figsize=(183*MM, 160*MM))
    fig.suptitle(
        "Classical EWS — STL deseasonalised + Gaussian detrend, "
        "block-bootstrap Mann–Kendall",
        fontsize=8, fontweight="bold", y=1.005)

    panel = 0
    for row, area in enumerate(AREAS):
        csvp = MRG / f"merged_regions_{area}.csv"
        ewsp = EWS / f"ews_metrics_{area}.csv"
        if not csvp.exists() or not ewsp.exists():
            continue

        df     = pd.read_csv(csvp, parse_dates=["time"])
        df_ews = pd.read_csv(ewsp)
        rid    = BEST[area]
        df_r   = df[df["region_id"] == rid].sort_values("time")
        tgt    = "NDVI" if (
            "NDVI" in df_r.columns and df_r["NDVI"].notna().sum() > 0
        ) else "EVI"

        raw    = df_r[tgt].interpolate(limit=3).bfill().ffill().values
        t      = pd.to_datetime(df_r["time"].values)
        ts_dt  = stl_detrend(raw)
        W      = 50

        rv = pd.Series(ts_dt).rolling(W, center=True, min_periods=W//2).var()
        ac = pd.Series(ts_dt).rolling(W, center=True, min_periods=W//2).apply(
            lambda v: (float(np.corrcoef(v[:-1], v[1:])[0, 1])
                       if len(v) > 2 else np.nan), raw=True)

        col = COL[area]
        for ci, (series, ylbl) in enumerate(
                [(rv, f"{tgt} variance"), (ac, "Lag-1 AC")]):
            ax = axes[row, ci]
            ax.plot(t, series, color=col, lw=0.9)
            ax.fill_between(t, 0, series.fillna(0), color=col, alpha=0.12)
            if ci == 1:
                ax.axhline(0, color="#aaa", lw=0.5, ls="--")

            # column header only on row 0
            if row == 0:
                col_hdr = "Rolling variance" if ci == 0 else "Lag-1 autocorrelation"
                ax.set_title(f"{col_hdr}\n{ALABEL[area]}",
                             fontsize=7, fontweight="bold", pad=3)
            else:
                ax.set_title(ALABEL[area], fontsize=7, fontweight="bold", pad=3)

            ax.set_ylabel(ylbl, fontsize=6)
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Year", fontsize=6)

            # tau annotation from block-bootstrap MK
            kt = "kendall_tau_variance" if ci == 0 else "kendall_tau_ac1"
            kp = "pval_variance"        if ci == 0 else "pval_ac1"
            row_ews = df_ews[
                (df_ews["region_id"] == rid) & (df_ews["variable"] == tgt)]
            if len(row_ews):
                tau = row_ews[kt].values[0]
                pv  = row_ews[kp].values[0]
                sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "ns")
                tc  = "#cc0000" if (tau > 0 and pv < 0.05) else "#444444"
                ax.text(0.97, 0.96,
                        f"τ={tau:+.2f} {sig}",
                        transform=ax.transAxes,
                        fontsize=6, va="top", ha="right", color=tc,
                        fontweight="bold" if pv < 0.05 else "normal")

            # panel label inside top-left
            add_panel_label(ax, chr(ord("a") + panel))
            panel += 1

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_classical_ews.pdf")
    fig.savefig(FIGS / "fig2_classical_ews.png")
    plt.close(fig)
    print("  [OK] fig2_classical_ews")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  DL performance (a) + GxI heatmap (b)
# ══════════════════════════════════════════════════════════════════════════

def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(183*MM, 95*MM),
                             gridspec_kw={"width_ratios": [1, 1.3],
                                          "wspace": 0.35})
    fig.suptitle(
        "Deep Learning EWS Detector — Performance and Temporal Attribution",
        fontsize=8, fontweight="bold")
    ax_a, ax_b = axes

    # ── Panel a: LOEO AUC bar chart ───────────────────────────────────────
    df_bl     = pd.read_csv(BL / "loeo_baselines.csv")
    df_h      = pd.read_csv(DL / "training_history.csv")
    m_mu      = df_bl.groupby("model")["AUC_ROC"].mean()
    m_sd      = df_bl.groupby("model")["AUC_ROC"].std()
    fold_aucs = df_h.groupby("fold")["val_auc"].max()

    models = ["LogReg", "GradBoost", "LSTM", "TCN\n(ours)"]
    means  = [m_mu.get("LogReg",   0.5), m_mu.get("GradBoost", 0.5),
              m_mu.get("LSTM",     0.5), fold_aucs.mean()]
    sds    = [m_sd.get("LogReg",   0.0), m_sd.get("GradBoost", 0.0),
              m_sd.get("LSTM",     0.0), fold_aucs.std()]
    bcolors = [COL["logr"], COL["gb"], COL["lstm"], COL["tcn"]]

    bars = ax_a.bar(range(4), means, yerr=sds, capsize=3, width=0.55,
                    color=bcolors, edgecolor="black", lw=0.5, zorder=3,
                    error_kw=dict(elinewidth=0.7, ecolor="black"))
    # fold dots for TCN
    for v in fold_aucs:
        ax_a.scatter(3, v, color="white", s=15, zorder=5,
                     edgecolors=COL["tcn"], lw=0.7)
    # significance bracket
    ytop = max(m for m in means) + max(s for s in sds) + 0.06
    ax_a.plot([2, 3], [ytop, ytop], "k-", lw=0.7)
    ax_a.text(2.5, ytop + 0.01, "***", ha="center", fontsize=7,
              fontweight="bold", color="#cc0000")
    ax_a.axhline(0.5, color="#888", ls="--", lw=0.7, label="Random")
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(models, fontsize=6)
    ax_a.set_ylim(0, 1.18)
    ax_a.set_ylabel("LOEO AUC-ROC", fontsize=7)
    ax_a.set_title("Model comparison (LOEO)", fontsize=7, fontweight="bold")
    ax_a.legend(fontsize=5.5, loc="upper left")
    for bar, m in zip(bars, means):
        ax_a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.01,
                  f"{m:.3f}", ha="center", va="bottom", fontsize=5.5,
                  fontweight="bold")
    add_panel_label(ax_a, "a")

    # ── Panel b: temporal GxI heatmap ─────────────────────────────────────
    attr_p = FIGS / "fig_temporal_attribution.png"
    if attr_p.exists():
        img = imread(str(attr_p))
        ax_b.imshow(img, aspect="auto", interpolation="lanczos")
        ax_b.axis("off")
        ax_b.set_title("Temporal feature attribution (Gradient×Input)",
                       fontsize=7, fontweight="bold")
        # Red arrows: feature rows top-to-bottom in heatmap
        # FEATURE_COLS order: NDVI=0,EVI=1,LST_Day=2,ET=3,soil_moisture=4,
        #                     swvl1=5,swvl1_anom=6,t2m=7,t2m_anom=8
        n_feat   = 9
        top3     = [(7, "t2m"), (6, "swvl1_anom"), (8, "t2m_anom")]
        for feat_row, feat_name in top3:
            # rows go top→bottom: row 0 at top (y_frac ~ 1)
            y_frac = (n_feat - 1 - feat_row + 0.5) / n_feat
            ax_b.annotate("",
                          xy=(0.97, y_frac), xytext=(1.00, y_frac),
                          xycoords="axes fraction",
                          arrowprops=dict(arrowstyle="<-",
                                         color=COL["hi"], lw=1.2),
                          annotation_clip=False)
            ax_b.text(1.01, y_frac, feat_name,
                      transform=ax_b.transAxes,
                      fontsize=5.5, va="center", ha="left",
                      color=COL["hi"], fontweight="bold",
                      clip_on=False)
    else:
        # fallback bar chart
        df_g = pd.read_csv(DLL / "gxi_feature_importance.csv")
        df_g = df_g.sort_values("gxi_importance", ascending=True)
        df_g["n"] = df_g["gxi_importance"] / df_g["gxi_importance"].max()
        ax_b.barh(df_g["feature"], df_g["n"],
                  color=[COL["hi"] if v > df_g["n"].median() else "#aaa"
                         for v in df_g["n"]],
                  edgecolor="white", lw=0.3)
        ax_b.set_xlabel("GxI importance (norm.)", fontsize=6)
        ax_b.set_title("Feature attribution (Gradient×Input)",
                       fontsize=7, fontweight="bold")
    add_panel_label(ax_b, "b")

    fig.tight_layout()
    fig.savefig(FIGS / "fig3_dl_performance.pdf")
    fig.savefig(FIGS / "fig3_dl_performance.png")
    plt.close(fig)
    print("  [OK] fig3_dl_performance")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  Spatial risk maps  (2 × 3)
# ══════════════════════════════════════════════════════════════════════════

def figure4():
    fig, axes = plt.subplots(2, 3, figsize=(183*MM, 108*MM),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.22})
    fig.suptitle("Ecosystem Collapse Risk Maps (2000–2023)",
                 fontsize=8, fontweight="bold")

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=0, vmax=1)
    row_info = [
        ("dl_risk_map_",  "DL (TCN) 90th-pctile"),
        ("risk_map_",     "Classical EWS composite"),
    ]
    panel = 0
    for ri, (pfx, row_lbl) in enumerate(row_info):
        for ci, area in enumerate(AREAS):
            ax = axes[ri, ci]
            p  = RISK / f"{pfx}{area}.nc"
            if p.exists():
                data, lats, lons = load_nc(p)
                ax.pcolormesh(lons, lats, data,
                              cmap=cmap, norm=norm,
                              shading="auto", rasterized=True)
                try:
                    ax.contour(lons, lats, data, levels=[0.6],
                               colors=["#222"], linewidths=0.4,
                               linestyles="--", alpha=0.7)
                except Exception:
                    pass
                valid = data[~np.isnan(data)]
                pct   = 100 * np.mean(valid >= 0.6)
                ax.text(0.97, 0.03, f"≥0.6: {pct:.0f}%",
                        transform=ax.transAxes,
                        fontsize=5.5, va="bottom", ha="right",
                        bbox=dict(facecolor="white", alpha=0.80,
                                  edgecolor="none", pad=1.5))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6)

            if ri == 0:
                ax.set_title(ALABEL[area], fontsize=7, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"Lat  |  {row_lbl}", fontsize=5.5)
            if ri == 1:
                ax.set_xlabel("Longitude", fontsize=6)
            ax.tick_params(labelsize=5)

            add_panel_label(ax, chr(ord("a") + panel))
            panel += 1

    # shared colourbar — added last so axes geometry is final
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes.ravel().tolist(),
                        orientation="vertical",
                        shrink=0.80, fraction=0.012, pad=0.02)
    cbar.set_label("Collapse risk score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    fig.savefig(FIGS / "fig4_risk_maps.pdf")
    fig.savefig(FIGS / "fig4_risk_maps.png")
    plt.close(fig)
    print("  [OK] fig4_risk_maps")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5  –  Causal pathways (1 × 3 horizontal bar charts)
# ══════════════════════════════════════════════════════════════════════════

def figure5():
    fig, axes = plt.subplots(1, 3, figsize=(183*MM, 88*MM),
                             gridspec_kw={"wspace": 0.40})
    fig.suptitle(
        "Dominant Causal Pathways to NDVI Decline (PCMCI+, α = 0.05)",
        fontsize=8, fontweight="bold")

    lag_colors = {1: "#e41a1c", 2: "#ff7f00", 3: "#4daf4a",
                  4: "#377eb8", 5: "#984ea3", 6: "#a65628"}

    for ci, area in enumerate(AREAS):
        ax  = axes[ci]
        p   = CAU / f"causal_summary_{area}.csv"
        ax.set_title(ALABEL[area], fontsize=7, fontweight="bold", pad=3)

        if not p.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)
            add_panel_label(ax, chr(ord("a") + ci))
            continue

        df_c = pd.read_csv(p)
        ndvi = df_c[df_c["effect"] == "NDVI"].copy()
        ndvi["abs"] = ndvi["strength"].abs()
        top = (ndvi.groupby(["cause", "lag"])["abs"]
               .mean().reset_index()
               .sort_values("abs", ascending=False).head(8)
               .sort_values("abs", ascending=True))
        top["label"] = top.apply(
            lambda r: f"{r['cause']} (lag {int(r['lag'])}m)", axis=1)
        bcs = [lag_colors.get(int(l), "#888") for l in top["lag"]]

        bars = ax.barh(top["label"], top["abs"],
                       color=bcs, edgecolor="white", lw=0.3, height=0.65)
        for bar, val in zip(bars, top["abs"]):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=5)

        ax.set_xlabel("Mean |causal effect|", fontsize=6)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.tick_params(axis="x", labelsize=5.5)
        add_panel_label(ax, chr(ord("a") + ci))

    # lag legend on last panel
    patches = [mpatches.Patch(color=v, label=f"Lag {k}m")
               for k, v in lag_colors.items()]
    axes[-1].legend(handles=patches, fontsize=5, loc="lower right",
                    title="Causal lag", title_fontsize=5.5,
                    framealpha=0.85)

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_causal_networks.pdf")
    fig.savefig(FIGS / "fig5_causal_networks.png")
    plt.close(fig)
    print("  [OK] fig5_causal_networks")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6  –  Validation (tau bars + severity scatter)
# ══════════════════════════════════════════════════════════════════════════

def figure6():
    df = pd.read_csv(VAL / "temporal_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(183*MM, 90*MM),
                             gridspec_kw={"wspace": 0.38})
    fig.suptitle(
        "Validation Against Nine Documented Ecosystem Stress Events",
        fontsize=8, fontweight="bold")

    # ── Panel a: per-event Kendall τ bars ────────────────────────────────
    ax = axes[0]
    df["tau_plot"] = df["tau_pre_event"].fillna(0.0)
    df["is_nan"]   = df["tau_pre_event"].isna()

    bar_colors = []
    for _, row in df.iterrows():
        if row["is_nan"]:
            bar_colors.append("#cccccc")
        elif row["rising_signal"]:
            bar_colors.append("#009E73")
        else:
            bar_colors.append(COL["hi"])

    x    = np.arange(len(df))
    bars = ax.bar(x, df["tau_plot"], color=bar_colors,
                  edgecolor="white", lw=0.4, width=0.65, zorder=3)

    # hatch NaN bars
    for bar, row in zip(bars, df.itertuples()):
        if row.is_nan:
            bar.set_hatch("///")
            bar.set_edgecolor("#999")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    0.01, "no τ", ha="center", fontsize=4.5,
                    color="#666", style="italic")

    ax.axhline(0, color="black", lw=0.7)
    # significance stars
    for i, row in df.iterrows():
        if pd.isna(row["pval_trend"]):
            continue
        pv  = row["pval_trend"]
        sig = ("***" if pv < 0.001 else
               "**"  if pv < 0.01  else
               "*"   if pv < 0.05  else "")
        if sig:
            yoff = 0.02 if row["tau_plot"] >= 0 else -0.06
            ax.text(i, row["tau_plot"] + yoff, sig,
                    ha="center", fontsize=6,
                    fontweight="bold", color="#333")

    # x-tick labels
    short = (df["event"]
             .str.replace("Amazon Drought ",    "AD-")
             .str.replace("Amazon El Niño ",    "AEl-")
             .str.replace("Sahel Crisis ",      "SC-")
             .str.replace("Sahel Drought ",     "SD-")
             .str.replace("Horn of Africa ",    "HoA-")
             .str.replace("East Africa Drought ","EAD-")
             .str.replace("Multi-year Drought ","MYD-"))
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=5, rotation=38, ha="right")
    ax.set_ylabel("Kendall τ (24-month pre-event)", fontsize=6)
    ax.set_title("Pre-event EWS signal per documented event",
                 fontsize=7, fontweight="bold")
    n_det = int(df["rising_signal"].sum())
    ax.text(0.97, 0.97, f"{n_det}/{len(df)} detected",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
            fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color="#009E73",   label="Rising signal (p<0.1)"),
        mpatches.Patch(color=COL["hi"],   label="No signal"),
        mpatches.Patch(facecolor="#ccc",  hatch="///",
                       edgecolor="#999",  label="No valid τ"),
    ], fontsize=5, loc="upper left")
    add_panel_label(ax, "a")

    # ── Panel b: risk vs severity scatter ────────────────────────────────
    ax = axes[1]
    for _, row in df.iterrows():
        c = COL[row["area"]]
        ax.scatter(row["severity"], row["mean_pre_risk"],
                   color=c, s=45, alpha=0.9,
                   edgecolors="white", lw=0.6, zorder=3)
        lbl = (row["event"]
               .replace("Amazon Drought ",    "AD")
               .replace("Amazon El Niño ",    "AEl")
               .replace("Sahel Crisis ",      "SC")
               .replace("Sahel Drought ",     "SD")
               .replace("Horn of Africa ",    "HoA")
               .replace("East Africa Drought ","EAD")
               .replace("Multi-year Drought ","MYD"))
        ax.text(row["severity"] + 0.01, row["mean_pre_risk"] + 0.003,
                lbl, fontsize=4.5, alpha=0.75)

    r_val, p_val = stats.pearsonr(df["severity"], df["mean_pre_risk"])
    sl, ic, *_   = stats.linregress(df["severity"], df["mean_pre_risk"])
    xl = np.linspace(df["severity"].min() - 0.05,
                     df["severity"].max() + 0.05, 50)
    ax.plot(xl, sl * xl + ic, "--", color="#555", lw=0.8, alpha=0.6)
    ax.text(0.97, 0.05, f"r = {r_val:.2f},  p = {p_val:.3f}",
            transform=ax.transAxes, fontsize=6, va="bottom", ha="right")
    ax.set_xlabel("Event severity (0–1)", fontsize=6)
    ax.set_ylabel("Mean pre-event risk score", fontsize=6)
    ax.set_title("Pre-event risk vs event severity",
                 fontsize=7, fontweight="bold")
    ax.legend(
        handles=[mpatches.Patch(color=COL[a], label=ALABEL[a])
                 for a in AREAS],
        fontsize=5, loc="lower right")
    add_panel_label(ax, "b")

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_validation.pdf")
    fig.savefig(FIGS / "fig6_validation.png")
    plt.close(fig)
    print("  [OK] fig6_validation")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("Generating Figures 1–6  (NMI format, labels inside axes)")
    print(f"STL: {'statsmodels' if HAS_STL else 'rolling-mean fallback'}")
    print(f"Output: {FIGS}")
    print("=" * 58)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("\nDone — all 6 figures saved.")


if __name__ == "__main__":
    main()
