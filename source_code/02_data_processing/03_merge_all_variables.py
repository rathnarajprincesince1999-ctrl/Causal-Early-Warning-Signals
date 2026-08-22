"""
Multi-Variable Data Merging Script
=====================================
Merges all processed data sources into a single analysis-ready dataset
per study area and per region (spatial zones).

Input sources (all processed):
  - MODIS NDVI monthly
  - MODIS EVI monthly
  - MODIS LST Day monthly
  - MODIS ET monthly
  - MODIS Burned Area monthly
  - ERA5 Precipitation (raw + anomaly)
  - ERA5 Temperature (raw + anomaly)
  - ERA5 Potential ET (raw + anomaly)
  - ERA5 Soil Water (raw + anomaly)
  - ESA CCI Soil Moisture monthly

Steps:
  1. Load all variables
  2. Regrid all to common 0.05 deg grid (if not already)
  3. Align to common monthly time axis (2000-01 to 2023-12)
  4. Mask to valid land pixels only
  5. Extract regional time series (spatially aggregated zones)
  6. Save merged per-pixel dataset and per-region dataset

Outputs:
  data/processed/merged/
    merged_pixel_{area}.nc          — full pixel-level merged dataset
    merged_regions_{area}.csv       — regional time series (for EWS/causal analysis)
    region_masks_{area}.nc          — spatial region masks
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from scipy.ndimage import label
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[1]
MODIS_DIR = BASE_DIR / "data" / "processed" / "MODIS"
ERA5_DIR  = BASE_DIR / "data" / "processed" / "ERA5"
ESMCCI_DIR= BASE_DIR / "data" / "processed" / "ESA_CCI_SoilMoisture"
OUT_DIR   = BASE_DIR / "data" / "processed" / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Common grid ────────────────────────────────────────────────────────────
RESOLUTION = 0.05   # degrees

# ── Common time axis ───────────────────────────────────────────────────────
TIME_START = "2000-01"
TIME_END   = "2023-12"
COMMON_TIME = pd.date_range(TIME_START, TIME_END, freq="MS")

# ── Study areas ────────────────────────────────────────────────────────────
STUDY_AREAS = {
    "amazon":      {"lon": (-73.0, -44.0), "lat": (-18.0,  5.0)},
    "sahel":       {"lon": (-17.0,  40.0), "lat": ( 10.0, 20.0)},
    "east_africa": {"lon": ( 33.0,  51.0), "lat": ( -5.0, 15.0)},
}

# ── Region grid: divide each area into a grid of sub-regions ──────────────
REGION_GRID = {
    "amazon":      (5, 6),    # 5 rows x 6 cols = 30 regions
    "sahel":       (2, 8),    # 2 rows x 8 cols = 16 regions
    "east_africa": (4, 3),    # 4 rows x 3 cols = 12 regions
}


def regrid_to_common(da, resolution=RESOLUTION):
    """Interpolate a DataArray to the common grid."""
    # Detect lat/lon dimension names
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

    lat_min = float(da[lat_name].min())
    lat_max = float(da[lat_name].max())
    lon_min = float(da[lon_name].min())
    lon_max = float(da[lon_name].max())

    new_lat = np.arange(lat_min, lat_max + resolution/2, resolution)
    new_lon = np.arange(lon_min, lon_max + resolution/2, resolution)

    kwargs = {lat_name: new_lat, lon_name: new_lon}
    return da.interp(**kwargs, method="linear")


def align_time(da, common_time=COMMON_TIME):
    """Reindex DataArray to common monthly time axis."""
    if "time" not in da.dims:
        return da
    return da.reindex(time=common_time, method="nearest", tolerance="16D")


def create_region_masks(area_name, bbox):
    """
    Divide study area into rectangular sub-regions.
    Returns an integer mask array [lat, lon] where each integer = region ID.
    """
    nrows, ncols = REGION_GRID[area_name]
    lon_min, lon_max = bbox["lon"]
    lat_min, lat_max = bbox["lat"]

    lats = np.arange(lat_min + RESOLUTION/2, lat_max, RESOLUTION)
    lons = np.arange(lon_min + RESOLUTION/2, lon_max, RESOLUTION)

    mask = np.zeros((len(lats), len(lons)), dtype=int)

    lat_edges = np.linspace(lat_min, lat_max, nrows + 1)
    lon_edges = np.linspace(lon_min, lon_max, ncols + 1)

    region_id = 1
    region_meta = []

    for r in range(nrows):
        for c in range(ncols):
            lat_lo, lat_hi = lat_edges[r], lat_edges[r + 1]
            lon_lo, lon_hi = lon_edges[c], lon_edges[c + 1]

            lat_idx = np.where((lats >= lat_lo) & (lats < lat_hi))[0]
            lon_idx = np.where((lons >= lon_lo) & (lons < lon_hi))[0]

            if len(lat_idx) > 0 and len(lon_idx) > 0:
                mask[np.ix_(lat_idx, lon_idx)] = region_id
                region_meta.append({
                    "region_id":  region_id,
                    "area":       area_name,
                    "lat_center": (lat_lo + lat_hi) / 2,
                    "lon_center": (lon_lo + lon_hi) / 2,
                    "lat_min":    lat_lo, "lat_max": lat_hi,
                    "lon_min":    lon_lo, "lon_max": lon_hi,
                    "row": r, "col": c,
                })
                region_id += 1

    mask_da = xr.DataArray(
        mask,
        dims=["lat", "lon"],
        coords={
            "lat": lats,
            "lon": lons,
        },
        attrs={"area": area_name, "description": "Region ID mask"},
    )

    return mask_da, pd.DataFrame(region_meta)


def load_modis_variable(var_name, area_name):
    """Load a processed MODIS variable NetCDF."""
    path = MODIS_DIR / f"{var_name}_monthly_{area_name}.nc"
    if not path.exists():
        print(f"    [MISSING] {path.name}")
        return None
    ds = xr.open_dataset(path)
    da = ds[var_name]
    print(f"    [OK] MODIS {var_name}: {da.shape}")
    return da


def load_era5_variable(var_name, area_name):
    """Load a processed ERA5 variable."""
    path = ERA5_DIR / f"ERA5_processed_{area_name}.nc"
    if not path.exists():
        print(f"    [MISSING] ERA5 file for {area_name}")
        return None
    ds = xr.open_dataset(path)
    if var_name not in ds:
        print(f"    [MISSING] Variable {var_name} in ERA5 file")
        return None
    da = ds[var_name]
    print(f"    [OK] ERA5 {var_name}: {da.shape}")
    return da


def load_cci_soilmoisture(area_name):
    """Load processed ESA CCI soil moisture."""
    path = ESMCCI_DIR / f"ESA_CCI_SM_monthly_{area_name}.nc"
    if not path.exists():
        print(f"    [MISSING] {path.name}")
        return None
    ds = xr.open_dataset(path)
    da = ds["sm"]
    print(f"    [OK] ESA CCI SM: {da.shape}")
    return da


def extract_regional_timeseries(merged_ds, mask_da, region_meta, area_name):
    """
    For each region, compute spatially averaged time series.
    Returns a DataFrame: rows=time, columns=region_variable combinations.
    """
    print(f"    Extracting regional time series for {area_name}...")

    # Align mask to merged dataset grid
    mask_interp = mask_da.interp(
        lat=merged_ds.lat,
        lon=merged_ds.lon,
        method="nearest",
    )

    rows = []
    for _, region in region_meta.iterrows():
        rid = int(region["region_id"])
        region_mask = (mask_interp == rid)

        row_data = {"time": None}
        for var in merged_ds.data_vars:
            da = merged_ds[var]
            # Spatial mean over region
            da_masked = da.where(region_mask)
            da_mean   = da_masked.mean(dim=["lat", "lon"])
            for t, val in zip(da_mean.time.values, da_mean.values):
                rows.append({
                    "time":       pd.Timestamp(t),
                    "region_id":  rid,
                    "area":       area_name,
                    "lat_center": region["lat_center"],
                    "lon_center": region["lon_center"],
                    "variable":   var,
                    "value":      float(val) if not np.isnan(val) else np.nan,
                })

    df_long = pd.DataFrame(rows)

    # Pivot to wide format: rows=time × region, columns=variables
    df_wide = df_long.pivot_table(
        index=["time", "region_id", "area", "lat_center", "lon_center"],
        columns="variable",
        values="value",
    ).reset_index()
    df_wide.columns.name = None

    return df_wide


def merge_area(area_name, bbox):
    """Full merge pipeline for one study area."""
    print(f"\n{'='*60}")
    print(f"Merging: {area_name}")
    print(f"{'='*60}")

    out_pixel  = OUT_DIR / f"merged_pixel_{area_name}.nc"
    out_region = OUT_DIR / f"merged_regions_{area_name}.csv"

    if out_pixel.exists() and out_region.exists():
        print(f"  [SKIP] Merged files already exist for {area_name}")
        return

    variables = {}

    # MODIS variables
    for var in ["NDVI", "EVI", "LST_Day", "ET", "BurnDate"]:
        da = load_modis_variable(var, area_name)
        if da is not None:
            variables[var] = da

    # ERA5 variables
    for var in ["tp", "t2m", "pev", "swvl1",
                "tp_anom", "t2m_anom", "pev_anom", "swvl1_anom"]:
        da = load_era5_variable(var, area_name)
        if da is not None:
            variables[var] = da

    # ESA CCI Soil Moisture
    da_sm = load_cci_soilmoisture(area_name)
    if da_sm is not None:
        variables["soil_moisture"] = da_sm

    if not variables:
        print(f"  [ERROR] No data loaded for {area_name}. Run download scripts first.")
        return

    print(f"\n  Loaded {len(variables)} variables: {list(variables.keys())}")

    # ── Regrid all to common 0.05 deg ─────────────────────────────────────
    print("  Regridding all to 0.05 deg common grid...")
    regridded = {}
    for vname, da in variables.items():
        try:
            da_rg = regrid_to_common(da)
            da_rg = align_time(da_rg)
            # Standardize coord names
            rename_map = {}
            if "latitude" in da_rg.dims:
                rename_map["latitude"] = "lat"
            if "longitude" in da_rg.dims:
                rename_map["longitude"] = "lon"
            if rename_map:
                da_rg = da_rg.rename(rename_map)
            regridded[vname] = da_rg
        except Exception as e:
            print(f"    [WARN] Could not regrid {vname}: {e}")

    # ── Create common lat/lon grid from first variable ─────────────────────
    ref_da  = list(regridded.values())[0]
    ref_lat = ref_da.lat.values
    ref_lon = ref_da.lon.values

    # Re-interpolate all to identical lat/lon coords
    aligned = {}
    for vname, da in regridded.items():
        try:
            da_al = da.interp(lat=ref_lat, lon=ref_lon, method="linear")
            aligned[vname] = da_al
        except Exception as e:
            print(f"    [WARN] Could not align {vname}: {e}")

    # ── Build merged Dataset ───────────────────────────────────────────────
    print("  Building merged dataset...")
    ds_dict = {vname: da for vname, da in aligned.items()}
    merged  = xr.Dataset(ds_dict)

    merged.attrs["study_area"]  = area_name
    merged.attrs["period"]      = f"{TIME_START} to {TIME_END}"
    merged.attrs["resolution"]  = "0.05 degrees"
    merged.attrs["sources"]     = "MODIS MOD13A3/MOD11A2/MOD16A2/MCD64A1, ERA5, ESA CCI SM"
    merged.attrs["variables"]   = ", ".join(list(aligned.keys()))

    # Save pixel-level dataset
    if not out_pixel.exists():
        print(f"  Saving pixel-level dataset...")
        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
            for v in merged.data_vars
        }
        merged.to_netcdf(out_pixel, encoding=encoding)
        size_mb = out_pixel.stat().st_size / (1024 * 1024)
        print(f"  [SAVED] {out_pixel.name} ({size_mb:.1f} MB)")

    # ── Create region masks ────────────────────────────────────────────────
    print("  Creating region masks...")
    mask_da, region_meta = create_region_masks(area_name, bbox)

    mask_out = OUT_DIR / f"region_masks_{area_name}.nc"
    if not mask_out.exists():
        mask_da.to_dataset(name="region_id").to_netcdf(mask_out)
        region_meta.to_csv(
            OUT_DIR / f"region_metadata_{area_name}.csv", index=False
        )
        print(f"  [SAVED] {mask_out.name} ({len(region_meta)} regions)")

    # ── Extract regional time series ───────────────────────────────────────
    if not out_region.exists():
        df_regions = extract_regional_timeseries(merged, mask_da, region_meta, area_name)
        df_regions.to_csv(out_region, index=False)
        print(f"  [SAVED] {out_region.name} ({df_regions.shape[0]} rows)")

    print(f"\n  Merge complete for {area_name}.")
    print(f"  Variables in merged dataset: {list(merged.data_vars)}")
    print(f"  Time steps: {len(merged.time)}")
    print(f"  Spatial extent: lat {float(merged.lat.min()):.2f}–{float(merged.lat.max()):.2f}, "
          f"lon {float(merged.lon.min()):.2f}–{float(merged.lon.max()):.2f}")


def main():
    print("=" * 60)
    print("Multi-Variable Data Merge — Ecosystem Collapse EWS Study")
    print("=" * 60)

    for area_name, bbox in STUDY_AREAS.items():
        merge_area(area_name, bbox)

    print("\n" + "=" * 60)
    print("Merge complete.")
    print(f"Output: {OUT_DIR}")
    print("\nFiles created per study area:")
    print("  merged_pixel_{area}.nc          — full pixel dataset")
    print("  merged_regions_{area}.csv       — regional time series")
    print("  region_masks_{area}.nc          — region ID masks")
    print("  region_metadata_{area}.csv      — region center coords")
    print("=" * 60)


if __name__ == "__main__":
    main()
