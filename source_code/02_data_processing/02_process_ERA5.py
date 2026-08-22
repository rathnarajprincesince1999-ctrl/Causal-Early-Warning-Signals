"""
ERA5 Data Processing Script
=============================
Processes raw ERA5 NetCDF files:
  1. Load NetCDF files per study area
  2. Convert units (precipitation m → mm, temperature K → C)
  3. Compute monthly anomalies (deviation from 2000–2020 climatology)
  4. Regrid to common 0.05 degree grid
  5. Save clean processed NetCDF per study area

Variables processed:
  - Precipitation         : m/month → mm/month
  - 2m Temperature        : K → Celsius
  - Potential ET          : m → mm
  - Soil water layer 1    : m3/m3 (no conversion)
"""

import numpy as np
import xarray as xr
import zipfile
import io
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw" / "ERA5"
PROC_DIR = BASE_DIR / "data" / "processed" / "ERA5"
PROC_DIR.mkdir(parents=True, exist_ok=True)

STUDY_AREAS = ["amazon", "sahel", "east_africa"]

# ── Climatology baseline period ────────────────────────────────────────────
CLIM_START = 2000
CLIM_END   = 2020


def convert_units(ds):
    """Convert ERA5 raw units to standard scientific units."""

    # Precipitation: m → mm (multiply by 1000)
    if "tp" in ds:
        ds["tp"] = ds["tp"] * 1000.0
        ds["tp"].attrs["units"]     = "mm/month"
        ds["tp"].attrs["long_name"] = "Total Precipitation"

    # 2m Temperature: K → Celsius
    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] - 273.15
        ds["t2m"].attrs["units"]     = "degrees_Celsius"
        ds["t2m"].attrs["long_name"] = "2m Air Temperature"

    # Potential evaporation: m → mm (negative in ERA5 → flip sign)
    if "pev" in ds:
        ds["pev"] = ds["pev"] * -1000.0   # ERA5 pev is negative (flux convention)
        ds["pev"].attrs["units"]     = "mm/month"
        ds["pev"].attrs["long_name"] = "Potential Evaporation"

    # Soil water: m3/m3 — no conversion needed
    if "swvl1" in ds:
        ds["swvl1"].attrs["units"]     = "m3/m3"
        ds["swvl1"].attrs["long_name"] = "Volumetric Soil Water Layer 1 (0-7cm)"

    return ds


def compute_anomalies(ds, baseline_start=CLIM_START, baseline_end=CLIM_END):
    """
    Compute monthly anomalies relative to a climatological baseline.
    Anomaly = value - monthly_mean(baseline period)
    """
    anomaly_ds = xr.Dataset()
    baseline   = ds.sel(
        time=slice(f"{baseline_start}-01-01", f"{baseline_end}-12-31")
    )

    for var in ds.data_vars:
        # Monthly climatology (mean for each calendar month)
        clim = baseline[var].groupby("time.month").mean("time")

        # Anomaly = observed - climatology
        anom = ds[var].groupby("time.month") - clim
        anom.name               = f"{var}_anom"
        anom.attrs["units"]     = ds[var].attrs.get("units", "")
        anom.attrs["long_name"] = f"{ds[var].attrs.get('long_name', var)} Anomaly"
        anom.attrs["baseline"]  = f"{baseline_start}–{baseline_end} climatology"

        anomaly_ds[f"{var}_anom"] = anom

    return anomaly_ds


def regrid_to_common_grid(ds, resolution=0.05):
    """
    Interpolate ERA5 (0.25 deg) to common 0.05 deg grid.
    Uses linear interpolation — appropriate for reanalysis data.
    """
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())

    new_lat = np.arange(lat_min, lat_max + resolution, resolution)
    new_lon = np.arange(lon_min, lon_max + resolution, resolution)

    ds_regridded = ds.interp(
        latitude=new_lat,
        longitude=new_lon,
        method="linear",
    )
    return ds_regridded


def process_era5_area(area_name):
    """Process ERA5 data for one study area."""
    raw_file = RAW_DIR / area_name / f"ERA5_monthly_{area_name}_2000_2023.nc"

    if not raw_file.exists():
        print(f"  [SKIP] Raw file not found: {raw_file}")
        print(f"         Run 01_data_download/02_download_ERA5.py first.")
        return

    out_file = PROC_DIR / f"ERA5_processed_{area_name}.nc"

    if out_file.exists():
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {out_file.name} already exists ({size_mb:.1f} MB)")
        return

    print(f"\n  Processing ERA5 — {area_name}")
    print(f"  Input : {raw_file}")

    # CDS API downloads ERA5 as a ZIP containing two NetCDF files
    # One has t2m + swvl1, the other has tp + pev
    if zipfile.is_zipfile(raw_file):
        print("    Detected ZIP format — extracting NetCDF files...")
        dsets = []
        with zipfile.ZipFile(raw_file) as z:
            for name in z.namelist():
                if name.endswith(".nc"):
                    with z.open(name) as nc:
                        data = nc.read()
                        part = xr.open_dataset(io.BytesIO(data), engine="h5netcdf")
                        # Rename valid_time → time
                        if "valid_time" in part.dims:
                            part = part.rename({"valid_time": "time"})
                        # Drop ancillary coords that cause merge conflicts
                        drop_vars = [v for v in ["expver", "number"] if v in part]
                        if drop_vars:
                            part = part.drop_vars(drop_vars)
                        dsets.append(part)
        ds = xr.merge(dsets)
        print(f"    Merged from ZIP: {list(ds.data_vars)} | Time steps: {len(ds.time)}")
    else:
        ds = xr.open_dataset(raw_file, engine="netcdf4")
        print(f"    Loaded: {list(ds.data_vars)} | Time steps: {len(ds.time)}")

    # Rename coords if needed (ERA5 uses 'latitude'/'longitude')
    if "latitude" not in ds.dims and "lat" in ds.dims:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # Convert units
    print("    Converting units...")
    ds = convert_units(ds)

    # Regrid to 0.05 deg
    print("    Regridding to 0.05 degree...")
    ds = regrid_to_common_grid(ds, resolution=0.05)

    # Compute anomalies
    print("    Computing anomalies...")
    ds_anom = compute_anomalies(ds)

    # Merge original + anomalies
    ds_final = xr.merge([ds, ds_anom])

    # Add metadata
    ds_final.attrs["source"]      = "ERA5 Monthly Reanalysis — Copernicus/ECMWF"
    ds_final.attrs["study_area"]  = area_name
    ds_final.attrs["period"]      = "2000–2023"
    ds_final.attrs["resolution"]  = "0.05 degrees (regridded from 0.25)"
    ds_final.attrs["baseline"]    = f"Anomalies relative to {CLIM_START}–{CLIM_END}"

    # Save compressed
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
        for var in ds_final.data_vars
    }
    ds_final.to_netcdf(out_file, encoding=encoding)

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"    [SAVED] {out_file.name} ({size_mb:.1f} MB)")
    print(f"    Variables: {list(ds_final.data_vars)}")


def main():
    print("=" * 60)
    print("ERA5 Processing — Ecosystem Collapse EWS Study")
    print("=" * 60)

    for area_name in STUDY_AREAS:
        process_era5_area(area_name)

    print("\n" + "=" * 60)
    print("ERA5 processing complete.")
    print(f"Output: {PROC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
