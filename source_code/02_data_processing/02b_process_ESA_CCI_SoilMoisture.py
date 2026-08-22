"""
ESA CCI Soil Moisture Processing Script — Sequential Version
=============================================================
Processes raw ESA CCI daily soil moisture NetCDF files year-by-year.
Skips corrupted files gracefully. No multiprocessing (avoids HDF5 deadlocks).

Steps per study area:
  1. For each year: validate files, load good ones, clip to bbox, monthly mean
  2. Concatenate all years
  3. Regrid to 0.05 degree
  4. Save compressed NetCDF
"""

import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw"  / "ESA_CCI_SoilMoisture"
PROC_DIR = BASE_DIR / "data" / "processed" / "ESA_CCI_SoilMoisture"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Study areas ────────────────────────────────────────────────────────────
STUDY_AREAS = {
    "amazon":      {"lon": (-73.0, -44.0), "lat": (-18.0,  5.0)},
    "sahel":       {"lon": (-17.0,  40.0), "lat": ( 10.0, 20.0)},
    "east_africa": {"lon": ( 33.0,  51.0), "lat": ( -5.0, 15.0)},
}

OUT_RESOLUTION = 0.05


def process_one_year(year, year_dir, lat_min, lat_max, lon_min, lon_max):
    """
    Load valid daily files for one year, clip to bbox, return monthly mean DataArray.
    ESA CCI lat is DESCENDING (90 → -90), so slice high→low.
    """
    nc_files = sorted(year_dir.glob("*.nc"))
    if not nc_files:
        return None

    # Validate each file by reading a data chunk — skip corrupted ones
    good_files = []
    for f in nc_files:
        try:
            with xr.open_dataset(f, engine="netcdf4") as ds:
                if "sm" in ds:
                    _ = ds["sm"].isel(time=0).values  # triggers actual read
                    good_files.append(f)
        except Exception:
            continue

    if not good_files:
        return None

    try:
        ds = xr.open_mfdataset(
            good_files,
            combine="by_coords",
            parallel=False,
            engine="netcdf4",
        )

        if "sm" not in ds:
            ds.close()
            return None

        da = ds["sm"]

        # ESA CCI lat is descending: slice from max→min
        da = da.sel(
            lat=slice(lat_max, lat_min),   # descending: high to low
            lon=slice(lon_min, lon_max),
        )

        if da.lat.size == 0 or da.lon.size == 0:
            ds.close()
            return None

        # Monthly mean
        da_monthly = da.resample(time="1MS").mean().load()
        ds.close()

        return da_monthly

    except Exception as e:
        return None


def process_esa_cci_area(area_name, bbox):
    out_file = PROC_DIR / f"ESA_CCI_SM_monthly_{area_name}.nc"

    if out_file.exists():
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {out_file.name} already exists ({size_mb:.1f} MB)")
        return

    print(f"\n{'='*60}")
    print(f"Processing ESA CCI Soil Moisture — {area_name}")
    print(f"{'='*60}")

    lon_min, lon_max = bbox["lon"]
    lat_min, lat_max = bbox["lat"]

    # Collect available years
    year_dirs = []
    for year in range(2000, 2023):
        yd = RAW_DIR / str(year)
        if yd.exists() and any(yd.glob("*.nc")):
            year_dirs.append((year, yd))

    if not year_dirs:
        print(f"  [ERROR] No raw files found in {RAW_DIR}")
        return

    print(f"  Processing {len(year_dirs)} years sequentially...")

    yearly_results = []
    for year, year_dir in tqdm(year_dirs, desc="  Years", unit="yr"):
        da = process_one_year(year, year_dir, lat_min, lat_max, lon_min, lon_max)
        if da is not None:
            yearly_results.append(da)

    valid = [da for da in yearly_results if da is not None]
    if not valid:
        print(f"  [ERROR] No valid data for any year in {area_name}")
        return

    print(f"  Merging {len(valid)} years of monthly data...")
    combined = xr.concat(valid, dim="time").sortby("time")

    # Drop all-NaN time steps
    combined = combined.where(combined != combined.attrs.get("_FillValue", -9999.0))
    combined = combined.dropna(dim="time", how="all")

    if len(combined.time) == 0:
        print(f"  [ERROR] All time steps are NaN after merging.")
        return

    # Regrid to 0.05 degree (lat ascending for output)
    print("  Regridding to 0.05 degree...")
    new_lat = np.arange(lat_min, lat_max + OUT_RESOLUTION / 2, OUT_RESOLUTION)
    new_lon = np.arange(lon_min, lon_max + OUT_RESOLUTION / 2, OUT_RESOLUTION)

    # Sort lat ascending before interpolation
    combined = combined.sortby("lat")
    combined = combined.interp(lat=new_lat, lon=new_lon, method="linear")

    combined.name = "sm"
    combined.attrs["units"]      = "m3/m3"
    combined.attrs["long_name"]  = "Volumetric Soil Moisture (0-5cm)"
    combined.attrs["source"]     = "ESA CCI Soil Moisture v8.1 COMBINED"
    combined.attrs["study_area"] = area_name
    combined.attrs["processing"] = "Daily to monthly mean, regridded to 0.05 deg"

    print("  Saving...")
    encoding = {
        "sm": {
            "zlib": True, "complevel": 4,
            "dtype": "float32", "_FillValue": -9999.0,
        }
    }
    combined.to_dataset(name="sm").to_netcdf(out_file, encoding=encoding)

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"  [SAVED] {out_file.name} ({size_mb:.1f} MB)")
    print(f"  Time steps: {len(combined.time)}")
    print(f"  Lat: {float(combined.lat.min()):.2f} to {float(combined.lat.max()):.2f}")
    print(f"  Lon: {float(combined.lon.min()):.2f} to {float(combined.lon.max()):.2f}")


def main():
    print("=" * 60)
    print("ESA CCI Soil Moisture Processing")
    print("Sequential mode — handles corrupt files safely")
    print("=" * 60)

    for area_name, bbox in STUDY_AREAS.items():
        process_esa_cci_area(area_name, bbox)

    print("\n" + "=" * 60)
    print("ESA CCI Soil Moisture processing complete.")
    print(f"Output: {PROC_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
