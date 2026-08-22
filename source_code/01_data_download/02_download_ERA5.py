"""
ERA5 Data Download Script
==========================
Downloads ERA5 monthly reanalysis data from Copernicus Climate Data Store (CDS):
  - Total precipitation         (monthly means)
  - 2m air temperature          (monthly means)
  - Potential evapotranspiration (monthly means)
  - Soil water layer 1          (monthly means — 0-7cm depth)

Study period : 2000-01-01 to 2023-12-31
Study areas  : Amazon, Sahel, East Africa
Resolution   : 0.25 degree (~28km)
Format       : NetCDF

REQUIREMENTS:
  - CDS account (free): https://cds.climate.copernicus.eu/
  - Install CDS API key: https://cds.climate.copernicus.eu/api-how-to
    (creates ~/.cdsapirc with your UID and API key)
  - Run: pip install cdsapi
"""

import cdsapi
import os
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw" / "ERA5"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Study period ───────────────────────────────────────────────────────────
YEARS  = [str(y) for y in range(2000, 2024)]   # 2000–2023
MONTHS = [f"{m:02d}" for m in range(1, 13)]    # 01–12

# ── Study area bounding boxes ──────────────────────────────────────────────
# ERA5 area format: [lat_max, lon_min, lat_min, lon_max]  (N/W/S/E)
STUDY_AREAS = {
    "amazon": {
        "area":  [5.0, -73.0, -18.0, -44.0],
        "label": "Amazon Rainforest",
    },
    "sahel": {
        "area":  [20.0, -17.0, 10.0, 40.0],
        "label": "Sahel Drylands",
    },
    "east_africa": {
        "area":  [15.0, 33.0, -5.0, 51.0],
        "label": "East Africa Drylands",
    },
}

# ── Variables to download ──────────────────────────────────────────────────
VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "potential_evaporation",
    "volumetric_soil_water_layer_1",
]


def download_era5(area_name, area_config):
    """Download ERA5 monthly data for one study area."""
    out_dir = RAW_DIR / area_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"ERA5_monthly_{area_name}_2000_2023.nc"

    if out_file.exists():
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  [SKIP] {out_file.name} already exists ({size_mb:.1f} MB)")
        return

    print(f"\n{'='*60}")
    print(f"ERA5 Download — {area_config['label']}")
    print(f"Area (N/W/S/E): {area_config['area']}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Period: 2000–2023 (monthly)")
    print(f"Output: {out_file}")
    print(f"{'='*60}")

    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     VARIABLES,
            "year":         YEARS,
            "month":        MONTHS,
            "time":         "00:00",
            "area":         area_config["area"],
            "format":       "netcdf",
            "grid":         [0.25, 0.25],
        },
        str(out_file),
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"  [DONE] Saved {size_mb:.1f} MB → {out_file}")


def main():
    print("=" * 60)
    print("ERA5 Data Download — Ecosystem Collapse EWS Study")
    print("=" * 60)
    print("\nNOTE: Requires CDS API key in ~/.cdsapirc")
    print("      Register free at: https://cds.climate.copernicus.eu/\n")

    for area_name, area_config in STUDY_AREAS.items():
        try:
            download_era5(area_name, area_config)
        except Exception as e:
            print(f"  [ERROR] {area_name}: {e}")
            print("  Check your CDS API key and internet connection.")

    print("\n" + "=" * 60)
    print("ERA5 download complete.")
    print(f"Data saved to: {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
