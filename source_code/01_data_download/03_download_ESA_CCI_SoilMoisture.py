"""
ESA CCI Soil Moisture Download Script
======================================
Downloads ESA CCI Combined Soil Moisture v8.1 (PASSIVE + ACTIVE combined)
  - Daily surface soil moisture (0–5 cm)
  - Resolution: 0.25 degree
  - Format: NetCDF
  - Period: 2000–2023

Source: ESA Climate Change Initiative — Soil Moisture
  - Direct FTP: https://www.esa-soilmoisture-cci.org/
  - CEDA Archive: https://data.ceda.ac.uk/neodc/esacci/soil_moisture/

REQUIREMENTS:
  - CEDA account (free): https://services.ceda.ac.uk/cedasite/register/info/
  - FTP credentials for data.ceda.ac.uk
  - OR use the ESA CCI Open Data Portal

Two download methods provided:
  Method A: Direct HTTPS from OpenDAP/CEDA (recommended)
  Method B: Manual download instructions for bulk FTP
"""

import os
import sys
import requests
import ftplib
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import time

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw" / "ESA_CCI_SoilMoisture"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Study areas (for subsetting after download) ───────────────────────────
STUDY_AREAS = {
    "amazon":      {"lon": (-73.0, -44.0), "lat": (-18.0, 5.0)},
    "sahel":       {"lon": (-17.0,  40.0), "lat": (10.0, 20.0)},
    "east_africa": {"lon": ( 33.0,  51.0), "lat": (-5.0, 15.0)},
}

# ── ESA CCI product info ───────────────────────────────────────────────────
# Version 8.1 COMBINED (active + passive merged — highest quality)
PRODUCT_VERSION = "v08.1"
PRODUCT_TYPE    = "COMBINED"  # OPTIONS: ACTIVE, PASSIVE, COMBINED

# Base URL for ESA CCI Soil Moisture via CEDA THREDDS/OpenDAP
# Annual files available per year
CEDA_BASE_URL = (
    "https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/"
    f"COMBINED/{PRODUCT_VERSION}/"
)

# Alternative: ESA Open Data Portal direct link
ESA_DIRECT_URL = (
    "https://catalogue.ceda.ac.uk/uuid/b810601740bd4848b0d7965e6d83d26c"
)


def download_file_https(url, out_path, session=None):
    """Download a single file via HTTPS with progress bar."""
    if out_path.exists():
        return True
    try:
        s = session or requests.Session()
        response = s.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            total = int(response.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm(
                desc=out_path.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            return True
        else:
            print(f"  HTTP {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def download_esa_cci_annual(year, session):
    """
    Download ESA CCI daily soil moisture files for one year.
    Files are named: ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-YYYYMMDD000000-fv08.1.nc
    """
    year_dir = RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Year {year}:")

    # Build dates for the year
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    success = 0
    for date in tqdm(dates, desc=f"  {year}", unit="day"):
        datestr  = date.strftime("%Y%m%d")
        filename = (
            f"ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-"
            f"{datestr}000000-fv08.1.nc"
        )
        url      = CEDA_BASE_URL + f"{year}/{filename}"
        out_path = year_dir / filename

        if download_file_https(url, out_path, session):
            success += 1
        else:
            # Small delay before retry to avoid rate limiting
            time.sleep(0.5)

    print(f"    Downloaded {success}/{len(dates)} daily files for {year}")
    return success


def print_manual_download_instructions():
    """
    Print manual download instructions for users who prefer bulk FTP.
    This is the fallback if HTTPS automated download fails.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ESA CCI SOIL MOISTURE — MANUAL DOWNLOAD             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  If automated download fails, use one of these methods:     ║
║                                                              ║
║  METHOD 1: CEDA Archive (recommended)                       ║
║  ─────────────────────────────────────                      ║
║  1. Register free at: https://services.ceda.ac.uk/          ║
║  2. Browse to:                                               ║
║     https://data.ceda.ac.uk/neodc/esacci/soil_moisture/     ║
║     data/daily_files/COMBINED/v08.1/                        ║
║  3. Download years 2000–2023                                 ║
║  4. Save to: data/raw/ESA_CCI_SoilMoisture/                 ║
║                                                              ║
║  METHOD 2: ESA CCI Open Data Portal                         ║
║  ──────────────────────────────────                         ║
║  1. Go to: https://www.esa-soilmoisture-cci.org/             ║
║  2. Select: COMBINED v8.1, 2000–2023                        ║
║  3. Define bounding box for your study area                  ║
║  4. Download NetCDF files                                    ║
║                                                              ║
║  METHOD 3: Climate Data Store (Copernicus)                  ║
║  ─────────────────────────────────────────                  ║
║  ESA CCI SM is also available via CDS:                      ║
║  cds.climate.copernicus.eu → Search "soil moisture"         ║
║  Dataset: "Soil moisture gridded data from 1978 to present" ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def get_credentials():
    """
    Get CEDA credentials from (in priority order):
      1. Environment variables CEDA_USERNAME / CEDA_PASSWORD
      2. ~/.ceda_credentials file  (two lines: username\\npassword)
      3. Interactive prompt (fallback)
    Returns (username, password) or (None, None) to skip.
    """
    import getpass

    # 1. Environment variables
    username = os.environ.get("CEDA_USERNAME", "").strip()
    password = os.environ.get("CEDA_PASSWORD", "").strip()
    if username and password:
        print("  Using credentials from environment variables.")
        return username, password

    # 2. Credentials file
    cred_file = Path.home() / ".ceda_credentials"
    if cred_file.exists():
        lines = cred_file.read_text().strip().splitlines()
        if len(lines) >= 2:
            print(f"  Using credentials from {cred_file}")
            return lines[0].strip(), lines[1].strip()

    # 3. Interactive prompt
    print("NOTE: Credentials can also be stored in ~/.ceda_credentials")
    print("      (line 1: username, line 2: password)")
    print("      or set env vars CEDA_USERNAME and CEDA_PASSWORD\n")
    username = input("CEDA username (or press Enter to skip): ").strip()
    if not username:
        return None, None
    password = getpass.getpass("CEDA password: ")
    return username, password


def main():
    print("=" * 60)
    print("ESA CCI Soil Moisture Download")
    print(f"Product: COMBINED {PRODUCT_VERSION} | Daily | 0.25deg")
    print(f"Period : 2000–2023")
    print("=" * 60)

    print("\nAttempting automated download via CEDA archive...\n")

    username, password = get_credentials()

    if not username:
        print_manual_download_instructions()
        return

    session = requests.Session()
    session.auth = (username, password)

    total_success = 0
    for year in range(2000, 2024):
        success = download_esa_cci_annual(year, session)
        total_success += success

    print("\n" + "=" * 60)
    print(f"ESA CCI Soil Moisture download complete.")
    print(f"Total files downloaded: {total_success}")
    print(f"Data saved to: {RAW_DIR}")
    print("=" * 60)

    if total_success == 0:
        print("\nAutomated download failed. Showing manual instructions:")
        print_manual_download_instructions()


if __name__ == "__main__":
    main()
