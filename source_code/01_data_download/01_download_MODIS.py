"""
MODIS Data Download Script (Batched)
======================================
Downloads MODIS products via NASA earthaccess in small batches
so progress is visible and downloads are resumable.

Products:
  - MOD13A3 v6.1  : NDVI / EVI (1km, monthly)
  - MOD11A2 v6.1  : Land Surface Temperature (1km, 8-day)
  - MOD16A2 v6.1  : Evapotranspiration (500m, 8-day)
  - MCD64A1 v6.1  : Burned Area (500m, monthly)
"""

import os
import earthaccess
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw" / "MODIS"

# ── Study period ───────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = "2023-12-31"

# ── Study areas ────────────────────────────────────────────────────────────
STUDY_AREAS = {
    "amazon":      (-73.0, -18.0, -44.0,  5.0),
    "sahel":       (-17.0,  10.0,  40.0, 20.0),
    "east_africa": ( 33.0,  -5.0,  51.0, 15.0),
}

# ── MODIS products ─────────────────────────────────────────────────────────
PRODUCTS = [
    {"short_name": "MOD13A3", "version": "061", "description": "NDVI/EVI monthly 1km"},
    {"short_name": "MOD11A2", "version": "061", "description": "Land Surface Temp 8-day 1km"},
    {"short_name": "MOD16A2", "version": "061", "description": "Evapotranspiration 8-day 500m"},
    {"short_name": "MCD64A1", "version": "061", "description": "Burned Area monthly 500m"},
]

BATCH_SIZE = 50   # Download 50 granules at a time — shows progress clearly


def search_and_download(product, area_name, bbox):
    short_name = product["short_name"]
    version    = product["version"]

    out_dir = RAW_DIR / short_name / area_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {short_name} | {area_name} | {product['description']}")
    print(f"{'='*60}")

    # Search all granules
    print("  Searching granules...", end=" ", flush=True)
    results = earthaccess.search_data(
        short_name   = short_name,
        version      = version,
        temporal     = (START_DATE, END_DATE),
        bounding_box = bbox,
    )
    print(f"{len(results)} found.")

    if not results:
        print("  Nothing to download.")
        return

    # Check already downloaded
    existing = {f.stem for f in out_dir.glob("*.hdf")}
    existing.update({f.stem for f in out_dir.glob("*.h5")})

    # Filter to only missing granules
    to_download = []
    for g in results:
        # Get filename from granule metadata
        try:
            fname = g["umm"]["RelatedUrls"][0]["URL"].split("/")[-1]
            stem  = Path(fname).stem
            if stem not in existing:
                to_download.append(g)
        except Exception:
            to_download.append(g)

    print(f"  Already downloaded: {len(existing)}  |  Remaining: {len(to_download)}")

    if not to_download:
        print("  All granules already downloaded. Skipping.")
        return

    # Download in batches
    total_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE
    downloaded_total = 0

    for i in range(0, len(to_download), BATCH_SIZE):
        batch      = to_download[i : i + BATCH_SIZE]
        batch_num  = i // BATCH_SIZE + 1
        print(f"\n  Batch {batch_num}/{total_batches} "
              f"({len(batch)} granules)...", end=" ", flush=True)
        try:
            files = earthaccess.download(batch, local_path=str(out_dir))
            downloaded_total += len(files)
            print(f"✓ {downloaded_total}/{len(to_download)} total")
        except Exception as e:
            print(f"\n  [WARN] Batch {batch_num} error: {e} — continuing...")

    print(f"\n  [DONE] {short_name}/{area_name}: {downloaded_total} files downloaded")

    # Show folder size
    files = list(out_dir.glob("*.hdf")) + list(out_dir.glob("*.h5"))
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"  Folder total: {len(files)} files | {total_mb:.0f} MB")


def main():
    print("=" * 60)
    print("MODIS Download — Ecosystem Collapse EWS Study")
    print(f"Products: {len(PRODUCTS)} | Areas: {len(STUDY_AREAS)}")
    print(f"Period  : {START_DATE} → {END_DATE}")
    print("=" * 60)

    # Authenticate
    print("\nAuthenticating with NASA Earthdata...")
    earthaccess.login(strategy="netrc")
    print("Authenticated.\n")

    # Download all products × all areas
    for product in PRODUCTS:
        for area_name, bbox in STUDY_AREAS.items():
            search_and_download(product, area_name, bbox)

    # Final summary
    print("\n" + "=" * 60)
    print("MODIS download complete. Summary:")
    for product in PRODUCTS:
        for area_name in STUDY_AREAS:
            d = RAW_DIR / product["short_name"] / area_name
            if d.exists():
                files = list(d.glob("*.hdf")) + list(d.glob("*.h5"))
                mb = sum(f.stat().st_size for f in files) / (1024*1024)
                print(f"  {product['short_name']}/{area_name}: "
                      f"{len(files)} files | {mb:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
