"""
MOD16A2GF Missing Years Download
==================================
Downloads MOD16A2GF (gap-filled ET) for 2000–2020.
MOD16A2 v6.1 only starts from 2021; MOD16A2GF covers full 2000–2023.
2021–2023 already downloaded as MOD16A2; this gets 2000–2020 as MOD16A2GF.
"""

import earthaccess
from pathlib import Path
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parents[1]
RAW_DIR    = BASE_DIR / "data" / "raw" / "MODIS"

START_DATE = "2000-01-01"
END_DATE   = "2020-12-31"

STUDY_AREAS = {
    "amazon":      (-73.0, -18.0, -44.0,  5.0),
    "sahel":       (-17.0,  10.0,  40.0, 20.0),
    "east_africa": ( 33.0,  -5.0,  51.0, 15.0),
}

BATCH_SIZE = 50

# Use MOD16A2GF (gap-filled) which covers 2000–present
SHORT_NAME = "MOD16A2GF"


def download_mod16a2(area_name, bbox):
    # Store in MOD16A2 folder so processing script finds it alongside 2021-2023 files
    out_dir = RAW_DIR / "MOD16A2" / area_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {SHORT_NAME} | {area_name} | 2000–2020")
    print(f"{'='*60}")

    print("  Searching granules...", end=" ", flush=True)
    results = earthaccess.search_data(
        short_name   = SHORT_NAME,
        version      = "061",
        temporal     = (START_DATE, END_DATE),
        bounding_box = bbox,
    )
    print(f"{len(results)} found.")

    if not results:
        print("  Nothing to download.")
        return

    # Skip already downloaded
    existing = {f.stem for f in out_dir.glob("*.hdf")}
    to_download = []
    for g in results:
        try:
            fname = g["umm"]["RelatedUrls"][0]["URL"].split("/")[-1]
            if Path(fname).stem not in existing:
                to_download.append(g)
        except Exception:
            to_download.append(g)

    print(f"  Already downloaded: {len(existing)}  |  Remaining: {len(to_download)}")

    if not to_download:
        print("  All granules already present. Skipping.")
        return

    total_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE
    downloaded = 0

    for i in range(0, len(to_download), BATCH_SIZE):
        batch     = to_download[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} granules)...",
              end=" ", flush=True)
        try:
            files = earthaccess.download(batch, local_path=str(out_dir))
            downloaded += len(files)
            print(f"✓ {downloaded}/{len(to_download)} total")
        except Exception as e:
            print(f"\n  [WARN] Batch {batch_num} error: {e} — continuing...")

    print(f"\n  [DONE] MOD16A2/{area_name}: {downloaded} new files downloaded")


def main():
    print("=" * 60)
    print("MOD16A2 Missing Years Download (2000–2020)")
    print("=" * 60)

    print("\nAuthenticating with NASA Earthdata...")
    earthaccess.login(strategy="netrc")
    print("Authenticated.\n")

    for area_name, bbox in STUDY_AREAS.items():
        download_mod16a2(area_name, bbox)

    print("\n" + "=" * 60)
    print(f"MOD16A2GF download complete (2000–2020).")
    print("=" * 60)


if __name__ == "__main__":
    main()
