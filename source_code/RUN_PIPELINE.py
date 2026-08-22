"""
MASTER PIPELINE RUNNER
========================
Causal Early Warning System for Ecosystem Collapse
— Nature Machine Intelligence Submission

Run this script to execute the full pipeline step by step.
Each phase can be run independently or all together.

Usage:
    python RUN_PIPELINE.py              # Run all phases
    python RUN_PIPELINE.py --phase 1   # Run only Phase 1 (setup check)
    python RUN_PIPELINE.py --phase 2   # Run only Phase 2 (download)
    python RUN_PIPELINE.py --phase 3   # Run only Phase 3 (processing)
    python RUN_PIPELINE.py --phase 4   # Run only Phase 4 (merge)
"""

import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PHASES = {
    1: {
        "name":   "Environment Setup Check",
        "script": BASE_DIR / "00_setup" / "setup_environment.py",
    },
    2: {
        "name":   "Data Download",
        "scripts": [
            BASE_DIR / "01_data_download" / "01_download_MODIS.py",
            BASE_DIR / "01_data_download" / "02_download_ERA5.py",
            BASE_DIR / "01_data_download" / "03_download_ESA_CCI_SoilMoisture.py",
        ],
    },
    3: {
        "name":   "Data Processing",
        "scripts": [
            BASE_DIR / "02_data_processing" / "01_process_MODIS.py",
            BASE_DIR / "02_data_processing" / "02_process_ERA5.py",
        ],
    },
    4: {
        "name":   "Data Merge",
        "scripts": [
            BASE_DIR / "02_data_processing" / "03_merge_all_variables.py",
        ],
    },
}


def run_script(script_path):
    """Run a Python script and stream output."""
    print(f"\n  Running: {script_path.name}")
    print("  " + "-" * 50)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print(f"\n  [ERROR] Script failed with code {result.returncode}")
        return False
    return True


def run_phase(phase_num):
    """Run all scripts in a phase."""
    phase = PHASES[phase_num]
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}: {phase['name']}")
    print(f"{'='*60}")

    scripts = phase.get("scripts", [phase.get("script")])
    for script in scripts:
        if not run_script(script):
            print(f"\n[STOPPED] Phase {phase_num} failed. Fix errors before continuing.")
            sys.exit(1)

    print(f"\n[DONE] Phase {phase_num} complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Ecosystem Collapse EWS Pipeline Runner"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific phase only (1=setup, 2=download, 3=process, 4=merge)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ECOSYSTEM COLLAPSE CAUSAL EWS — FULL PIPELINE")
    print("Nature Machine Intelligence Study")
    print("=" * 60)
    print("\nPipeline phases:")
    for num, phase in PHASES.items():
        print(f"  Phase {num}: {phase['name']}")

    if args.phase:
        run_phase(args.phase)
    else:
        print("\nRunning all phases sequentially...")
        for phase_num in sorted(PHASES.keys()):
            run_phase(phase_num)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Next step: Run 03_ews_analysis/ scripts")
    print("=" * 60)


if __name__ == "__main__":
    main()
