"""
MASTER MODELLING PIPELINE
==========================
Causal Early Warning System for Ecosystem Collapse
— Nature Machine Intelligence Submission

Runs all 5 modelling components in sequence.

Usage:
    python RUN_MODELLING.py              # Run all components
    python RUN_MODELLING.py --step 1    # Classical EWS only
    python RUN_MODELLING.py --step 2    # Deep learning only
    python RUN_MODELLING.py --step 3    # Causal discovery only
    python RUN_MODELLING.py --step 4    # Risk mapping only
    python RUN_MODELLING.py --step 5    # Validation + figures only
"""

import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = {
    1: {
        "name":   "Classical Early Warning Signals (baseline)",
        "script": BASE_DIR / "03_ews_analysis" / "01_classical_ews.py",
    },
    2: {
        "name":   "Deep Learning EWS Detector (TCN + Attention)",
        "script": BASE_DIR / "04_deep_learning" / "01_train_ews_detector.py",
    },
    3: {
        "name":   "Causal Discovery (PCMCI+)",
        "script": BASE_DIR / "05_causal_discovery" / "01_pcmci_analysis.py",
    },
    4: {
        "name":   "Spatial Risk Mapping",
        "script": BASE_DIR / "06_risk_mapping" / "01_spatial_risk_maps.py",
    },
    5: {
        "name":   "Validation + Publication Figures",
        "scripts": [
            BASE_DIR / "07_validation_figures" / "01_validation.py",
            BASE_DIR / "07_validation_figures" / "02_figures.py",
        ],
    },
}


def run_script(script_path):
    print(f"\n  Running: {script_path.name}")
    print("  " + "-" * 50)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"\n  [ERROR] Script failed with code {result.returncode}")
        return False
    return True


def run_step(step_num):
    step = STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step['name']}")
    print(f"{'='*60}")

    scripts = step.get("scripts", [step.get("script")])
    for script in scripts:
        if not run_script(script):
            print(f"\n[STOPPED] Step {step_num} failed.")
            sys.exit(1)

    print(f"\n[DONE] Step {step_num} complete.")


def main():
    parser = argparse.ArgumentParser(description="Ecosystem Collapse EWS — Modelling Pipeline")
    parser.add_argument("--step", type=int, choices=range(1, 6),
                        help="Run specific step (1–5)")
    args = parser.parse_args()

    print("=" * 60)
    print("ECOSYSTEM COLLAPSE CAUSAL EWS — MODELLING PIPELINE")
    print("Nature Machine Intelligence Study")
    print("=" * 60)
    print("\nPipeline steps:")
    for num, step in STEPS.items():
        print(f"  Step {num}: {step['name']}")

    # Check prerequisites
    merged_dir = BASE_DIR / "data" / "processed" / "merged"
    if not any(merged_dir.glob("merged_regions_*.csv")):
        print("\n[ERROR] Merged data not found.")
        print("Run the data pipeline first: python RUN_PIPELINE.py")
        sys.exit(1)

    if args.step:
        run_step(args.step)
    else:
        print("\nRunning all steps sequentially...")
        for step_num in sorted(STEPS.keys()):
            run_step(step_num)

    print("\n" + "=" * 60)
    print("Modelling pipeline complete.")
    print("\nKey outputs:")
    print("  results/ews_classical/      — baseline EWS metrics")
    print("  results/deep_learning/      — trained model + predictions")
    print("  results/causal/             — causal networks")
    print("  results/risk_maps/          — spatial risk maps")
    print("  results/validation/         — validation scores")
    print("  results/figures/            — publication figures (PDF + PNG)")
    print("=" * 60)


if __name__ == "__main__":
    main()
