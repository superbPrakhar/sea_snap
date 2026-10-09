"""One-click setup: generate demo data (if missing) and train both models.

Usage:
    python train_all.py                 # full training with defaults
    python train_all.py --quick         # fast smoke-test (small models, few epochs)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

OIL_DATA = ROOT / "data" / "oil_spill"
CYCLONE_CSV = ROOT / "data" / "cyclone" / "cyclone_data.csv"
OIL_CKPT = ROOT / "checkpoints" / "oil_spill_unetpp.pth"
CYCLONE_CKPT = ROOT / "checkpoints" / "cyclone_ann.pth"


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run([PY, "-m"] + cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test: synthetic data + 3 epochs, light encoder")
    ap.add_argument("--skip-data", action="store_true", help="do not generate demo data")
    args = ap.parse_args()

    # 1. Data ----------------------------------------------------------------
    if not args.skip_data:
        if not (OIL_DATA / "train" / "images").exists():
            run(["oil_spill.make_synthetic_dataset"])
        else:
            print("Oil-spill dataset already present, skipping generation.")
        if (ROOT / "test_images").exists():
            run(["oil_spill.make_real_patches"])
        if not CYCLONE_CSV.exists():
            run(["cyclone.make_dataset"])
        else:
            print("Cyclone dataset already present, skipping generation.")

    # 2. Oil spill model -----------------------------------------------------
    if args.quick:
        run(["oil_spill.train", "--epochs", "3", "--encoder", "efficientnet-b0",
             "--encoder-weights", "None", "--patience", "3"])
    else:
        run(["oil_spill.train"])

    # 3. Cyclone model -------------------------------------------------------
    if args.quick:
        run(["cyclone.train", "--epochs", "10", "--patience", "5"])
    else:
        run(["cyclone.train"])

    # 4. Evaluation ----------------------------------------------------------
    if OIL_CKPT.exists():
        run(["oil_spill.evaluate", "--split", "test"])

    print("\nAll done! Start the dashboard with:\n"
          "    python dashboard/app.py")


if __name__ == "__main__":
    main()
