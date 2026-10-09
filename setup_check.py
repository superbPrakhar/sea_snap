"""Environment sanity check - run this first if something does not work.

Usage:  python setup_check.py
"""

import importlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, BAD = "[ok]  ", "[MISS]"


def check_module(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
        try:
            from importlib.metadata import version
            ver = version(name)
        except Exception:
            ver = getattr(mod, "__version__", "")
        print(f"{OK} {name} {ver}")
        return True
    except ImportError:
        print(f"{BAD} {name}  ->  pip install {name}")
        return False


def main():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 9):
        print("WARNING: Python 3.9+ recommended.")

    print("\n-- Core dependencies --")
    missing = [m for m in ("torch", "torchvision", "cv2", "albumentations",
                           "segmentation_models_pytorch", "torchmetrics", "flask",
                           "pandas", "numpy", "matplotlib", "seaborn", "sklearn",
                           "PIL", "tqdm") if not check_module(m)]

    print("\n-- GPU --")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{OK} CUDA available: {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        else:
            print(f"{OK} torch works, but CUDA is NOT available - training will use CPU (slow).")
    except ImportError:
        pass

    print("\n-- Data --")
    from config import (CYCLONE_CSV, OIL_SPILL_CKPT, OIL_SPILL_DATA_DIR, CYCLONE_CKPT)
    train_imgs = OIL_SPILL_DATA_DIR / "train" / "images"
    print(f"{OK if train_imgs.exists() else BAD} oil-spill dataset "
          f"({train_imgs})" + ("" if train_imgs.exists() else
                               "  ->  python oil_spill/make_synthetic_dataset.py"))
    print(f"{OK if CYCLONE_CSV.exists() else BAD} cyclone dataset ({CYCLONE_CSV})"
          + ("" if CYCLONE_CSV.exists() else "  ->  python cyclone/make_dataset.py"))

    print("\n-- Checkpoints --")
    print(f"{OK if OIL_SPILL_CKPT.exists() else BAD} oil-spill model ({OIL_SPILL_CKPT})"
          + ("" if OIL_SPILL_CKPT.exists() else "  ->  python oil_spill/train.py"))
    print(f"{OK if CYCLONE_CKPT.exists() else BAD} cyclone model ({CYCLONE_CKPT})"
          + ("" if CYCLONE_CKPT.exists() else "  ->  python cyclone/train.py"))

    if missing:
        print(f"\nResult: {len(missing)} package(s) missing. Install them and re-run this check.")
        sys.exit(1)
    print("\nResult: environment looks good. Next: python train_all.py")


if __name__ == "__main__":
    main()
