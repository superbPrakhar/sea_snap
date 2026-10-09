"""Central configuration for the AI-Based Ocean Hazard Reporting System.

All paths, class definitions and model hyper-parameters live here so that the
oil-spill module, the cyclone module and the dashboard stay consistent.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Oil spill dataset: expects train/val/test splits, each with images/ and masks/
OIL_SPILL_DATA_DIR = DATA_DIR / "oil_spill"
OIL_SPILL_ZIP = DATA_DIR / "oil_spill.zip"  # optional: drop your dataset.zip here

CYCLONE_DATA_DIR = DATA_DIR / "cyclone"
CYCLONE_CSV = CYCLONE_DATA_DIR / "cyclone_data.csv"

OIL_SPILL_CKPT = CHECKPOINT_DIR / "oil_spill_unetpp.pth"
OIL_SPILL_META = CHECKPOINT_DIR / "oil_spill_unetpp.json"
CYCLONE_CKPT = CHECKPOINT_DIR / "cyclone_ann.pth"

# ---------------------------------------------------------------------------
# Oil spill segmentation
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Background", "Oil", "Others", "Water"]
NUM_CLASSES = len(CLASS_NAMES)

# RGB colours used by the dataset masks (same convention as the original
# project dataset: black=background, pink=oil, yellow=others, cyan=water).
COLOR_MAP = {
    (0, 0, 0): 0,          # Background
    (255, 0, 124): 1,      # Oil
    (255, 204, 51): 2,     # Others (ships, docks, ...)
    (51, 221, 255): 3,     # Water
}
CLASS_COLORS = list(COLOR_MAP.keys())

IMG_SIZE = 256          # training resolution (square)
ENCODER = "efficientnet-b0"   # default: light encoder, works on 4 GB GPUs.
                              # For the full proposal setup use "timm-efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"  # set to None for offline / synthetic-data training

BATCH_SIZE = 8
NUM_EPOCHS = 30
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-5

# Alert thresholds used by the dashboard (percent of oil-covered pixels).
# Calibrated on real aerial photos (Wikimedia validation set): mottled reef
# flats can read as a few % "oil", so warn only above 2%. Re-tune when
# training on the real project dataset.
OIL_ALERT_LOW = 2.0     # >= 2% oil   -> warning alert
OIL_ALERT_HIGH = 8.0    # >= 8% oil   -> high alert

# ---------------------------------------------------------------------------
# Cyclone prediction
# ---------------------------------------------------------------------------
CYCLONE_FEATURES = [
    "sea_surface_temp",   # deg C
    "pressure",           # hPa
    "humidity",           # %
    "wind_speed",         # m/s
    "wind_direction",     # deg
    "latitude",           # deg
    "longitude",          # deg
]
CYCLONE_TARGET = "cyclone"  # 1 = cyclone formed, 0 = no cyclone

# Risk levels for the dashboard
CYCLONE_RISK_LEVELS = [
    (0.25, "Low"),
    (0.50, "Moderate"),
    (0.75, "High"),
    (1.01, "Severe"),
]


def cyclone_risk_level(prob: float) -> str:
    for upper, name in CYCLONE_RISK_LEVELS:
        if prob < upper:
            return name
    return "Severe"
