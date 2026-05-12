"""
config.py — Central configuration for Astrocyte U-Net-DC inference pipeline.

CHECKPOINT_PATH:
  - Set to None  → script will AUTO-SEARCH for any .pth file under the current
                   working directory (or the folder you run it from).
  - Set to a Path → use that exact file.

OUTPUT_DIR:
  - Set to None  → results are saved automatically next to the .pth file,
                   inside a subfolder called  "inference_results/".
  - Set to a Path → use that exact folder.
"""
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (None = auto-detect / auto-create)
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = None   # ← set to Path("your/model.pth") to pin a specific file
OUTPUT_DIR      = None   # ← set to Path("your/output/folder") to pin an output dir

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE      = 512
BASE_CHANNELS = 32
IN_CHANNELS   = 1
NUM_CLASSES   = 4

CLASS_NAMES   = ["Full", "Soma", "Process", "Spongiform"]

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLDS       = [0.5, 0.5, 0.35, 0.45]   # one per class
MIN_SOMA_AREA    = 500   # pixels
MIN_PROCESS_AREA = 200   # pixels

# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────
PROB_CMAP = "magma"
