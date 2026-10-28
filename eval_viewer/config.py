"""Paths and defaults of the eval viewer.

The dataset location can be overridden with the ``SR_EVAL_DATA`` environment variable
or from the UI. The cache can be moved with ``SR_EVAL_CACHE``.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "SR Eval Viewer"
PORT = 8051

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

# Model weights: the base model ships with the viewer; run checkpoints are discovered.
MODELS_DIR = EVAL_DIR / "models"
BASE_MODEL_PATH = MODELS_DIR / "base" / "spfi_x4_base.pt"
RUNS_DIR = REPO_ROOT / "runs"

CACHE_DIR = Path(os.environ.get("SR_EVAL_CACHE", EVAL_DIR / "eval_cache"))

DEFAULT_DATA_DIR = os.environ.get(
    "SR_EVAL_DATA",
    str(Path.home() / "Downloads" / "NanoVision-spfi" / "datasets" / "emps" / "EMPS"),
)

DEFAULT_N_IMAGES = 100
DEFAULT_SEED = 0
# HR crop side. 256 is the training patch; "full" uses the whole image. The model's
# attention makes run time grow faster than the area (CPU: ~2 s at 256, ~20 s at 512).
CROP_OPTIONS = ["256", "320", "512", "full"]
DEFAULT_CROP = "256"
# LR side must be even for the DTCWT, so the HR side must be a multiple of 2 * scale.
SIZE_MULTIPLE = 8

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
