from pathlib import Path

APP_NAME = "SPFI Studio"

# Folder that holds one sub-folder per training run (meta.json, metrics.jsonl, samples/)
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

REPO_ROOT = Path(__file__).resolve().parents[2]
# Dash DiskcacheManager storage; also holds the Inference tab's per-job stop flags
DASH_CACHE_DIR = REPO_ROOT / ".cache" / "dash"
# One sub-folder per inference job: input/ (saved uploads) and output/ (SR results)
INFERENCE_RUNS_DIR = REPO_ROOT / "tmp" / "inference"
