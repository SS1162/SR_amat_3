from pathlib import Path

APP_NAME = "SPFI Studio"

# Folder that holds one sub-folder per training run (meta.json, metrics.jsonl, samples/)
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
