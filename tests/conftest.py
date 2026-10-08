import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))   # app.py imports the package as top-level ``spfi_ui``
