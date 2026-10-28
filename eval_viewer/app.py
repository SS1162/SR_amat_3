"""SR Eval Viewer - a local developer tool.

Run from the repo root:

    python -m eval_viewer.app

Open http://127.0.0.1:8051
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``algo`` and ``inference`` are imported from the repo root (also when run as a script).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dash_bootstrap_components as dbc
from dash import Dash
from flask import abort, send_from_directory

from eval_viewer.callbacks import register_callbacks
from eval_viewer.config import APP_NAME, CACHE_DIR, PORT
from eval_viewer.layout import build_layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title=APP_NAME,
           suppress_callback_exceptions=True)
server = app.server
app.layout = build_layout()
register_callbacks(app)


@server.route("/cache/<path:sub>")
def cached_image(sub: str):
    """Previews from the eval cache; send_from_directory rejects paths outside it."""
    if not sub.endswith(".png"):
        abort(404)
    return send_from_directory(CACHE_DIR, sub, max_age=3600)


if __name__ == "__main__":
    # No reloader: it would start a second process with its own job thread.
    app.run(host="127.0.0.1", port=PORT, debug=False)
