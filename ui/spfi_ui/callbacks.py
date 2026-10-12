from pathlib import Path
import subprocess
import sys

from dash import Input, Output

from .config import RUNS_DIR

REPO_ROOT = RUNS_DIR.parent


def _run_osascript(script):
    if sys.platform != "darwin" or not subprocess.run(["which", "osascript"], capture_output=True, text=True).returncode == 0:
        return None

    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def choose_folder(prompt_text):
    if sys.platform == "darwin":
        script = f'POSIX path of (choose folder with prompt "{prompt_text}")'
        return _run_osascript(script)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=prompt_text)
        root.destroy()
        return selected or None
    except Exception:
        return None


def choose_file(prompt_text):
    if sys.platform == "darwin":
        script = f'POSIX path of (choose file with prompt "{prompt_text}")'
        return _run_osascript(script)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=prompt_text)
        root.destroy()
        return selected or None
    except Exception:
        return None


def _is_creatable_directory(target: Path) -> bool:
    if target.exists():
        return target.is_dir()

    cursor = target
    while True:
        if cursor.exists() and not cursor.is_dir():
            return False
        parent = cursor.parent
        if parent == cursor:
            return True
        cursor = parent


def output_directory_feedback(path_value):
    if not path_value or not path_value.strip():
        return False, "soft-input is-invalid", "Directory path is required.", "field-feedback field-feedback-invalid"

    target = Path(path_value.strip()).expanduser()
    if not target.is_absolute():
        target = REPO_ROOT / target

    if target.exists() and not target.is_dir():
        return False, "soft-input is-invalid", "Path points to a file, not a directory.", "field-feedback field-feedback-invalid"

    if not _is_creatable_directory(target):
        return False, "soft-input is-invalid", "Parent directory does not exist or is not a folder.", "field-feedback field-feedback-invalid"

    if target.exists():
        feedback = "Directory exists."
    else:
        feedback = "Directory will be created when training starts."
    return True, "soft-input is-valid", feedback, "field-feedback field-feedback-valid"


def register_app_callbacks(app):
    @app.callback(
        Output("app-shell", "className"),
        Input("theme-toggle", "value"),
    )
    def toggle_theme(theme_value):
        if theme_value and "light" in theme_value:
            return "app-shell theme-light"
        return "app-shell theme-dark"
