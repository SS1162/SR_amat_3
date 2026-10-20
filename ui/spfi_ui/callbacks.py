from pathlib import Path
import subprocess
import sys

from dash import Input, Output, State, no_update

from .config import LOGIN_EMAIL, LOGIN_PASSWORD, RUNS_DIR
from .layout import build_login_layout, build_workspace_layout

REPO_ROOT = RUNS_DIR.parent


def choose_folder(prompt_text):
    if sys.platform != "darwin":
        return None
    apple_script = f'POSIX path of (choose folder with prompt "{prompt_text}")'
    result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def choose_file(prompt_text):
    if sys.platform != "darwin":
        return None
    apple_script = f'POSIX path of (choose file with prompt "{prompt_text}")'
    result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def output_directory_feedback(path_value):
    if not path_value or not path_value.strip():
        return False, "soft-input is-invalid", "Directory path is required.", "field-feedback field-feedback-invalid"

    target = Path(path_value.strip()).expanduser()
    if not target.is_absolute():
        target = REPO_ROOT / target

    if target.exists() and not target.is_dir():
        return False, "soft-input is-invalid", "Path points to a file, not a directory.", "field-feedback field-feedback-invalid"

    parent = target if target.exists() else target.parent
    if not parent.exists() or not parent.is_dir():
        return False, "soft-input is-invalid", "Parent directory does not exist.", "field-feedback field-feedback-invalid"

    if target.exists():
        feedback = "Directory exists."
    else:
        feedback = "Directory will be created when training starts."
    return True, "soft-input is-valid", feedback, "field-feedback field-feedback-valid"


def register_app_callbacks(app):
    @app.callback(Output("root-content", "children"), Input("auth-state", "data"))
    def render_root_content(auth_state):
        if auth_state and auth_state.get("authenticated"):
            return build_workspace_layout()
        return build_login_layout()

    @app.callback(
        Output("auth-state", "data"),
        Output("login-feedback", "children"),
        Output("login-feedback", "className"),
        Input("login-submit", "n_clicks"),
        State("login-email", "value"),
        State("login-password", "value"),
        prevent_initial_call=True,
    )
    def login_user(_, email, password):
        if not email or not password:
            return no_update, "Email and password are required.", "text-danger small mt-2"

        if not LOGIN_EMAIL or not LOGIN_PASSWORD:
            return (
                no_update,
                "Authentication is not configured. Set APP_LOGIN_EMAIL and APP_LOGIN_PASSWORD.",
                "text-danger small mt-2",
            )

        if email.strip().lower() == LOGIN_EMAIL.strip().lower() and password == LOGIN_PASSWORD:
            return {"authenticated": True}, "", "d-none"

        return no_update, "Invalid email or password.", "text-danger small mt-2"

    @app.callback(
        Output("auth-state", "data", allow_duplicate=True),
        Input("logout-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def logout_user(n_clicks):
        if not n_clicks:
            return no_update
        return {"authenticated": False}

    @app.callback(
        Output("app-shell", "className"),
        Input("theme-toggle", "value"),
    )
    def toggle_theme(theme_value):
        if theme_value and "light" in theme_value:
            return "app-shell theme-light"
        return "app-shell theme-dark"
