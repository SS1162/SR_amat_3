import socket

from dash import Dash
import dash_bootstrap_components as dbc

from spfi_ui.config import APP_NAME
from spfi_ui.inference.callbacks import register_inference_callbacks
from spfi_ui.layout import build_layout
from spfi_ui.tensorboard.callbacks import register_tensorboard_callbacks
from spfi_ui.training.callbacks import register_training_callbacks


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=APP_NAME,
)
server = app.server

app.layout = build_layout()
register_training_callbacks(app)
register_inference_callbacks(app)
register_tensorboard_callbacks(app)


def _available_port(start_port=8050, host="127.0.0.1", max_attempts=20):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=_available_port())
