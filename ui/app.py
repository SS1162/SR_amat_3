from dash import Dash
import dash_bootstrap_components as dbc

from spfi_ui.callbacks import register_callbacks
from spfi_ui.config import APP_NAME
from spfi_ui.layout import build_layout


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=APP_NAME,
)
server = app.server

app.layout = build_layout()
register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
