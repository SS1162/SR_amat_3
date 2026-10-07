from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import APP_NAME
from .components import topbar
from .pages import inference_page, tensorboard_page, training_page


def build_layout():
    return html.Div(
        id="app-shell",
        className="app-shell theme-dark",
        children=[
            dcc.Interval(id="fake-train-timer", interval=700, n_intervals=0, disabled=True),
            topbar(),
            html.Main(
                className="main-stage",
                children=[
                    dbc.Tabs(
                        [
                            dbc.Tab(training_page(), label="Training", tab_id="training"),
                            dbc.Tab(inference_page(), label="Inference", tab_id="inference"),
                            dbc.Tab(tensorboard_page(), label="TensorBoard", tab_id="tensorboard"),
                        ],
                        id="main-tabs",
                        active_tab="training",
                        className="main-tabs",
                    )
                ],
            ),
            html.Footer(
                [html.Span(APP_NAME), html.Span("Dash + dash-bootstrap-components UI prototype")],
                className="footer",
            ),
        ],
    )
