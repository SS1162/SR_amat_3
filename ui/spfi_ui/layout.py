from dash import html
import dash_bootstrap_components as dbc

from .config import APP_NAME
from .components import topbar
from .inference.page import inference_page
from .tensorboard.page import tensorboard_page
from .training.page import training_page


def build_layout():
    return html.Div(
        id="app-shell",
        className="app-shell theme-dark",
        children=[
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
