from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import APP_NAME
from .components import topbar
from .inference.page import inference_page
from .tensorboard.page import tensorboard_page
from .training.page import training_page


def build_workspace_layout():
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


def build_login_layout():
    return dbc.Container(
        fluid=True,
        className="py-5",
        children=[
            dbc.Row(
                justify="center",
                children=[
                    dbc.Col(
                        md=5,
                        lg=4,
                        children=dbc.Card(
                            children=dbc.CardBody(
                                [
                                    html.H3("Sign in", className="mb-3"),
                                    dbc.Label("Email", html_for="login-email"),
                                    dbc.Input(id="login-email", type="email", placeholder="you@example.com", className="mb-3"),
                                    dbc.Label("Password", html_for="login-password"),
                                    dbc.Input(id="login-password", type="password", placeholder="Password", className="mb-3"),
                                    dbc.Button("Login", id="login-submit", color="primary", className="w-100"),
                                    html.Div(id="login-feedback", className="text-danger small mt-2"),
                                ]
                            )
                        ),
                    )
                ],
            )
        ],
    )


def build_layout():
    return html.Div(
        [
            dcc.Store(id="auth-state", storage_type="session"),
            html.Div(id="root-content"),
        ]
    )
