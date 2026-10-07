"""TensorBoard tab: live monitoring of a training run.

Layout only. Every element that will change at runtime has an id starting with "tb-";
the callbacks that fill them live in callbacks.py.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ..components import section_header, status_pill

REFRESH_MS = 2500
PLAY_MS = 600
SAMPLE_IDS = [1, 2, 3, 4]
IMAGE_TILES = [
    ("lr", "LR input"),
    ("bicubic", "Bicubic"),
    ("sr", "SR · model"),
    ("hr", "HR · ground truth"),
    ("error", "Error |SR − HR|"),
]


def empty_figure(message="Waiting for data"):
    """Dark, transparent Plotly figure used before any data arrives."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=16, t=16, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=message, showarrow=False, font=dict(color="#6f849e", size=13))],
    )
    return fig


def _stat_card(card_id, label, accent):
    """Like components.metric_card, but with ids so callbacks can update value and hint."""
    return dbc.Card(
        className=f"metric-card metric-{accent}",
        children=dbc.CardBody(
            [
                html.Div(label, className="metric-label"),
                html.Div("—", id=f"tb-{card_id}-value", className="metric-value"),
                html.Div("No data yet", id=f"tb-{card_id}-hint", className="metric-hint"),
                html.Div(id=f"tb-{card_id}-extra"),
            ]
        ),
    )


def _status_cards():
    return html.Div(
        className="tb-stats-grid",
        children=[
            _stat_card("iter", "Iteration", "cyan"),
            _stat_card("best", "Best PSNR", "green"),
            _stat_card("val", "Val loss", "violet"),
            _stat_card("patience", "Early stop", "amber"),
            _stat_card("cost", "Model cost", "cyan"),
        ],
    )


def _chart_card(kicker, title, graph_id, controls=None):
    return dbc.Card(
        className="panel-card h-100",
        children=dbc.CardBody(
            [
                html.Div(kicker, className="card-kicker"),
                html.H4(title, className="card-title"),
                dcc.Graph(
                    id=graph_id,
                    figure=empty_figure(),
                    config={"displaylogo": False},
                    className="tb-graph",
                ),
                controls,
            ]
        ),
    )


def _charts():
    loss_controls = html.Div(
        className="tb-chart-controls",
        children=[
            html.Div(
                [
                    html.Label("Smoothing", className="field-label"),
                    dcc.Slider(
                        id="tb-smoothing",
                        min=0,
                        max=0.99,
                        step=0.01,
                        value=0.6,
                        marks={0: "0", 0.5: "0.5", 0.99: "0.99"},
                        tooltip={"placement": "bottom"},
                        updatemode="drag",
                        allow_direct_input=False,
                        className="tb-slider",
                    ),
                ],
                className="tb-smoothing",
            ),
            dbc.Checklist(
                id="tb-loss-split",
                options=[{"label": "Show Charbonnier / SSIM split", "value": "split"}],
                value=[],
                switch=True,
                className="tb-switch",
            ),
        ],
    )
    return dbc.Row(
        [
            dbc.Col(_chart_card("LOSS", "Training & validation loss", "tb-loss-graph", loss_controls), lg=6),
            dbc.Col(_chart_card("QUALITY", "Validation PSNR vs. bicubic", "tb-psnr-graph"), lg=6),
        ],
        className="g-4 mt-1",
    )


LIVE_TILES = {"sr", "error"}     # these follow the iteration slider; the others are fixed references


def _image_tile(key, label):
    live = key in LIVE_TILES
    return html.Div(
        className="tb-image-tile" + (" tb-live" if live else ""),
        children=[
            html.Div(
                [label, html.Span("● changes with slider" if live else "fixed", className="tb-image-tag")],
                className="tb-image-label",
            ),
            html.Div(
                id=f"tb-img-{key}",
                className="tb-image-box",
                children=html.Div("▧", className="preview-placeholder-icon"),
            ),
        ],
    )


def _sharpening():
    return dbc.Card(
        className="panel-card mt-4",
        children=dbc.CardBody(
            [
                html.Div(
                    className="tb-sharpen-head",
                    children=[
                        html.Div(
                            [
                                html.Div("SHARPENING PROGRESS", className="card-kicker"),
                                html.H4("How the model sees a fixed validation patch", className="card-title"),
                            ]
                        ),
                        dbc.RadioItems(
                            id="tb-sample",
                            options=[{"label": f"Sample {k}", "value": k} for k in SAMPLE_IDS],
                            value=SAMPLE_IDS[0],
                            inline=True,
                            className="tb-sample-picker",
                            inputClassName="btn-check",
                            labelClassName="tb-sample-btn",
                            labelCheckedClassName="active",
                        ),
                    ],
                ),
                html.Div(
                    className="tb-image-row",
                    children=[_image_tile(key, label) for key, label in IMAGE_TILES],
                ),
                html.Div(
                    className="tb-timeline",
                    children=[
                        dbc.Button("▶", id="tb-play", className="tb-play-btn", n_clicks=0),
                        html.Div(
                            dcc.Slider(
                                id="tb-step-slider",
                                min=0,
                                max=0,
                                step=1,          # position in the list of saved iterations
                                value=0,
                                marks={},
                                allow_direct_input=False,
                                className="tb-slider",
                            ),
                            className="tb-timeline-slider",
                        ),
                        html.Div("First samples at iteration 250", id="tb-step-label", className="tb-step-label"),
                    ],
                ),
            ]
        ),
    )


def _empty_state():
    return dbc.Card(
        id="tb-empty",
        className="panel-card tb-empty d-none",
        children=dbc.CardBody(
            [
                html.Div("⌁", className="tb-empty-icon"),
                html.H4("No training runs yet", className="card-title"),
                html.P(
                    "Runs appear here automatically as soon as training starts writing to the runs folder.",
                    className="card-copy",
                ),
                html.Div("Developing the UI without a model? Generate a demo run:", className="tb-empty-hint"),
                html.Code("python ui/dev/fake_run.py", className="tb-empty-code"),
            ]
        ),
    )


def tensorboard_page():
    header_right = html.Div(
        className="tb-header-right",
        children=[
            dcc.Dropdown(
                id="tb-run",
                options=[],
                placeholder="Select run",
                clearable=False,
                className="tb-run-select",
            ),
            status_pill("Waiting for run", "muted", "tb-status"),
        ],
    )
    return html.Div(
        className="tab-content-wrap tb-page",
        children=[
            section_header(
                "MONITORING",
                "TensorBoard",
                "Live loss curves, validation quality and sample outputs for the selected training run.",
                header_right,
            ),
            dcc.Interval(id="tb-interval", interval=REFRESH_MS, n_intervals=0),
            dcc.Interval(id="tb-play-timer", interval=PLAY_MS, n_intervals=0, disabled=True),
            dcc.Store(id="tb-chart-key"),
            dcc.Store(id="tb-steps", data=[]),
            _empty_state(),
            html.Div(
                id="tb-content",
                children=[_status_cards(), _charts(), _sharpening()],
            ),
        ],
    )
