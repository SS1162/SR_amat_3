from dash import dcc, html
import dash_bootstrap_components as dbc

from ..components import metric_card, section_header, status_pill, upload_box


def training_controls():
    return dbc.Card(
        className="panel-card controls-card",
        children=dbc.CardBody(
            [
                dcc.Interval(id="train-poll-timer", interval=1000, n_intervals=0, disabled=True),
                html.Div("Run configuration", className="card-kicker"),
                html.H4("Training controls", className="card-title"),
                html.P(
                    "Configure the run, choose where artifacts will be saved, and start or stop training.",
                    className="card-copy",
                ),
                html.Div(
                    [
                        html.Label("Output directory", className="field-label"),
                        dbc.InputGroup(
                            [
                                dbc.Input(
                                    id="output-path",
                                    placeholder="runs",
                                    value="runs",
                                    className="soft-input",
                                ),
                                dbc.Button("Select Routing", id="choose-output", className="input-button", n_clicks=0),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            "Click Select Routing to open folders on your computer and choose the directory you want.",
                            className="field-feedback",
                        ),
                        html.Div(id="output-path-feedback", className="field-feedback"),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Run name", className="field-label"),
                        dbc.Input(id="run-name", value="spfi_baseline_01", className="soft-input mb-3"),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Compute device", className="field-label"),
                        dbc.Select(
                            id="device-select",
                            options=[
                                {"label": "Auto", "value": "auto"},
                                {"label": "CUDA / GPU", "value": "cuda"},
                                {"label": "CPU", "value": "cpu"},
                            ],
                            value="auto",
                            className="soft-input mb-4",
                        ),
                    ]
                ),
                html.Div(
                    className="action-stack",
                    children=[
                        dbc.Button("Start training", id="start-training", className="primary-action", n_clicks=0),
                        dbc.Button("Stop", id="stop-training", className="danger-action", n_clicks=0, disabled=True),
                        dbc.Button("Resume", id="resume-training", className="resume-action d-none", n_clicks=0),
                    ],
                ),
                html.Div(
                    className="run-status-block",
                    children=[
                        html.Div(
                            [html.Span("Run status", className="field-label"), status_pill("Idle", "muted", "train-status")],
                            className="status-line",
                        ),
                        dbc.Progress(id="train-progress", value=0, className="train-progress", striped=False),
                        html.Div("No active training run", id="train-status-copy", className="status-copy"),
                    ],
                ),
            ]
        ),
    )


def training_metrics():
    return html.Div(
        className="metrics-grid mt-4",
        children=[
            metric_card("Training loss", "—", "Available when a run begins", "cyan", "metric-loss"),
            metric_card("Validation PSNR", "—", "Available after the first checkpoint", "violet", "metric-val"),
            metric_card("Iteration", "0", "No active training run", "green", "metric-iter"),
            metric_card("Elapsed", "00:00", "Training time, excluding pauses", "amber", "metric-elapsed"),
        ],
    )


def data_card():
    return dbc.Card(
        className="panel-card",
        children=dbc.CardBody(
            [
                html.Div("01 · DATA", className="card-kicker"),
                html.H4("Training inputs", className="card-title"),
                html.P(
                    "Add the training images. Matching masks and an optional small image can be added using the same file names when needed.",
                    className="card-copy",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            upload_box(
                                "train-image-upload",
                                "Training images",
                                "PNG · required",
                                icon="＋",
                                multiple=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            upload_box(
                                "mask-upload",
                                "Optional masks",
                                "Same file names as the images",
                                icon="◌",
                                multiple=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            upload_box(
                                "train-small-image-upload",
                                "Small image",
                                "Optional small image input",
                                icon="▣",
                            ),
                            md=4,
                        ),
                    ],
                    className="g-3",
                ),
                html.Div(id="train-upload-summary", className="upload-summary"),
            ]
        ),
    )


def base_model_card():
    """Optional checkpoint: empty trains from scratch, a checkpoint makes the run a fine-tune."""
    return dbc.Card(
        className="panel-card mt-4",
        children=dbc.CardBody(
            [
                html.Div("02 · FINE-TUNING · OPTIONAL", className="card-kicker"),
                html.H4("Fine-tuning: continue from an existing model", className="card-title"),
                html.P(
                    "To fine-tune, choose a trained checkpoint and it will keep training on the images above; "
                    "the original checkpoint is not changed. Leave empty to train a new model from scratch.",
                    className="card-copy",
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="base-model",
                            placeholder="runs/spfi_baseline_01/checkpoints/spfi_x4_best.pt",
                            className="soft-input",
                        ),
                        dbc.Button("Choose", id="choose-base-model", className="input-button", n_clicks=0),
                    ],
                ),
                html.Div("Training from scratch", id="base-model-hint", className="field-feedback"),
            ]
        ),
    )


def pipeline_card():
    return dbc.Card(
        className="panel-card mt-4",
        children=dbc.CardBody(
            [
                html.Div("03 · PIPELINE", className="card-kicker"),
                html.H4("Data preparation preview", className="card-title"),
                html.P(
                    "The computation stays in the backend, but the UI communicates the flow clearly to the user.",
                    className="card-copy",
                ),
                html.Div(
                    className="pipeline",
                    children=[
                        html.Div([html.Div("HR", className="pipeline-node-icon"), html.Div("Source image")], className="pipeline-node active"),
                        html.Div("→", className="pipeline-arrow"),
                        html.Div([html.Div("LR", className="pipeline-node-icon"), html.Div("Downsample")], className="pipeline-node"),
                        html.Div("→", className="pipeline-arrow"),
                        html.Div([html.Div("SPFI", className="pipeline-node-icon wide"), html.Div("Model")], className="pipeline-node"),
                        html.Div("→", className="pipeline-arrow"),
                        html.Div([html.Div("SR", className="pipeline-node-icon"), html.Div("Prediction")], className="pipeline-node"),
                        html.Div("↔", className="pipeline-arrow"),
                        html.Div([html.Div("GT", className="pipeline-node-icon"), html.Div("Compare")], className="pipeline-node"),
                    ],
                ),
            ]
        ),
    )


def training_page():
    return html.Div(
        className="tab-content-wrap",
        children=[
            section_header(
                "TRAINING",
                "Train a super-resolution model",
                "Upload training images and start a run. Add a base model to fine-tune it instead of starting from scratch.",
                status_pill("From scratch", "success", "train-mode-pill"),
            ),
            dbc.Row(
                [
                    dbc.Col([data_card(), base_model_card(), pipeline_card()], lg=8),
                    dbc.Col(training_controls(), lg=4),
                ],
                className="g-4",
            ),
            training_metrics(),
        ],
    )
