from dash import dcc, html
import dash_bootstrap_components as dbc

from ..components import metric_card, section_header, status_pill, upload_box


def training_controls(prefix="", run_name="spfi_baseline_01", title="Training controls"):
    """Run controls; ``prefix`` keeps ids unique when the panel appears on more than one tab."""
    return dbc.Card(
        className="panel-card controls-card",
        children=dbc.CardBody(
            [
                dcc.Interval(id=f"{prefix}train-poll-timer", interval=1000, n_intervals=0, disabled=True),
                html.Div("Run configuration", className="card-kicker"),
                html.H4(title, className="card-title"),
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
                                    id=f"{prefix}output-path",
                                    placeholder="runs",
                                    value="runs",
                                    className="soft-input",
                                ),
                                dbc.Button("Choose", id=f"{prefix}choose-output", className="input-button", n_clicks=0),
                            ],
                            className="mb-3",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Run name", className="field-label"),
                        dbc.Input(id=f"{prefix}run-name", value=run_name, className="soft-input mb-3"),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Compute device", className="field-label"),
                        dbc.Select(
                            id=f"{prefix}device-select",
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
                        dbc.Button("Start training", id=f"{prefix}start-training", className="primary-action", n_clicks=0),
                        dbc.Button("Stop", id=f"{prefix}stop-training", className="danger-action", n_clicks=0, disabled=True),
                        dbc.Button("Resume", id=f"{prefix}resume-training", className="resume-action d-none", n_clicks=0),
                    ],
                ),
                html.Div(
                    className="run-status-block",
                    children=[
                        html.Div(
                            [html.Span("Run status", className="field-label"), status_pill("Idle", "muted", f"{prefix}train-status")],
                            className="status-line",
                        ),
                        dbc.Progress(id=f"{prefix}train-progress", value=0, className="train-progress", striped=False),
                        html.Div("No active training run", id=f"{prefix}train-status-copy", className="status-copy"),
                    ],
                ),
            ]
        ),
    )


def scratch_tab():
    return html.Div(
        className="tab-content-wrap",
        children=[
            section_header(
                "TRAINING / FROM SCRATCH",
                "Build a new super-resolution model",
                "Upload training inputs, prepare the run, and monitor model progress from one workspace.",
                status_pill("Ready", "success"),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                className="panel-card",
                                children=dbc.CardBody(
                                    [
                                        html.Div("01 · DATA", className="card-kicker"),
                                        html.H4("Training inputs", className="card-title"),
                                        html.P(
                                            "Add the training images. Matching masks can be added optionally, using the same file names as the images.",
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
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    upload_box(
                                                        "mask-upload",
                                                        "Optional masks",
                                                        "Same file names as the images",
                                                        icon="◌",
                                                        multiple=True,
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3",
                                        ),
                                        html.Div(id="train-upload-summary", className="upload-summary"),
                                    ]
                                ),
                            ),
                            dbc.Card(
                                className="panel-card mt-4",
                                children=dbc.CardBody(
                                    [
                                        html.Div("02 · PIPELINE", className="card-kicker"),
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
                            ),
                        ],
                        lg=8,
                    ),
                    dbc.Col(training_controls(), lg=4),
                ],
                className="g-4",
            ),
            training_metrics(),
        ],
    )


def training_metrics(prefix=""):
    return html.Div(
        className="metrics-grid mt-4",
        children=[
            metric_card("Training loss", "—", "Available when a run begins", "cyan", f"{prefix}metric-loss"),
            metric_card("Validation PSNR", "—", "Available after the first checkpoint", "violet", f"{prefix}metric-val"),
            metric_card("Iteration", "0", "No active training run", "green", f"{prefix}metric-iter"),
            metric_card("Elapsed", "00:00", "Training time, excluding pauses", "amber", f"{prefix}metric-elapsed"),
        ],
    )


def finetune_tab():
    return html.Div(
        className="tab-content-wrap",
        children=[
            section_header(
                "TRAINING / FINE-TUNING",
                "Continue from an existing model",
                "Load a trained checkpoint and keep training it on new images. The original checkpoint is not changed.",
                status_pill("Ready", "success"),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            className="panel-card",
                            children=dbc.CardBody(
                                [
                                    html.Div("BASE MODEL", className="card-kicker"),
                                    html.H4("Select a checkpoint", className="card-title"),
                                    html.P("Choose the model that will be used as the starting point for fine-tuning.", className="card-copy"),
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="ft-weights",
                                                placeholder="runs/spfi_baseline_01/checkpoints/spfi_x4_best.pt",
                                                className="soft-input",
                                            ),
                                            dbc.Button("Choose", className="input-button"),
                                        ],
                                        className="mb-4",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                upload_box(
                                                    "finetune-data-upload",
                                                    "Fine-tuning images",
                                                    "Dataset / images for adaptation",
                                                    icon="＋",
                                                    multiple=True,
                                                ),
                                                md=6,
                                            ),
                                            dbc.Col(
                                                upload_box(
                                                    "finetune-mask-upload",
                                                    "Optional masks",
                                                    "Matching masks for the fine-tuning images",
                                                    icon="◌",
                                                    multiple=True,
                                                ),
                                                md=6,
                                            ),
                                        ],
                                        className="g-3",
                                    ),
                                    html.Div(id="finetune-upload-summary", className="upload-summary"),
                                ]
                            ),
                        ),
                        lg=8,
                    ),
                    dbc.Col(training_controls("ft-", "spfi_finetune_01", "Fine-tuning controls"), lg=4),
                ],
                className="g-4",
            ),
            training_metrics("ft-"),
        ],
    )


def training_page():
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(scratch_tab(), label="From Scratch", tab_id="scratch"),
                    dbc.Tab(finetune_tab(), label="Fine-Tuning", tab_id="finetune"),
                ],
                active_tab="scratch",
                className="subtabs",
            )
        ]
    )

