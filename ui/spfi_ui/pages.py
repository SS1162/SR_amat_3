from dash import html
import dash_bootstrap_components as dbc

from .components import image_preview_card, metric_card, section_header, status_pill, training_controls, upload_box


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
                                            "Add the source image. A matching mask can be provided when the training sample requires one.",
                                            className="card-copy",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    upload_box(
                                                        "train-image-upload",
                                                        "Source image",
                                                        "PNG, JPG, TIFF · required",
                                                        icon="＋",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    upload_box(
                                                        "mask-upload",
                                                        "Optional mask",
                                                        "Matching mask for the source image",
                                                        icon="◌",
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
            html.Div(
                className="metrics-grid mt-4",
                children=[
                    metric_card("Training loss", "—", "Available when a run begins", "cyan"),
                    metric_card("Validation metric", "—", "Connect to your backend metric", "violet"),
                    metric_card("Epoch", "0", "Current / total epochs", "green"),
                    metric_card("Elapsed", "00:00", "Runtime of active experiment", "amber"),
                ],
            ),
        ],
    )


def finetune_tab():
    return html.Div(
        className="tab-content-wrap",
        children=[
            section_header(
                "TRAINING / FINE-TUNING",
                "Continue from an existing model",
                "The interface is already prepared for the future fine-tuning flow, while keeping it visually separate from training from scratch.",
                status_pill("Planned", "warning"),
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
                                            dbc.Input(placeholder="/models/spfi_checkpoint.pt", className="soft-input"),
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
                    dbc.Col(
                        dbc.Card(
                            className="panel-card future-card",
                            children=dbc.CardBody(
                                [
                                    html.Div("COMING NEXT", className="card-kicker"),
                                    html.H4("Fine-tuning engine", className="card-title"),
                                    html.P(
                                        "This area is intentionally present in the product now, so adding the backend later does not require redesigning the training experience.",
                                        className="card-copy",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Checkpoint loading", className="feature-chip"),
                                            html.Span("Freeze strategy", className="feature-chip"),
                                            html.Span("Learning-rate override", className="feature-chip"),
                                            html.Span("Resume tracking", className="feature-chip"),
                                        ],
                                        className="feature-chip-wrap",
                                    ),
                                    dbc.Button("Fine-tuning not enabled yet", disabled=True, className="disabled-action mt-4"),
                                ]
                            ),
                        ),
                        lg=4,
                    ),
                ],
                className="g-4",
            ),
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


def inference_page():
    return html.Div(
        className="tab-content-wrap",
        children=[
            section_header(
                "INFERENCE",
                "Upscale new images with a trained model",
                "Choose a checkpoint, provide one image or a batch, and inspect the high-resolution output in the same workspace.",
                status_pill("Model not loaded", "muted", "inference-status"),
            ),
            dbc.Card(
                className="panel-card mb-4",
                children=dbc.CardBody(
                    [
                        html.Div("MODEL & INPUT", className="card-kicker"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Trained model", className="field-label"),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="model-path", placeholder="/models/spfi_model.pt", className="soft-input"),
                                                dbc.Button("Choose model", id="choose-model", className="input-button"),
                                            ]
                                        ),
                                    ],
                                    lg=5,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Input mode", className="field-label"),
                                        dbc.RadioItems(
                                            id="input-mode",
                                            options=[
                                                {"label": "Single image", "value": "single"},
                                                {"label": "Batch / folder", "value": "batch"},
                                            ],
                                            value="single",
                                            inline=True,
                                            className="mode-radio",
                                        ),
                                    ],
                                    lg=4,
                                ),
                                dbc.Col(
                                    dbc.Button("Run inference", id="run-inference", className="primary-action inference-btn", n_clicks=0),
                                    lg=3,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="g-4",
                        ),
                    ]
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            className="panel-card h-100",
                            children=dbc.CardBody(
                                [
                                    html.Div("INPUT", className="card-kicker"),
                                    upload_box(
                                        "inference-upload",
                                        "Image or image batch",
                                        "Drop a file here or select from your computer",
                                        icon="＋",
                                        multiple=True,
                                    ),
                                    html.Div(id="inference-file-summary", className="upload-summary"),
                                ]
                            ),
                        ),
                        lg=4,
                    ),
                    dbc.Col(image_preview_card("Low-resolution input", "lr-preview", "Input preview will appear here"), lg=4),
                    dbc.Col(image_preview_card("High-resolution result", "hr-preview", "Model output will appear here"), lg=4),
                ],
                className="g-4",
            ),
        ],
    )
