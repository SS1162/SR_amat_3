from dash import dcc, html
import dash_bootstrap_components as dbc

from ..components import image_preview_card, section_header, status_pill, upload_box


def inference_page():
    return html.Div(
        className="tab-content-wrap",
        children=[
            # Final state of the last Run / Continue: {job_id, run_dir, status, next_index, processed, total, error}
            dcc.Store(id="inference-job"),
            # Job id of the running job, published by its first progress update so Stop can reach it
            dcc.Store(id="inference-active-job"),
            dcc.Download(id="inference-download-file"),
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
                                    lg=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Destination directory", className="field-label"),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="inference-output-path",
                                                    placeholder="outputs/inference",
                                                    value="outputs/inference",
                                                    className="soft-input",
                                                ),
                                                dbc.Button("Select Routing", id="choose-inference-output", className="input-button"),
                                            ]
                                        ),
                                        html.Div(
                                            "Click Select Routing to open folders on your computer and choose the directory you want.",
                                            className="field-feedback",
                                        ),
                                        html.Div(id="inference-output-path-feedback", className="field-feedback"),
                                    ],
                                    lg=4,
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
                                    lg=2,
                                ),
                                dbc.Col(
                                    dbc.Button("Run inference", id="run-inference", className="primary-action inference-btn", n_clicks=0),
                                    lg=2,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="g-4",
                        ),
                        html.Div(
                            className="run-status-block",
                            children=[
                                html.Div(
                                    [
                                        dbc.Button("Stop", id="stop-inference", className="danger-action", n_clicks=0, disabled=True),
                                        dbc.Button("Continue", id="continue-inference", className="resume-action", n_clicks=0, disabled=True),
                                        dbc.Button(
                                            "Download results",
                                            id="download-inference",
                                            className="input-button",
                                            n_clicks=0,
                                            disabled=True,
                                        ),
                                    ],
                                    className="inference-actions",
                                ),
                                dbc.Progress(id="inference-progress", value=0, label="", className="inference-progress"),
                                html.Div("No inference run yet", id="inference-progress-copy", className="status-copy"),
                            ],
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
