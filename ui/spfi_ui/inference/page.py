from dash import html
import dash_bootstrap_components as dbc

from ..components import image_preview_card, section_header, status_pill, upload_box


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
