from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import APP_NAME


def section_header(eyebrow, title, subtitle=None, right=None):
    return html.Div(
        className="section-header",
        children=[
            html.Div(
                [
                    html.Div(eyebrow, className="eyebrow"),
                    html.H2(title, className="section-title"),
                    html.P(subtitle, className="section-subtitle") if subtitle else None,
                ]
            ),
            right,
        ],
    )


def upload_box(component_id, title, helper, icon="↥", accept="image/*", multiple=False):
    return dcc.Upload(
        id=component_id,
        accept=accept,
        multiple=multiple,
        className="upload-dropzone",
        children=html.Div(
            [
                html.Div(icon, className="upload-icon"),
                html.Div(title, className="upload-title"),
                html.Div(helper, className="upload-helper"),
                html.Div("Browse files", className="upload-action"),
            ]
        ),
    )


def metric_card(label, value, hint, accent="cyan"):
    return dbc.Card(
        className=f"metric-card metric-{accent}",
        children=dbc.CardBody(
            [
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Div(hint, className="metric-hint"),
            ]
        ),
    )


def status_pill(text="Idle", tone="muted", element_id=None):
    props = {"className": f"status-pill status-{tone}"}
    if element_id is not None:
        props["id"] = element_id
    return html.Span(text, **props)


def topbar():
    return html.Header(
        className="topbar",
        children=[
            html.Div(
                className="brand-wrap",
                children=[
                    html.Div("S", className="brand-mark"),
                    html.Div(
                        [
                            html.Div(APP_NAME, className="brand-title"),
                            html.Div("Super-Resolution Model Workspace", className="brand-subtitle"),
                        ]
                    ),
                ],
            ),
            html.Div(
                className="topbar-actions",
                children=[
                    html.Div(
                        [html.Span(className="live-dot"), html.Span("Workspace ready")],
                        className="workspace-state",
                    ),
                    dbc.Checklist(
                        id="theme-toggle",
                        options=[{"label": "Light mode", "value": "light"}],
                        value=[],
                        switch=True,
                        className="theme-toggle",
                    ),
                    dbc.Button("Documentation", color="light", outline=True, className="ghost-btn"),
                ],
            ),
        ],
    )


def training_controls():
    return dbc.Card(
        className="panel-card controls-card",
        children=dbc.CardBody(
            [
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
                                    placeholder="/runs/experiment_001",
                                    value="/runs/experiment_001",
                                    className="soft-input",
                                ),
                                dbc.Button("Choose", id="choose-output", className="input-button", n_clicks=0),
                            ],
                            className="mb-3",
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


def image_preview_card(title, image_id, empty_copy):
    return dbc.Card(
        className="panel-card preview-card h-100",
        children=dbc.CardBody(
            [
                html.Div(title.upper(), className="card-kicker"),
                html.Div(
                    id=image_id,
                    className="image-preview empty-preview",
                    children=[
                        html.Div("▧", className="preview-placeholder-icon"),
                        html.Div(empty_copy, className="preview-placeholder-copy"),
                    ],
                ),
            ]
        ),
    )
