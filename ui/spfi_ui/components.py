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


def metric_card(label, value, hint, accent="cyan", element_id=None):
    """With ``element_id``, the value and hint get ids ``<element_id>-value`` / ``<element_id>-hint``."""
    value_props = {"id": f"{element_id}-value"} if element_id else {}
    hint_props = {"id": f"{element_id}-hint"} if element_id else {}
    return dbc.Card(
        className=f"metric-card metric-{accent}",
        children=dbc.CardBody(
            [
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value", **value_props),
                html.Div(hint, className="metric-hint", **hint_props),
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


def upload_chip(contents, filename):
    if not contents:
        return None
    return html.Div(
        [html.Span("✓", className="file-check"), html.Span(filename or "Uploaded file", className="file-name")],
        className="file-chip",
    )


def summarize_uploads(*uploads):
    """uploads: (contents, filenames) pairs from multi-file dcc.Upload components."""
    items = []
    for contents, names in uploads:
        if not contents:
            continue
        if isinstance(contents, str):
            contents, names = [contents], [names]
        names = names or [None] * len(contents)
        items.extend(upload_chip(content, name) for content, name in zip(contents, names))
    return items or "No files selected yet."
