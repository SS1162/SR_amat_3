"""Page layout: setup sidebar, summary, tabs (images / charts), detail modal."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from .config import APP_NAME, CROP_OPTIONS, DEFAULT_CROP, DEFAULT_DATA_DIR, DEFAULT_N_IMAGES, DEFAULT_SEED

SORT_OPTIONS = [
    {"label": "ΔPSNR vs bicubic", "value": "d_psnr"},
    {"label": "ΔPSNR vs base", "value": "d_base"},
    {"label": "PSNR", "value": "psnr"},
    {"label": "SSIM", "value": "ssim"},
    {"label": "ΔSSIM vs bicubic", "value": "d_ssim"},
    {"label": "Bicubic PSNR", "value": "bicubic_psnr"},
    {"label": "Blur σ", "value": "sigma"},
    {"label": "File name", "value": "stem"},
]

FILTER_OPTIONS = [
    {"label": "Worse than bicubic", "value": "worse_bicubic"},
    {"label": "Worse than base", "value": "worse_base"},
]

GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# (graph id, title, what to read from it)
CHARTS = [
    ("chart-hist", "PSNR distribution", "Model, bicubic and base on the same images"),
    ("chart-delta", "ΔPSNR per image", "Right of 0 = the model wins"),
    ("chart-scatter", "Model vs bicubic", "Below the dotted line = worse than bicubic"),
    ("chart-sigma", "Gain vs blur strength", "ΔPSNR against the σ of the degradation"),
]


def _field(label: str, control, className: str = "") -> html.Div:
    return html.Div([html.Label(label, className="ev-label"), control], className=f"ev-field {className}")


def _section(title: str, *children) -> html.Section:
    return html.Section([html.H2(title, className="ev-section-title"), *children], className="ev-section")


def _sidebar() -> html.Aside:
    return html.Aside([
        _section(
            "1 · Models",
            _field("Model", dcc.Dropdown(id="model", clearable=False, className="ev-dd")),
            _field("Compare with (base)", dcc.Dropdown(id="base", clearable=True, className="ev-dd",
                                                       placeholder="None")),
            dbc.Button("↻ Rescan checkpoints", id="refresh-models", className="ev-btn-ghost ev-btn-sm"),
        ),
        _section(
            "2 · Sample",
            _field("Dataset folder", dbc.Input(id="data-dir", value=DEFAULT_DATA_DIR, debounce=True)),
            html.Div([
                _field("Images", dbc.Input(id="n-images", type="number", min=1, step=1,
                                           value=DEFAULT_N_IMAGES, debounce=True)),
                _field("Seed", dbc.Input(id="seed", type="number", step=1, value=DEFAULT_SEED, debounce=True)),
            ], className="ev-pair"),
            html.Div([
                _field("HR crop", dcc.Dropdown(id="crop", options=CROP_OPTIONS, value=DEFAULT_CROP,
                                               clearable=False, className="ev-dd")),
                _field("Subset", dcc.Dropdown(id="subset", options=[{"label": "All", "value": "all"},
                                                                    {"label": "test.csv", "value": "test"}],
                                              value="all", clearable=False, className="ev-dd")),
            ], className="ev-pair"),
        ),
        _section(
            "3 · Run",
            html.Div([
                dbc.Button("Evaluate", id="evaluate", className="ev-btn-primary"),
                dbc.Button("Stop", id="stop", className="ev-btn-ghost"),
            ], className="ev-pair ev-actions"),
            dbc.Progress(id="progress", value=0, className="ev-progress"),
            html.Div(id="job-message", className="ev-muted ev-job-message"),
            html.Div("Only missing results are computed; everything else comes from the cache.",
                     className="ev-muted ev-note"),
        ),
    ], className="ev-sidebar")


def _grid_toolbar() -> html.Div:
    return html.Div([
        _field("Sort by", dcc.Dropdown(id="sort-by", options=SORT_OPTIONS, value="d_psnr",
                                       clearable=False, className="ev-dd"), "ev-w-sort"),
        _field("Order", dbc.RadioItems(id="sort-order", inline=True, value="asc",
                                       options=[{"label": "Worst first", "value": "asc"},
                                                {"label": "Best first", "value": "desc"}])),
        _field("Show only", dbc.Checklist(id="filters", options=FILTER_OPTIONS, value=[], inline=True)),
        _field("Blur σ", html.Div(dcc.RangeSlider(id="sigma-range", min=0, max=4, step=0.1, value=[0, 4],
                                                  marks={i: str(i) for i in range(5)},
                                                  tooltip={"placement": "bottom"}),
                                  className="ev-slider"), "ev-w-slider"),
        html.Div([
            html.Span(id="grid-count", className="ev-count"),
            dbc.Button("Export CSV", id="export", className="ev-btn-ghost ev-btn-sm"),
            dcc.Download(id="download"),
        ], className="ev-toolbar-end"),
    ], className="ev-toolbar")


def _chart(graph_id: str, title: str, hint: str) -> html.Div:
    return html.Div([
        html.Div([html.H3(title, className="ev-chart-title"), html.Span(hint, className="ev-muted")],
                 className="ev-chart-head"),
        dcc.Graph(id=graph_id, config=GRAPH_CONFIG),
    ], className="ev-chart")


def _tabs() -> dbc.Tabs:
    return dbc.Tabs([
        dbc.Tab(html.Div([_grid_toolbar(), html.Div(id="grid", className="ev-grid")], className="ev-tab-body"),
                label="Images", tab_id="images"),
        dbc.Tab(html.Div([
            html.Div("Click a point in the scatter plots to open that image.", className="ev-muted"),
            html.Div([_chart(*c) for c in CHARTS], className="ev-charts"),
        ], className="ev-tab-body"), label="Charts", tab_id="charts"),
    ], active_tab="images", className="ev-tabs")


def _detail_modal() -> dbc.Modal:
    return dbc.Modal([
        dbc.ModalHeader([
            html.Div(id="detail-title", className="ev-detail-title"),
            html.Div([
                dbc.Button("‹ Prev", id="detail-prev", className="ev-btn-ghost ev-btn-sm"),
                dbc.Button("Next ›", id="detail-next", className="ev-btn-ghost ev-btn-sm"),
            ], className="ev-detail-nav"),
        ], close_button=True),
        dbc.ModalBody([
            html.Div(id="detail-metrics", className="ev-detail-metrics"),
            html.Div("Drag to zoom any panel - all panels follow. Double-click to reset.",
                     className="ev-muted"),
            dcc.Loading(dcc.Graph(id="detail-graph", config=GRAPH_CONFIG), color="#54d7ff"),
        ]),
    ], id="detail", size="xl", is_open=False, className="ev-modal", scrollable=True)


def build_layout() -> html.Div:
    return html.Div([
        dcc.Store(id="data-version", data=0),
        dcc.Store(id="grid-order", data=[]),
        dcc.Store(id="detail-stem"),
        dcc.Interval(id="job-poll", interval=1000, disabled=False),
        html.Header([
            html.Div(APP_NAME, className="ev-brand"),
            html.Div("Visual check of SR checkpoints on a fixed random sample · inference only",
                     className="ev-muted"),
        ], className="ev-header"),
        html.Div([
            _sidebar(),
            html.Main([
                html.Div(id="status", className="ev-status"),
                html.Div(id="stats", className="ev-stats"),
                _tabs(),
            ], className="ev-main"),
        ], className="ev-body"),
        _detail_modal(),
    ], className="ev-shell")
