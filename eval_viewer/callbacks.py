"""Dash callbacks."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from dash import ALL, Input, Output, State, ctx, html, no_update

from . import jobs
from .charts import (delta_hist, detail_figure, empty_figure, psnr_hist, psnr_scatter,
                     sigma_scatter)
from .config import BASE_MODEL_PATH
from .data import TIE_DB, image_url, load_rows, load_views, model_meta, summarize
from .evaluate import SampleSpec, is_evaluated, list_models, model_label

SPEC_STATES = [State("data-dir", "value"), State("n-images", "value"), State("seed", "value"),
               State("crop", "value"), State("subset", "value"), State("model", "value"),
               State("base", "value")]
SPEC_INPUTS = [Input(s.component_id, s.component_property) for s in SPEC_STATES]

EXPORT_FIELDS = ["stem", "sigma", "psnr", "ssim", "bicubic_psnr", "bicubic_ssim", "d_psnr", "d_ssim",
                 "base_psnr", "base_ssim", "d_base", "seconds", "orig_w", "orig_h", "crop_w", "crop_h"]


def _context(data_dir, n, seed, crop, subset, model, base):
    """UI values -> (spec, model_path, base_path). Raises ValueError with a message for the user."""
    if not data_dir or not Path(data_dir).is_dir():
        raise ValueError(f"Dataset folder not found: {data_dir or '(empty)'}")
    if not model:
        raise ValueError("Pick a model")
    if not n or int(n) < 1:
        raise ValueError("Number of images must be at least 1")
    spec = SampleSpec(Path(data_dir), int(n), int(seed or 0), str(crop), subset)
    base_path = Path(base) if base and base != model else None
    return spec, Path(model), base_path


def _fmt(x: float | None, digits: int = 2, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _delta_badge(value: float | None, label: str) -> html.Span:
    if value is None:
        return html.Span()
    if value > TIE_DB:
        cls, icon = "good", "▲"
    elif value < -TIE_DB:
        cls, icon = "bad", "▼"
    else:
        cls, icon = "tie", "="
    return html.Span(f"{icon} {value:+.2f} dB {label}", className=f"ev-badge ev-{cls}")


def _tile(title: str, value: str, sub: str = "", tone: str = "") -> html.Div:
    return html.Div([html.Div(title, className="ev-tile-title"),
                     html.Div(value, className=f"ev-tile-value {tone}"),
                     html.Div(sub, className="ev-tile-sub")], className="ev-tile")


def _tone(delta: float) -> str:
    return "ev-good" if delta > TIE_DB else "ev-bad" if delta < -TIE_DB else ""


def _stats(s: dict) -> list:
    tiles = [
        _tile("Model PSNR", f"{s['psnr_mean']:.2f} dB",
              f"median {s['psnr_median']:.2f} · std {s['psnr_std']:.2f} · n={s['n']}"),
        _tile("Model SSIM", f"{s['ssim_mean']:.4f}", f"bicubic {s['bicubic_ssim_mean']:.4f}"),
        _tile("Bicubic PSNR", f"{s['bicubic_psnr_mean']:.2f} dB", "same LR input"),
        _tile("Gain vs bicubic", f"{s['d_psnr_mean']:+.2f} dB",
              f"▲ better on {s['better_pct']:.0f}% · ▼ worse on {s['worse_pct']:.0f}%", _tone(s["d_psnr_mean"])),
    ]
    if "d_base_mean" in s:
        tiles.append(_tile("Gain vs base", f"{s['d_base_mean']:+.2f} dB",
                           f"base {s['base_psnr_mean']:.2f} dB · ▲ better on {s['better_base_pct']:.0f}%",
                           _tone(s["d_base_mean"])))
    tiles.append(_tile("Time / image", f"{s['seconds_mean']:.2f} s", "model forward only"))
    return tiles


def _metric_row(name: str, color_cls: str, psnr: float | None, ssim: float | None) -> html.Tr:
    return html.Tr([html.Td([html.Span(className=f"ev-dot {color_cls}"), name]),
                    html.Td(_fmt(psnr)), html.Td(_fmt(ssim, 4))])


def _cell(row: dict, spec: SampleSpec, model_path: Path) -> html.Div:
    stem = row["stem"]
    d = row["d_psnr"]
    edge = "good" if d > TIE_DB else "bad" if d < -TIE_DB else "tie"
    table_rows = [_metric_row("Model", "ev-c-model", row["psnr"], row["ssim"]),
                  _metric_row("Bicubic", "ev-c-bicubic", row["bicubic_psnr"], row["bicubic_ssim"])]
    if "base_psnr" in row:
        table_rows.append(_metric_row("Base", "ev-c-base", row["base_psnr"], row["base_ssim"]))
    return html.Div([
        html.Div([
            html.Span(stem, className="ev-cell-name"),
            html.Span(f"σ {row['sigma']:.2f}", className="ev-muted"),
        ], className="ev-cell-head"),
        html.Div([
            html.Figure([html.Img(src=image_url(spec, "hr", stem)),
                         html.Figcaption("HR original")]),
            html.Figure([html.Img(src=image_url(spec, "sr", stem, model_path)),
                         html.Figcaption("Model output")]),
        ], className="ev-cell-images"),
        html.Div([
            html.Figure([html.Img(src=image_url(spec, "bicubic", stem)),
                         html.Figcaption("bicubic")], className="ev-cell-thumb"),
            html.Div([
                html.Table([html.Thead(html.Tr([html.Th(""), html.Th("PSNR"), html.Th("SSIM")])),
                            html.Tbody(table_rows)], className="ev-cell-table"),
                html.Div([_delta_badge(d, "vs bicubic"), _delta_badge(row.get("d_base"), "vs base")],
                         className="ev-cell-badges"),
            ]),
        ], className="ev-cell-foot"),
    ], id={"type": "cell", "stem": stem}, n_clicks=0, className=f"ev-cell ev-edge-{edge}",
        title="Open details")


def _apply_view(rows: list[dict], sort_by: str, order: str, filters: list[str], sigma_range) -> list[dict]:
    lo, hi = sigma_range or (0, 99)
    out = [r for r in rows if lo <= r["sigma"] <= hi]
    if "worse_bicubic" in filters:
        out = [r for r in out if r["d_psnr"] < -TIE_DB]
    if "worse_base" in filters:
        out = [r for r in out if r.get("d_base", 0) < -TIE_DB]
    if sort_by == "stem":
        return sorted(out, key=lambda r: r["stem"], reverse=order == "desc")
    has = [r for r in out if sort_by in r]
    missing = [r for r in out if sort_by not in r]
    return sorted(has, key=lambda r: r[sort_by], reverse=order == "desc") + missing


def register_callbacks(app) -> None:

    @app.callback(
        Output("model", "options"), Output("model", "value"),
        Output("base", "options"), Output("base", "value"),
        Input("refresh-models", "n_clicks"),
        State("model", "value"), State("base", "value"),
    )
    def refresh_models(_, model, base):
        paths = list_models()
        base_str = str(BASE_MODEL_PATH) if BASE_MODEL_PATH.is_file() else None
        # File name first: the folder part is long and the dropdown cuts the end off.
        options = [{"label": f"{p.name}  ·  {Path(model_label(p)).parent}"
                             + ("  ·  base" if str(p) == base_str else ""), "value": str(p)}
                   for p in paths]
        values = {o["value"] for o in options}
        if model not in values:
            others = [p for p in paths if str(p) != base_str]
            newest = max(others, key=lambda p: p.stat().st_mtime) if others else None
            model = str(newest) if newest else base_str
        if ctx.triggered_id is None or (base is not None and base not in values):
            base = base_str if base_str != model else None
        return options, model, options, base

    @app.callback(
        Output("job-poll", "disabled", allow_duplicate=True),
        Output("job-message", "children", allow_duplicate=True),
        Input("evaluate", "n_clicks"), Input("stop", "n_clicks"),
        *SPEC_STATES,
        prevent_initial_call=True,
    )
    def start_or_stop(_, __, *values):
        if ctx.triggered_id == "stop":
            jobs.stop()
            return no_update, "Stopping after the current image…"
        try:
            spec, model_path, base_path = _context(*values)
        except ValueError as exc:
            return no_update, str(exc)
        if not jobs.start(spec, [model_path] + ([base_path] if base_path else [])):
            return False, "A job is already running"
        return False, "Starting…"

    @app.callback(
        Output("progress", "value"), Output("progress", "label"),
        Output("job-message", "children"), Output("job-poll", "disabled"),
        Output("data-version", "data"),
        Input("job-poll", "n_intervals"),
        State("data-version", "data"),
    )
    def poll(_, version):
        st = jobs.state()
        pct = round(100 * st["done"] / st["total"]) if st["total"] else (100 if st["status"] == "done" else 0)
        label = f"{st['done']}/{st['total']}" if st["total"] else ""
        running = jobs.is_running()
        new_version = st["finished"] if st["finished"] != version else no_update
        return pct, label, st["message"], not running, new_version

    @app.callback(
        Output("stats", "children"), Output("status", "children"),
        Output("chart-hist", "figure"), Output("chart-delta", "figure"),
        Output("chart-scatter", "figure"), Output("chart-sigma", "figure"),
        Output("grid", "children"), Output("grid-order", "data"), Output("grid-count", "children"),
        *SPEC_INPUTS,
        Input("data-version", "data"),
        Input("sort-by", "value"), Input("sort-order", "value"),
        Input("filters", "value"), Input("sigma-range", "value"),
    )
    def render(data_dir, n, seed, crop, subset, model, base, _version, sort_by, order, filters, sigma_range):
        empty = empty_figure("No results yet")
        try:
            spec, model_path, base_path = _context(data_dir, n, seed, crop, subset, model, base)
        except ValueError as exc:
            return [], html.Span(str(exc), className="ev-bad"), empty, empty, empty, empty, [], [], ""
        rows = load_rows(spec, model_path, base_path)
        if rows is None:
            msg = ("This model has no results for this sample yet. Press Evaluate "
                   "(CPU: about 2 s per image at crop 256).")
            return [], msg, empty, empty, empty, empty, [], [], ""

        meta = model_meta(spec, model_path)
        hp = meta.get("hparams") or {}
        label = meta.get("label", model_label(model_path))
        chips = [("iteration", meta.get("iteration", "?")), ("embed_dim", hp.get("embed_dim", "?")),
                 ("scale", f"x{hp.get('scale', '?')}"), ("sample", spec.key)]
        status = [html.Div([
            html.Span(Path(label).name, className="ev-status-model", title=label),
            *[html.Span([html.Span(k, className="ev-chip-key"), str(v)], className="ev-chip") for k, v in chips],
        ], className="ev-status-row")]
        if base_path is not None and not is_evaluated(spec, base_path):
            status.append(html.Div("The base model has no results for this sample: press Evaluate "
                                   "to add the comparison.", className="ev-warn"))

        view = _apply_view(rows, sort_by, order, filters or [], sigma_range)
        cells = [_cell(r, spec, model_path) for r in view]
        return (_stats(summarize(rows)), status,
                psnr_hist(rows), delta_hist(rows), psnr_scatter(rows), sigma_scatter(rows),
                cells, [r["stem"] for r in view], f"{len(view)} of {len(rows)} images")

    @app.callback(
        Output("detail", "is_open"), Output("detail-stem", "data"),
        Output("chart-scatter", "clickData"), Output("chart-sigma", "clickData"),
        Input({"type": "cell", "stem": ALL}, "n_clicks"),
        Input("chart-scatter", "clickData"), Input("chart-sigma", "clickData"),
        Input("detail-prev", "n_clicks"), Input("detail-next", "n_clicks"),
        State("grid-order", "data"), State("detail-stem", "data"),
        prevent_initial_call=True,
    )
    def open_detail(_cells, scatter_click, sigma_click, _prev, _next, order, current):
        trig = ctx.triggered_id
        if isinstance(trig, dict):
            if not ctx.triggered[0]["value"]:      # cells were just re-rendered, not clicked
                return no_update, no_update, no_update, no_update
            return True, trig["stem"], no_update, no_update
        if trig in ("chart-scatter", "chart-sigma"):
            click = scatter_click if trig == "chart-scatter" else sigma_click
            if not click:
                return no_update, no_update, no_update, no_update
            return True, click["points"][0]["customdata"][0], None, None
        if trig in ("detail-prev", "detail-next") and order and current in order:
            step = -1 if trig == "detail-prev" else 1
            return True, order[(order.index(current) + step) % len(order)], no_update, no_update
        return no_update, no_update, no_update, no_update

    @app.callback(
        Output("detail-title", "children"), Output("detail-metrics", "children"),
        Output("detail-graph", "figure"),
        Input("detail-stem", "data"),
        *SPEC_STATES,
        State("grid-order", "data"),
        prevent_initial_call=True,
    )
    def show_detail(stem, data_dir, n, seed, crop, subset, model, base, order):
        if not stem:
            return no_update, no_update, no_update
        spec, model_path, base_path = _context(data_dir, n, seed, crop, subset, model, base)
        rows = {r["stem"]: r for r in load_rows(spec, model_path, base_path) or []}
        row = rows.get(stem)
        if row is None:
            return stem, "No results for this image", empty_figure("")
        views = load_views(spec, stem, model_path, base_path)
        titles = {"lr": "LR input (nearest ×4)", "bicubic": "Bicubic", "sr": "Model", "hr": "HR original",
                  "base": "Base"}
        fig = detail_figure(views, titles)
        fig.update_layout(uirevision=stem)
        pos = f"  ·  {order.index(stem) + 1} / {len(order)} in grid" if stem in (order or []) else ""
        title = [html.Span(stem, className="ev-cell-name"),
                 html.Span(f"  σ {row['sigma']:.2f} · {int(row['crop_w'])}×{int(row['crop_h'])} crop of "
                           f"{int(row['orig_w'])}×{int(row['orig_h'])}{pos}", className="ev-muted")]
        table = [_metric_row("Model", "ev-c-model", row["psnr"], row["ssim"]),
                 _metric_row("Bicubic", "ev-c-bicubic", row["bicubic_psnr"], row["bicubic_ssim"])]
        if "base_psnr" in row:
            table.append(_metric_row("Base", "ev-c-base", row["base_psnr"], row["base_ssim"]))
        metrics = html.Div([
            html.Table([html.Thead(html.Tr([html.Th(""), html.Th("PSNR"), html.Th("SSIM")])),
                        html.Tbody(table)], className="ev-cell-table"),
            html.Div([_delta_badge(row["d_psnr"], "vs bicubic"), _delta_badge(row.get("d_base"), "vs base")],
                     className="ev-cell-badges"),
        ], className="ev-detail-row")
        return title, metrics, fig

    @app.callback(
        Output("download", "data"),
        Input("export", "n_clicks"),
        *SPEC_STATES,
        State("grid-order", "data"),
        prevent_initial_call=True,
    )
    def export(_, data_dir, n, seed, crop, subset, model, base, order):
        spec, model_path, base_path = _context(data_dir, n, seed, crop, subset, model, base)
        rows = {r["stem"]: r for r in load_rows(spec, model_path, base_path) or []}
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[s] for s in order or [] if s in rows)
        name = f"eval_{model_path.stem}_{spec.key}.csv"
        return {"content": buf.getvalue(), "filename": name}
