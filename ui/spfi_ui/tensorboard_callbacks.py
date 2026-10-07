"""Callbacks for the TensorBoard tab: poll the run folder and refresh cards, charts and samples.

Every REFRESH_MS the page asks metrics_reader for new data. Charts and images are only
re-sent when something actually changed, so an idle run costs almost nothing.
"""

import base64
import io
import math

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate
from PIL import Image

from . import metrics_reader as mr
from .tensorboard_page import IMAGE_TILES, empty_figure

TAB_ID = "tensorboard"
MAX_RAW_POINTS = 3000          # thin out the raw train curve beyond this many points
ERROR_VMAX = 0.2               # fixed scale for the error map, so steps are comparable
STALE_AFTER_S = 60             # "running" with no new lines for this long looks stuck

COLORS = {
    "train": "#54d7ff",
    "val": "#a886ff",
    "psnr": "#66e2b0",
    "bicubic": "#ffca75",
    "charb": "#7ce2ff",
    "ssim": "#ff9bac",
}

STATUS_PILLS = {
    "waiting": ("Waiting for run", "muted"),
    "running": ("Running", "success"),
    "stopped": ("Stopped", "warning"),
    "early_stopped": ("Early stopped", "warning"),
    "done": ("Finished", "success"),
}

CARD_IDS = ["iter", "best", "val", "patience", "cost"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, pattern, fallback="—"):
    return fallback if value is None else pattern.format(value)


def _ago(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _status_pill(summary):
    text, tone = STATUS_PILLS.get(summary["status"], STATUS_PILLS["waiting"])
    since = summary["seconds_since_update"]
    if summary["status"] == "running" and since is not None and since > STALE_AFTER_S:
        text, tone = f"No updates · {_ago(since)}", "warning"
    return text, f"status-pill status-{tone}"


def _cards(summary):
    """(value, hint, extra) for each card in CARD_IDS order."""
    s = summary
    gain = None
    if s["best_psnr"] is not None and s["bicubic_psnr"] is not None:
        gain = s["best_psnr"] - s["bicubic_psnr"]

    change = s["val_loss_change_pct"]
    if change is None:
        val_hint = "Latest validation"
    else:
        val_hint = f"{'▼' if change < 0 else '▲'} {abs(change):.1f}% vs previous check"

    patience = s["patience"] or 0
    fill = min(100, 100 * s["no_improve"] / patience) if patience else 0
    bar = html.Div(html.Span(style={"width": f"{fill:.0f}%"}), className="tb-bar")

    params = s["params"]
    return [
        (f"{s['step']:,}", _fmt(s["its_per_sec"], "{:.1f} it/s"), None),
        (_fmt(s["best_psnr"], "{:.2f} dB"),
         "No validation yet" if s["best_psnr_step"] is None else
         f"@ iter {s['best_psnr_step']:,}" + (f" · {gain:+.2f} dB vs bicubic" if gain is not None else ""),
         None),
        (_fmt(s["val_loss"], "{:.4f}"), val_hint, None),
        (f"{s['no_improve']} / {patience or '—'}", "Checks without improvement", bar),
        (_fmt(s["gflops"], "{:.2f} GFLOPs"),
         "Parameters unknown" if params is None else f"{params / 1e6:.2f}M parameters · 64×64 input",
         None),
    ]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _smooth(values, weight):
    """TensorBoard-style exponential moving average with bias correction."""
    out, last, debias = [], 0.0, 1.0
    for v in values:
        last = last * weight + (1 - weight) * v
        debias *= weight
        out.append(last / (1 - debias) if debias < 1 else v)
    return out


def _thin(xs, ys):
    stride = max(1, math.ceil(len(xs) / MAX_RAW_POINTS))
    return xs[::stride], ys[::stride]


def _style(fig, run, y_title):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8ea3bd", size=11),
        margin=dict(l=56, r=16, t=10, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        uirevision=run,                       # keep zoom / hidden traces across refreshes
        xaxis=dict(title="Iteration", gridcolor="rgba(255,255,255,.05)", zeroline=False),
        yaxis=dict(title=y_title, gridcolor="rgba(255,255,255,.05)", zeroline=False),
    )
    return fig


EVENT_MARKERS = {
    "stop": ("Stopped", "#ffca75"),
    "resume": ("Resumed", "#66e2b0"),
    "early_stop": ("Early stop", "#ff718b"),
    "done": ("Finished", "#66e2b0"),
}


def _add_events(fig, events):
    """Dotted vertical line + small label for every stop / resume / end of training."""
    for e in events:
        label, color = EVENT_MARKERS.get(e.get("event"), (None, None))
        if label is None or not e.get("step"):
            continue
        fig.add_vline(x=e["step"], line=dict(color=color, width=1, dash="dot"), opacity=0.7,
                      annotation=dict(text=label, font=dict(color=color, size=10), textangle=-90,
                                      yanchor="top", xanchor="right"),
                      annotation_position="top left")


def loss_figure(data, run, smoothing, split):
    if not data.train and not data.val:
        return empty_figure("No loss values yet")
    fig = go.Figure()
    if data.train:
        steps = [r["step"] for r in data.train]
        loss = [r["loss"] for r in data.train]
        raw_x, raw_y = _thin(steps, loss)
        fig.add_scatter(x=raw_x, y=raw_y, name="train (raw)", mode="lines", hoverinfo="skip",
                        line=dict(color=COLORS["train"], width=1), opacity=0.22, showlegend=False)
        fig.add_scatter(x=steps, y=_smooth(loss, smoothing), name="train", mode="lines",
                        line=dict(color=COLORS["train"], width=2))
        if split:
            for key, name, color in (("charbonnier", "Charbonnier", COLORS["charb"]),
                                     ("ssim_loss", "1 − SSIM", COLORS["ssim"])):
                ys = [r.get(key) for r in data.train]
                if all(y is not None for y in ys):
                    fig.add_scatter(x=steps, y=_smooth(ys, smoothing), name=name, mode="lines",
                                    line=dict(color=color, width=1.5, dash="dot"))
    if data.val:
        fig.add_scatter(x=[r["step"] for r in data.val], y=[r["val_loss"] for r in data.val],
                        name="validation", mode="lines+markers",
                        line=dict(color=COLORS["val"], width=2), marker=dict(size=5))
    _add_events(fig, data.events)
    return _style(fig, run, "Loss")


def psnr_figure(data, run):
    if not data.val:
        return empty_figure(f"First validation at iteration {(data.meta or {}).get('val_every', 250)}")
    fig = go.Figure()
    steps = [r["step"] for r in data.val]
    fig.add_scatter(x=steps, y=[r["val_psnr"] for r in data.val], name="SPFI (val)",
                    mode="lines+markers", line=dict(color=COLORS["psnr"], width=2), marker=dict(size=5))

    bicubic = (data.meta or {}).get("bicubic_psnr")
    if bicubic is not None:
        fig.add_scatter(x=[steps[0], steps[-1]], y=[bicubic, bicubic], name="bicubic", mode="lines",
                        line=dict(color=COLORS["bicubic"], width=1.5, dash="dash"), hoverinfo="skip")

    best = [r for r in data.val if r.get("is_best")]
    if best:
        b = best[-1]
        fig.add_scatter(x=[b["step"]], y=[b["val_psnr"]], name="best checkpoint", mode="markers",
                        marker=dict(symbol="star", size=14, color=COLORS["bicubic"],
                                    line=dict(color="#04111c", width=1)))
    _add_events(fig, data.events)
    return _style(fig, run, "PSNR (dB)")


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

# black → violet → coral → yellow, for |SR − HR|
_CMAP_POS = [0.0, 0.35, 0.7, 1.0]
_CMAP_RGB = np.array([[5, 11, 19], [110, 60, 200], [255, 113, 139], [255, 230, 140]], dtype=np.float32)


def _error_uri(err):
    t = np.clip(err / ERROR_VMAX, 0, 1)
    rgb = np.stack([np.interp(t, _CMAP_POS, _CMAP_RGB[:, c]) for c in range(3)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _img(src, alt):
    if src is None:
        return html.Div("▧", className="preview-placeholder-icon")
    return html.Img(src=src, alt=alt)


def _slider_marks(steps):
    """Label about 6 evenly spaced positions so the marks never overlap."""
    if not steps:
        return {}
    every = max(1, math.ceil(len(steps) / 6))
    idx = sorted(set(range(0, len(steps), every)) | {len(steps) - 1})
    return {i: f"{steps[i]:,}" for i in idx}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_tensorboard_callbacks(app):
    @app.callback(
        Output("tb-run", "options"),
        Output("tb-run", "value"),
        Output("tb-empty", "className"),
        Output("tb-content", "className"),
        Input("tb-interval", "n_intervals"),
        Input("main-tabs", "active_tab"),
        State("tb-run", "value"),
        State("tb-run", "options"),
    )
    def refresh_runs(_, active_tab, current, current_options):
        if active_tab != TAB_ID:
            raise PreventUpdate
        runs = mr.list_runs()
        options = [{"label": r, "value": r} for r in runs]
        known = {o["value"] for o in current_options or []}
        new_runs = [r for r in runs if r not in known]
        if current_options and new_runs:
            value = new_runs[0]                 # a run just appeared: jump to it
        else:
            value = current if current in runs else (runs[0] if runs else None)
        return (no_update if options == current_options else options,
                no_update if value == current else value,
                "panel-card tb-empty" + ("" if not runs else " d-none"),
                "" if runs else "d-none")

    @app.callback(
        Output("tb-status", "children"),
        Output("tb-status", "className"),
        *[Output(f"tb-{c}-{part}", "children") for c in CARD_IDS for part in ("value", "hint", "extra")],
        Output("tb-loss-graph", "figure"),
        Output("tb-psnr-graph", "figure"),
        Output("tb-chart-key", "data"),
        Input("tb-interval", "n_intervals"),
        Input("main-tabs", "active_tab"),
        Input("tb-run", "value"),
        Input("tb-smoothing", "value"),
        Input("tb-loss-split", "value"),
        State("tb-chart-key", "data"),
    )
    def refresh_metrics(_, active_tab, run, smoothing, split, chart_key):
        if active_tab != TAB_ID:
            raise PreventUpdate

        if not run:
            status = (STATUS_PILLS["waiting"][0], "status-pill status-muted")
            cards = [("—", "No data yet", None)] * len(CARD_IDS)
            figs = [empty_figure("No runs found in runs/"), empty_figure("No runs found in runs/")]
            key = None
        else:
            data = mr.load_run(run)
            summary = mr.summarize(data)
            status = _status_pill(summary)
            cards = _cards(summary)
            key = [run, len(data.train), len(data.val), len(data.events), smoothing, bool(split)]
            if key == chart_key:
                figs = [no_update, no_update]
            else:
                figs = [loss_figure(data, run, smoothing or 0, bool(split)), psnr_figure(data, run)]

        flat_cards = [part for card in cards for part in card]
        return (*status, *flat_cards, *figs, key)

    @app.callback(
        Output("tb-step-slider", "max"),
        Output("tb-step-slider", "marks"),
        Output("tb-step-slider", "value"),
        Output("tb-steps", "data"),
        Input("tb-interval", "n_intervals"),
        Input("main-tabs", "active_tab"),
        Input("tb-run", "value"),
        State("tb-step-slider", "value"),
        State("tb-steps", "data"),
    )
    def refresh_steps(_, active_tab, run, position, old_steps):
        if active_tab != TAB_ID:
            raise PreventUpdate
        steps = mr.sample_steps(run) if run else []
        if steps == old_steps and ctx.triggered_id != "tb-run":
            raise PreventUpdate

        # follow the newest iteration unless the user moved the slider back
        following = not old_steps or position is None or position >= len(old_steps) - 1
        if ctx.triggered_id == "tb-run" or following or position >= len(steps):
            position = max(0, len(steps) - 1)
        return max(0, len(steps) - 1), _slider_marks(steps), position, steps

    @app.callback(
        *[Output(f"tb-img-{key}", "children") for key, _ in IMAGE_TILES],
        Output("tb-step-label", "children"),
        Input("tb-steps", "data"),
        Input("tb-step-slider", "value"),
        Input("tb-sample", "value"),
        State("tb-run", "value"),
    )
    def show_sample(steps, position, sample, run):
        empty = [_img(None, "")] * len(IMAGE_TILES)
        if not run:
            return (*empty, "No run selected")

        fixed = {key: mr.load_sample(run, sample, key) for key in ("lr", "bicubic", "hr")}
        if not steps:
            val_every = (mr.read_meta(run) or {}).get("val_every", 250)
            images = [_img(fixed.get(key), key) if key in fixed else _img(None, key) for key, _ in IMAGE_TILES]
            return (*images, f"First samples at iteration {val_every}")

        step = steps[min(position or 0, len(steps) - 1)]
        err = mr.load_error_map(run, sample, step)
        sources = {
            **fixed,
            "sr": mr.load_sample(run, sample, step),
            "error": _error_uri(err) if err is not None else None,
        }
        images = [_img(sources[key], label) for key, label in IMAGE_TILES]

        label = f"Iteration {step:,}"
        if err is not None:
            mse = float(np.mean(err ** 2))
            label += f" · PSNR {10 * math.log10(1 / mse):.2f} dB" if mse > 0 else " · PSNR ∞"
        return (*images, label)

    @app.callback(
        Output("tb-play-timer", "disabled"),
        Output("tb-play", "children"),
        Output("tb-play", "className"),
        Output("tb-step-slider", "value", allow_duplicate=True),
        Input("tb-play", "n_clicks"),
        Input("tb-play-timer", "n_intervals"),
        Input("tb-run", "value"),
        State("tb-play-timer", "disabled"),
        State("tb-step-slider", "value"),
        State("tb-steps", "data"),
        prevent_initial_call=True,
    )
    def play_timeline(_, __, ___, paused, position, steps):
        """▶ walks the slider through the saved iterations; ❚❚ or reaching the end stops it."""
        stopped = (True, "▶", "tb-play-btn", no_update)
        last = len(steps or []) - 1
        if ctx.triggered_id == "tb-run" or last < 1:
            return stopped

        if ctx.triggered_id == "tb-play":
            if not paused:
                return stopped
            start = 0 if (position or 0) >= last else position   # at the end: replay from the start
            return False, "❚❚", "tb-play-btn playing", start

        nxt = (position or 0) + 1                                 # timer tick
        if nxt >= last:
            return True, "▶", "tb-play-btn", last
        return no_update, no_update, no_update, nxt
