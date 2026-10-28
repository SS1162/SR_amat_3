"""Plotly figures: distributions, scatter plots and the zoomable detail view."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import png_uri

# Validated for the dark surface (dataviz validate_palette.js, all checks pass).
COLOR_MODEL = "#1f8fc0"
COLOR_BICUBIC = "#b97f22"
COLOR_BASE = "#8466e0"
INK = "#c7d4e5"
MUTED = "#8ea3bd"
GRID = "rgba(151,177,209,0.12)"
SURFACE = "#101f34"

# |SR - HR| above this is drawn at full intensity, so maps of different images compare.
ERROR_MAX = 0.25


def _base_layout(fig: go.Figure, x: str, y: str) -> go.Figure:
    """Titles live in the HTML card around the graph, so the legend gets the top row."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        margin=dict(l=52, r=16, t=28, b=44), height=300,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK)),
        bargap=0.06,
    )
    fig.update_xaxes(title=x, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=y, gridcolor=GRID, zeroline=False)
    return fig


def psnr_hist(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    series = [("Bicubic", "bicubic_psnr", COLOR_BICUBIC), ("Model", "psnr", COLOR_MODEL)]
    if any("base_psnr" in r for r in rows):
        series.append(("Base", "base_psnr", COLOR_BASE))
    for name, key, color in series:
        fig.add_histogram(x=[r[key] for r in rows if key in r], name=name, marker_color=color,
                          opacity=0.65, nbinsx=30,
                          hovertemplate=f"{name}<br>%{{x}} dB<br>%{{y}} images<extra></extra>")
    fig.update_layout(barmode="overlay")
    return _base_layout(fig, "PSNR (dB)", "Images")


def delta_hist(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=[r["d_psnr"] for r in rows], name="Model − bicubic", marker_color=COLOR_MODEL,
                      opacity=0.75, nbinsx=30,
                      hovertemplate="Model − bicubic<br>%{x} dB<br>%{y} images<extra></extra>")
    if any("d_base" in r for r in rows):
        fig.add_histogram(x=[r["d_base"] for r in rows if "d_base" in r], name="Model − base",
                          marker_color=COLOR_BASE, opacity=0.6, nbinsx=30,
                          hovertemplate="Model − base<br>%{x} dB<br>%{y} images<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
    return _base_layout(fig, "ΔPSNR (dB)", "Images")


def _scatter(rows: list[dict], x: str, y: str, name: str, color: str, xlabel: str, ylabel: str) -> go.Scatter:
    pts = [r for r in rows if x in r and y in r]
    return go.Scatter(
        x=[r[x] for r in pts], y=[r[y] for r in pts], mode="markers", name=name,
        customdata=[[r["stem"], r["sigma"]] for r in pts],
        marker=dict(size=8, color=color, opacity=0.85, line=dict(color=SURFACE, width=1)),
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}"
                       "<br>σ: %{customdata[1]:.2f}<extra></extra>"),
    )


def psnr_scatter(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_scatter(rows, "bicubic_psnr", "psnr", "Model", COLOR_MODEL, "Bicubic", "Model"))
    if any("base_psnr" in r for r in rows):
        fig.add_trace(_scatter(rows, "bicubic_psnr", "base_psnr", "Base", COLOR_BASE, "Bicubic", "Base"))
    vals = [r[k] for r in rows for k in ("bicubic_psnr", "psnr", "base_psnr") if k in r]
    lo, hi = min(vals) - 0.5, max(vals) + 0.5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                             line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
    fig = _base_layout(fig, "Bicubic PSNR (dB)", "PSNR (dB)")
    # Same range on both axes keeps y = x on the diagonal without the empty margins of scaleanchor.
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi])
    return fig


def sigma_scatter(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_scatter(rows, "sigma", "d_psnr", "Model − bicubic", COLOR_MODEL, "σ", "ΔPSNR"))
    if any("d_base" in r for r in rows):
        fig.add_trace(_scatter(rows, "sigma", "d_base", "Model − base", COLOR_BASE, "σ", "ΔPSNR"))
    fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
    return _base_layout(fig, "Blur σ (HR px)", "ΔPSNR (dB)")


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, "", "")


def detail_figure(views: dict[str, np.ndarray], titles: dict[str, str]) -> go.Figure:
    """Images and error maps in one figure with matched axes: zoom / pan one, all follow."""
    hr = views["hr"]
    top = [k for k in ("lr", "bicubic", "sr", "base", "hr") if k in views]
    bottom = [k for k in ("bicubic", "sr", "base") if k in views]
    cols = len(top)
    names = [titles[k] for k in top] + [f"|{titles[k]} − HR|" for k in bottom] + [""] * (cols - len(bottom))
    fig = make_subplots(rows=2, cols=cols, subplot_titles=names,
                        horizontal_spacing=0.01, vertical_spacing=0.06)
    for c, key in enumerate(top, start=1):
        fig.add_trace(go.Image(source=png_uri(views[key]), name=key,
                               hovertemplate=f"{titles[key]}<br>x %{{x}}, y %{{y}}<extra></extra>"),
                      row=1, col=c)
    for c, key in enumerate(bottom, start=1):
        err = np.abs(views[key] - hr)
        fig.add_trace(go.Heatmap(z=err, zmin=0, zmax=ERROR_MAX, colorscale="Inferno",
                                 showscale=(c == len(bottom)),
                                 colorbar=dict(title="|err|", len=0.45, y=0.22, thickness=12),
                                 hovertemplate="x %{x}, y %{y}<br>|err| %{z:.3f}<extra></extra>"),
                      row=2, col=c)
    fig.update_xaxes(matches="x", showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(matches="y", autorange="reversed", showticklabels=False, showgrid=False,
                     zeroline=False, scaleanchor="x", scaleratio=1)
    h, w = hr.shape
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=28, b=4), height=int(2 * 280 * h / w) + 60,
        font=dict(color=MUTED, size=12), dragmode="zoom", uirevision="detail",
    )
    fig.update_annotations(font=dict(size=12, color=INK))
    return fig
