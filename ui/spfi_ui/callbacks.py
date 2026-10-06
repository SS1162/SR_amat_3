from pathlib import Path

from dash import Input, Output, State, ctx, html

from .training_runner import RUN


def _parse_upload(contents, filename):
    if not contents:
        return None
    return html.Div(
        [html.Span("✓", className="file-check"), html.Span(filename or "Uploaded file", className="file-name")],
        className="file-chip",
    )


def _summarize_uploads(*uploads):
    """uploads: (contents, filenames) pairs from multi-file dcc.Upload components."""
    items = []
    for contents, names in uploads:
        if not contents:
            continue
        if isinstance(contents, str):
            contents, names = [contents], [names]
        names = names or [None] * len(contents)
        items.extend(_parse_upload(content, name) for content, name in zip(contents, names))
    return items or "No files selected yet."


def _render_training(snap, error=None):
    """Map a TrainingRun snapshot to the training-controls outputs.

    Returns: timer disabled, start disabled, stop disabled, stop class, resume class,
    status text, status class, status copy, progress value.
    """
    state, iteration = snap["state"], snap["iteration"]
    loss = f" · loss {snap['loss']:.4f}" if snap["loss"] is not None else ""
    ckpt_every = snap["ckpt_every"]
    progress = 100 * (iteration % ckpt_every) / ckpt_every if ckpt_every else 0
    stop_shown, resume_hidden = "danger-action", "resume-action d-none"

    if state == "preparing":
        out = (False, True, False, stop_shown, resume_hidden, "Preparing", "warning",
               "Saving data and building the model…", 0)
    elif state == "running":
        out = (False, True, False, stop_shown, resume_hidden, "Running", "success",
               f"Iteration {iteration}{loss} · bar = progress to next checkpoint", progress)
    elif state == "stopping":
        out = (False, True, True, stop_shown, resume_hidden, "Stopping", "warning",
               "Finishing the current iteration…", progress)
    elif state == "stopped":
        out = (True, False, True, "danger-action d-none", "resume-action", "Stopped", "warning",
               f"Stopped at iteration {iteration}{loss}", progress)
    elif state == "finished":
        out = (True, False, True, stop_shown, resume_hidden, "Finished", "success",
               f"Early stopping at iteration {iteration} · checkpoints in {snap['run_dir']}", 100)
    elif state == "error":
        out = (True, False, True, stop_shown, resume_hidden, "Error", "danger", snap["error"], 0)
    else:
        out = (True, False, True, stop_shown, resume_hidden, "Idle", "muted", "No active training run", 0)

    timer_off, start_off, stop_off, stop_cls, resume_cls, text, tone, copy, value = out
    if error:
        text, tone, copy = "Input required", "danger", error
    return timer_off, start_off, stop_off, stop_cls, resume_cls, text, f"status-pill status-{tone}", copy, value


def _empty_preview(message):
    return [
        html.Div("▧", className="preview-placeholder-icon"),
        html.Div(message, className="preview-placeholder-copy"),
    ]


def register_callbacks(app):
    @app.callback(
        Output("train-upload-summary", "children"),
        Input("train-image-upload", "contents"),
        Input("mask-upload", "contents"),
        State("train-image-upload", "filename"),
        State("mask-upload", "filename"),
    )
    def summarize_training_uploads(image_contents, mask_contents, image_names, mask_names):
        return _summarize_uploads((image_contents, image_names), (mask_contents, mask_names))

    @app.callback(
        Output("finetune-upload-summary", "children"),
        Input("finetune-data-upload", "contents"),
        Input("finetune-mask-upload", "contents"),
        State("finetune-data-upload", "filename"),
        State("finetune-mask-upload", "filename"),
    )
    def summarize_finetune_uploads(image_contents, mask_contents, image_names, mask_names):
        return _summarize_uploads((image_contents, image_names), (mask_contents, mask_names))

    @app.callback(
        Output("inference-file-summary", "children"),
        Output("lr-preview", "children"),
        Output("lr-preview", "className"),
        Input("inference-upload", "contents"),
        State("inference-upload", "filename"),
    )
    def show_inference_upload(contents, filenames):
        if not contents:
            return "No files selected yet.", _empty_preview("Input preview will appear here"), "image-preview empty-preview"

        if isinstance(contents, str):
            contents = [contents]
        if filenames is None:
            filenames = [None] * len(contents)
        elif isinstance(filenames, str):
            filenames = [filenames]

        chips = [_parse_upload(content, filename) for content, filename in zip(contents, filenames)]
        return chips, html.Img(src=contents[0], className="preview-image"), "image-preview"

    @app.callback(
        Output("train-poll-timer", "disabled"),
        Output("start-training", "disabled"),
        Output("stop-training", "disabled"),
        Output("stop-training", "className"),
        Output("resume-training", "className"),
        Output("train-status", "children"),
        Output("train-status", "className"),
        Output("train-status-copy", "children"),
        Output("train-progress", "value"),
        Input("start-training", "n_clicks"),
        Input("stop-training", "n_clicks"),
        Input("resume-training", "n_clicks"),
        Input("train-poll-timer", "n_intervals"),
        State("train-image-upload", "contents"),
        State("train-image-upload", "filename"),
        State("mask-upload", "contents"),
        State("mask-upload", "filename"),
        State("output-path", "value"),
        State("run-name", "value"),
        State("device-select", "value"),
    )
    def control_training(_start, _stop, _resume, _tick, images, image_names, masks, mask_names,
                         output_path, run_name, device):
        error = None
        try:
            if ctx.triggered_id == "start-training":
                if not output_path or not run_name:
                    raise ValueError("Output directory and run name are required")
                RUN.start(images, image_names, masks, mask_names,
                          Path(output_path) / run_name, None if device == "auto" else device)
            elif ctx.triggered_id == "stop-training":
                RUN.stop()
            elif ctx.triggered_id == "resume-training":
                RUN.resume()
        except ValueError as exc:
            error = str(exc)
        return _render_training(RUN.snapshot(), error)

    @app.callback(
        Output("inference-status", "children"),
        Output("inference-status", "className"),
        Output("hr-preview", "children"),
        Output("hr-preview", "className"),
        Input("run-inference", "n_clicks"),
        State("inference-upload", "contents"),
        prevent_initial_call=True,
    )
    def demo_inference(n_clicks, contents):
        if not contents:
            return (
                "Input required",
                "status-pill status-warning",
                _empty_preview("Upload an image before running inference"),
                "image-preview empty-preview",
            )

        first = contents[0] if isinstance(contents, list) else contents
        return (
            "UI preview",
            "status-pill status-success",
            html.Div(
                [
                    html.Img(src=first, className="preview-image result-demo"),
                    html.Div("Backend output placeholder", className="result-ribbon"),
                ],
                className="result-wrap",
            ),
            "image-preview",
        )
