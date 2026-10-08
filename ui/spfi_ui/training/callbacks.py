from pathlib import Path

from dash import Input, Output, State, ctx

from ..callbacks import choose_file, choose_folder, output_directory_feedback
from ..components import summarize_uploads
from .runner import RUN


MODE_LABELS = {"scratch": "From scratch", "finetune": "Fine-tuning"}


def _training_mode(base_model):
    """An empty base model trains from scratch; a checkpoint path makes the run a fine-tune.

    Returns (mode, weights) for RUN.start.
    """
    base_model = (base_model or "").strip()
    return ("finetune", base_model) if base_model else ("scratch", None)


def _render_training(snap, error=None):
    """Map a TrainingRun snapshot to the training panel outputs.

    Returns: timer disabled, start disabled, stop disabled, stop class, resume class,
    status text, status class, status copy, progress value.
    """
    state, iteration = snap["state"], snap["iteration"]
    loss = f" · loss {snap['loss']:.4f}" if snap["loss"] is not None else ""
    mode = f"{MODE_LABELS[snap['mode']]} · " if snap["mode"] in MODE_LABELS else ""
    ckpt_every = snap["ckpt_every"]
    progress = 100 * (iteration % ckpt_every) / ckpt_every if ckpt_every else 0
    stop_shown, resume_hidden = "danger-action", "resume-action d-none"

    if state == "preparing":
        out = (False, True, False, stop_shown, resume_hidden, "Preparing", "warning",
               f"{mode}Saving data and building the model…", 0)
    elif state == "running":
        out = (False, True, False, stop_shown, resume_hidden, "Running", "success",
               f"{mode}Iteration {iteration}{loss} · bar = progress to next checkpoint", progress)
    elif state == "stopping":
        out = (False, True, True, stop_shown, resume_hidden, "Stopping", "warning",
               "Finishing the current iteration…", progress)
    elif state == "stopped":
        out = (True, False, True, "danger-action d-none", "resume-action", "Stopped", "warning",
               f"{mode}Stopped at iteration {iteration}{loss}", progress)
    elif state == "finished":
        out = (True, False, True, stop_shown, resume_hidden, "Finished", "success",
               f"{mode}Early stopping at iteration {iteration} · checkpoints in {snap['run_dir']}", 100)
    elif state == "error":
        out = (True, False, True, stop_shown, resume_hidden, "Error", "danger", snap["error"], 0)
    else:
        out = (True, False, True, stop_shown, resume_hidden, "Idle", "muted", "No active training run", 0)

    timer_off, start_off, stop_off, stop_cls, resume_cls, text, tone, copy, value = out
    if error:
        text, tone, copy = "Input required", "danger", error
    return timer_off, start_off, stop_off, stop_cls, resume_cls, text, f"status-pill status-{tone}", copy, value


def _format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _render_metrics(snap):
    """Returns value/hint pairs for the loss, validation, iteration and elapsed cards."""
    if snap["state"] == "idle":
        return ("—", "Available when a run begins", "—", "Available after the first checkpoint",
                "0", "No active training run", "00:00", "Training time, excluding pauses")

    iteration, ckpt_every = snap["iteration"], snap["ckpt_every"]
    loss = f"{snap['loss']:.4f}" if snap["loss"] is not None else "—"
    if snap["val"] is not None:
        val_loss, val_psnr = snap["val"]
        val, val_hint = f"{val_psnr:.2f} dB", f"val loss {val_loss:.4f} · higher PSNR is better"
    else:
        val, val_hint = "—", "Available after the first checkpoint"
    next_ckpt = (iteration // ckpt_every + 1) * ckpt_every if ckpt_every else None
    iter_hint = f"Next checkpoint at {next_ckpt}" if next_ckpt else "Preparing…"
    return (loss, "Latest training iteration", val, val_hint,
            str(iteration), iter_hint, _format_elapsed(snap["elapsed"]), "Training time, excluding pauses")


def _describe_base_model(base_model):
    """Header pill text and the hint under the base-model field."""
    mode, weights = _training_mode(base_model)
    if mode == "scratch":
        return MODE_LABELS[mode], "Training from scratch"
    return MODE_LABELS[mode], f"Fine-tuning from {Path(weights).name}"


def register_training_callbacks(app):
    @app.callback(
        Output("output-path", "value"),
        Input("choose-output", "n_clicks"),
        State("output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_output_directory(n_clicks, current_value):
        selected = choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("base-model", "value"),
        Input("choose-base-model", "n_clicks"),
        State("base-model", "value"),
        prevent_initial_call=True,
    )
    def choose_base_model_file(n_clicks, current_value):
        selected = choose_file("Select base model file")
        return selected if selected else current_value

    @app.callback(
        Output("train-mode-pill", "children"),
        Output("base-model-hint", "children"),
        Input("base-model", "value"),
    )
    def describe_base_model(base_model):
        return _describe_base_model(base_model)

    @app.callback(
        Output("output-path", "className"),
        Output("output-path-feedback", "children"),
        Output("output-path-feedback", "className"),
        Input("output-path", "value"),
    )
    def validate_output_path(path_value):
        _valid, input_class, feedback, feedback_class = output_directory_feedback(path_value)
        return input_class, feedback, feedback_class

    @app.callback(
        Output("train-upload-summary", "children"),
        Input("train-image-upload", "contents"),
        Input("mask-upload", "contents"),
        Input("train-small-image-upload", "contents"),
        State("train-image-upload", "filename"),
        State("mask-upload", "filename"),
        State("train-small-image-upload", "filename"),
    )
    def summarize_training_uploads(image_contents, mask_contents, small_contents, image_names, mask_names, small_names):
        return summarize_uploads(
            (image_contents, image_names),
            (mask_contents, mask_names),
            (small_contents, small_names),
        )

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
        State("base-model", "value"),
    )
    def control_training(_start, _stop, _resume, _tick, images, image_names, masks, mask_names,
                         output_path, run_name, device, base_model):
        error = None
        try:
            if ctx.triggered_id == "start-training":
                if not output_path or not run_name:
                    raise ValueError("Output directory and run name are required")
                mode, weights = _training_mode(base_model)
                RUN.start(images, image_names, masks, mask_names, Path(output_path) / run_name,
                          None if device == "auto" else device, mode=mode, weights=weights)
            elif ctx.triggered_id == "stop-training":
                RUN.stop()
            elif ctx.triggered_id == "resume-training":
                RUN.resume()
        except ValueError as exc:
            error = str(exc)
        return _render_training(RUN.snapshot(), error)

    @app.callback(
        Output("metric-loss-value", "children"),
        Output("metric-loss-hint", "children"),
        Output("metric-val-value", "children"),
        Output("metric-val-hint", "children"),
        Output("metric-iter-value", "children"),
        Output("metric-iter-hint", "children"),
        Output("metric-elapsed-value", "children"),
        Output("metric-elapsed-hint", "children"),
        Input("train-poll-timer", "n_intervals"),
        Input("train-poll-timer", "disabled"),
    )
    def update_training_metrics(_tick, _timer_disabled):
        return _render_metrics(RUN.snapshot())
