from pathlib import Path

from dash import Input, Output, State, ctx

from ..callbacks import choose_file, choose_folder, output_directory_feedback
from ..components import summarize_uploads
from .runner import RUN


ACTIVE_STATES = ("preparing", "running", "stopping")


def _render_training(snap, mode, error=None):
    """Map a TrainingRun snapshot to the outputs of the ``mode`` panel ("scratch" or "finetune").

    Returns: timer disabled, start disabled, stop disabled, stop class, resume class,
    status text, status class, status copy, progress value.
    """
    state, iteration = snap["state"], snap["iteration"]
    loss = f" · loss {snap['loss']:.4f}" if snap["loss"] is not None else ""
    ckpt_every = snap["ckpt_every"]
    progress = 100 * (iteration % ckpt_every) / ckpt_every if ckpt_every else 0
    stop_shown, resume_hidden = "danger-action", "resume-action d-none"

    if snap["mode"] not in (None, mode):
        # The single training slot belongs to the other tab.
        if state in ACTIVE_STATES:
            other = "fine-tuning" if snap["mode"] == "finetune" else "from-scratch"
            out = (False, True, True, stop_shown, resume_hidden, "Busy", "muted",
                   f"A {other} run is active; stop it to start here", 0)
        else:
            out = (True, False, True, stop_shown, resume_hidden, "Idle", "muted", "No active training run", 0)
    elif state == "preparing":
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


def _format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _render_metrics(snap, mode):
    """Returns value/hint pairs for the loss, validation, iteration and elapsed cards of the ``mode`` panel."""
    if snap["state"] == "idle" or snap["mode"] != mode:
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


def _register_training_panel(app, prefix, mode, image_upload, mask_upload, weights_input=None):
    """Callbacks for one training-controls panel (see page.training_controls(prefix))."""
    p = prefix
    weights_state = [State(weights_input, "value")] if weights_input else []

    @app.callback(
        Output(f"{p}train-poll-timer", "disabled"),
        Output(f"{p}start-training", "disabled"),
        Output(f"{p}stop-training", "disabled"),
        Output(f"{p}stop-training", "className"),
        Output(f"{p}resume-training", "className"),
        Output(f"{p}train-status", "children"),
        Output(f"{p}train-status", "className"),
        Output(f"{p}train-status-copy", "children"),
        Output(f"{p}train-progress", "value"),
        Input(f"{p}start-training", "n_clicks"),
        Input(f"{p}stop-training", "n_clicks"),
        Input(f"{p}resume-training", "n_clicks"),
        Input(f"{p}train-poll-timer", "n_intervals"),
        State(image_upload, "contents"),
        State(image_upload, "filename"),
        State(mask_upload, "contents"),
        State(mask_upload, "filename"),
        State(f"{p}output-path", "value"),
        State(f"{p}run-name", "value"),
        State(f"{p}device-select", "value"),
        *weights_state,
    )
    def control_training(_start, _stop, _resume, _tick, images, image_names, masks, mask_names,
                         output_path, run_name, device, weights=None):
        error = None
        try:
            if ctx.triggered_id == f"{p}start-training":
                if not output_path or not run_name:
                    raise ValueError("Output directory and run name are required")
                RUN.start(images, image_names, masks, mask_names, Path(output_path) / run_name,
                          None if device == "auto" else device, mode=mode, weights=weights)
            elif ctx.triggered_id == f"{p}stop-training":
                RUN.stop()
            elif ctx.triggered_id == f"{p}resume-training":
                RUN.resume()
        except ValueError as exc:
            error = str(exc)
        return _render_training(RUN.snapshot(), mode, error)

    @app.callback(
        Output(f"{p}metric-loss-value", "children"),
        Output(f"{p}metric-loss-hint", "children"),
        Output(f"{p}metric-val-value", "children"),
        Output(f"{p}metric-val-hint", "children"),
        Output(f"{p}metric-iter-value", "children"),
        Output(f"{p}metric-iter-hint", "children"),
        Output(f"{p}metric-elapsed-value", "children"),
        Output(f"{p}metric-elapsed-hint", "children"),
        Input(f"{p}train-poll-timer", "n_intervals"),
        Input(f"{p}train-poll-timer", "disabled"),
    )
    def update_training_metrics(_tick, _timer_disabled):
        return _render_metrics(RUN.snapshot(), mode)


def register_training_callbacks(app):
    @app.callback(
        Output("output-path", "value"),
        Input("choose-output", "n_clicks"),
        State("output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_training_output_directory(n_clicks, current_value):
        selected = choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("ft-output-path", "value"),
        Input("ft-choose-output", "n_clicks"),
        State("ft-output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_finetune_output_directory(n_clicks, current_value):
        selected = choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("ft-weights", "value"),
        Input("choose-finetune-model", "n_clicks"),
        State("ft-weights", "value"),
        prevent_initial_call=True,
    )
    def choose_finetune_model_file(n_clicks, current_value):
        selected = choose_file("Select fine-tuning model file")
        return selected if selected else current_value

    @app.callback(
        Output("output-path", "className"),
        Output("output-path-feedback", "children"),
        Output("output-path-feedback", "className"),
        Input("output-path", "value"),
    )
    def validate_training_output_path(path_value):
        _valid, input_class, feedback, feedback_class = output_directory_feedback(path_value)
        return input_class, feedback, feedback_class

    @app.callback(
        Output("ft-output-path", "className"),
        Output("ft-output-path-feedback", "children"),
        Output("ft-output-path-feedback", "className"),
        Input("ft-output-path", "value"),
    )
    def validate_finetune_output_path(path_value):
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
        Output("finetune-upload-summary", "children"),
        Input("finetune-data-upload", "contents"),
        Input("finetune-mask-upload", "contents"),
        Input("finetune-small-image-upload", "contents"),
        State("finetune-data-upload", "filename"),
        State("finetune-mask-upload", "filename"),
        State("finetune-small-image-upload", "filename"),
    )
    def summarize_finetune_uploads(image_contents, mask_contents, small_contents, image_names, mask_names, small_names):
        return summarize_uploads(
            (image_contents, image_names),
            (mask_contents, mask_names),
            (small_contents, small_names),
        )

    _register_training_panel(app, "", "scratch", "train-image-upload", "mask-upload")
    _register_training_panel(app, "ft-", "finetune", "finetune-data-upload", "finetune-mask-upload",
                             weights_input="ft-weights")
