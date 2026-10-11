import time
from pathlib import Path
from typing import Callable

from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.development.base_component import Component
from diskcache import Cache

from inference.events import ProgressEvent, RunStatus

from ..callbacks import choose_file, choose_folder, output_directory_feedback
from ..components import summarize_uploads
from ..config import INFERENCE_RUNS_DIR
from . import job
from .job import JobState

RUN_BUTTON_ID = "run-inference"
CONTINUE_BUTTON_ID = "continue-inference"
PROGRESS_POLL_MS = 1000
RESULTS_ZIP_NAME = "inference_results_{job_id}.zip"
PREVIEW_CLASS = "image-preview"
EMPTY_PREVIEW_CLASS = "image-preview empty-preview"
ENDED_STATUSES = (RunStatus.COMPLETED.value, RunStatus.CANCELLED.value, RunStatus.FAILED.value)
# Pill text and tone per run status; FAILED also shows the error message.
STATUS_PILL = {
    RunStatus.STARTED: ("Starting", "warning"),
    RunStatus.RUNNING: ("Running", "success"),
    RunStatus.COMPLETED: ("Completed", "success"),
    RunStatus.CANCELLED: ("Stopped", "warning"),
    RunStatus.FAILED: ("Failed", "danger"),
}

# Progress outputs, in the order of the tuples passed to set_progress.
PROGRESS_OUTPUTS = [
    Output("inference-status", "children"),
    Output("inference-status", "className"),
    Output("inference-progress", "value"),
    Output("inference-progress", "label"),
    Output("inference-progress-copy", "children"),
    Output("lr-preview", "children"),
    Output("lr-preview", "className"),
    Output("hr-preview", "children"),
    Output("hr-preview", "className"),
    Output("inference-active-job", "data"),
]
ProgressValues = tuple[str, str, int, str, str, Component | list[Component], str, Component | list[Component], str, str]
SetProgress = Callable[[ProgressValues], None]


def _empty_preview(message: str) -> list[Component]:
    """Return the placeholder shown in an image preview without an image."""
    return [
        html.Div("▧", className="preview-placeholder-icon"),
        html.Div(message, className="preview-placeholder-copy"),
    ]


def _status_pill(status: RunStatus, error: str | None) -> tuple[str, str]:
    """Return the status pill text and className for ``status`` (with the error when FAILED)."""
    text, tone = STATUS_PILL[status]
    if status is RunStatus.FAILED and error:
        text = f"{text}: {error}"
    return text, f"status-pill status-{tone}"


def _progress_copy(processed: int, total: int, current_file: str | None) -> str:
    """Return the line under the progress bar, e.g. "3/10 images · img_003.png"."""
    copy = f"{processed}/{total} images"
    return f"{copy} · {current_file}" if current_file else copy


def _as_lists(contents: str | list[str] | None, filenames: str | list[str] | None) -> tuple[list[str], list[str | None]]:
    """Return dcc.Upload contents / filenames as two lists (single and multiple uploads)."""
    if not contents:
        return [], []
    if isinstance(contents, str):
        return [contents], [filenames if isinstance(filenames, str) else None]
    names = list(filenames) if isinstance(filenames, list) else [None] * len(contents)
    return list(contents), names


def _progress_listener(set_progress: SetProgress, job_id: str, first_upload: str | None) -> Callable[[ProgressEvent], None]:
    """Return a listener that sends every ProgressEvent to the UI through ``set_progress``.

    set_progress always sends every output, so the last previews are remembered and
    re-sent for events without images (STARTED, FAILED).
    """
    lr_preview: Component | list[Component] = (
        html.Img(src=first_upload, className="preview-image") if first_upload else _empty_preview("Input preview will appear here")
    )
    lr_class = PREVIEW_CLASS if first_upload else EMPTY_PREVIEW_CLASS
    previews: dict[str, tuple[Component | list[Component], str]] = {
        "lr": (lr_preview, lr_class),
        "hr": (_empty_preview("Processing…"), EMPTY_PREVIEW_CLASS),
    }

    def on_event(event: ProgressEvent) -> None:
        """Convert ``event`` to progress outputs and publish them."""
        if event.lr_image is not None and event.sr_image is not None:
            previews["lr"] = (html.Img(src=job.array_to_data_url(event.lr_image), className="preview-image"), PREVIEW_CLASS)
            previews["hr"] = (html.Img(src=job.array_to_data_url(event.sr_image), className="preview-image"), PREVIEW_CLASS)
        error = str(event.error) if event.error is not None else None
        pill_text, pill_class = _status_pill(event.status, error)
        set_progress(
            (
                pill_text,
                pill_class,
                event.percent,
                f"{event.percent}%",
                _progress_copy(event.processed, event.total, event.current_file),
                *previews["lr"],
                *previews["hr"],
                job_id,
            )
        )

    return on_event


def run_or_continue(
    set_progress: SetProgress,
    cache: Cache,
    contents: str | list[str] | None,
    filenames: str | list[str] | None,
    model_path: str | None,
    last_job: JobState | None,
) -> JobState:
    """Start a new job (Run) or resume the stored one (Continue) and return its final state."""
    upload_contents, upload_names = _as_lists(contents, filenames)
    if ctx.triggered_id == CONTINUE_BUTTON_ID and last_job is not None:
        job_id, run_dir, start_from = last_job["job_id"], Path(last_job["run_dir"]), last_job["next_index"]
    else:
        job_id = job.new_job_id()
        run_dir = job.create_run_dir(INFERENCE_RUNS_DIR, job_id)
        job.cleanup_old_runs(INFERENCE_RUNS_DIR, keep=run_dir, now=time.time())
        job.save_uploads(upload_contents, upload_names, job.input_dir(run_dir))
        start_from = 0

    first_upload = upload_contents[0] if upload_contents else None
    listener = _progress_listener(set_progress, job_id, first_upload)
    return job.run_job(Path(model_path or ""), run_dir, job_id, start_from, cache, listener)


def register_inference_callbacks(app: Dash, cache: Cache) -> None:
    """Register the Inference tab callbacks; ``cache`` is the DiskcacheManager's cache."""

    @app.callback(
        Output("model-path", "value"),
        Input("choose-model", "n_clicks"),
        State("model-path", "value"),
        prevent_initial_call=True,
    )
    def choose_model_file(n_clicks: int, current_value: str | None) -> str | None:
        """Fill the model path from the native file picker."""
        selected = choose_file("Select model file")
        return selected if selected else current_value

    @app.callback(
        Output("inference-output-path", "value"),
        Input("choose-inference-output", "n_clicks"),
        State("inference-output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_inference_output_directory(n_clicks: int, current_value: str | None) -> str | None:
        """Fill the destination directory from the native folder picker."""
        selected = choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output(RUN_BUTTON_ID, "disabled"),
        Output("inference-output-path", "className"),
        Output("inference-output-path-feedback", "children"),
        Output("inference-output-path-feedback", "className"),
        Input("inference-output-path", "value"),
    )
    def validate_inference_output_path(path_value: str | None) -> tuple[bool, str, str, str]:
        """Show destination-path feedback and disable Run for an invalid path."""
        is_valid, input_class, feedback, feedback_class = output_directory_feedback(path_value)
        return not is_valid, input_class, feedback, feedback_class

    @app.callback(
        Output("inference-file-summary", "children"),
        Output("lr-preview", "children"),
        Output("lr-preview", "className"),
        Input("inference-upload", "contents"),
        State("inference-upload", "filename"),
    )
    def show_inference_upload(
        contents: str | list[str] | None, filenames: str | list[str] | None
    ) -> tuple[str | list[Component], Component | list[Component], str]:
        """List the uploaded files and preview the first one."""
        if not contents:
            return "No files selected yet.", _empty_preview("Input preview will appear here"), EMPTY_PREVIEW_CLASS

        first = contents[0] if isinstance(contents, list) else contents
        return summarize_uploads((contents, filenames)), html.Img(src=first, className="preview-image"), PREVIEW_CLASS

    @app.callback(
        Output("inference-job", "data"),
        Input(RUN_BUTTON_ID, "n_clicks"),
        Input(CONTINUE_BUTTON_ID, "n_clicks"),
        State("inference-upload", "contents"),
        State("inference-upload", "filename"),
        State("model-path", "value"),
        State("inference-job", "data"),
        background=True,
        interval=PROGRESS_POLL_MS,
        progress=PROGRESS_OUTPUTS,
        running=[
            (Output(RUN_BUTTON_ID, "disabled"), True, False),
            (Output(CONTINUE_BUTTON_ID, "disabled"), True, True),
            (Output("download-inference", "disabled"), True, True),
            (Output("stop-inference", "disabled"), False, True),
        ],
        prevent_initial_call=True,
    )
    def start_or_continue_inference(
        set_progress: SetProgress,
        _run_clicks: int,
        _continue_clicks: int,
        contents: str | list[str] | None,
        filenames: str | list[str] | None,
        model_path: str | None,
        last_job: JobState | None,
    ) -> JobState:
        """Background job behind Run and Continue (told apart by the triggered button)."""
        return run_or_continue(set_progress, cache, contents, filenames, model_path, last_job)

    @app.callback(
        Input("stop-inference", "n_clicks"),
        State("inference-active-job", "data"),
        prevent_initial_call=True,
    )
    def stop_inference(_n_clicks: int, active_job_id: str | None) -> None:
        """Set the stop flag of the running job; it stops after its current image."""
        if active_job_id:
            job.request_stop(cache, active_job_id)

    @app.callback(
        Output("inference-status", "children"),
        Output("inference-status", "className"),
        Output(CONTINUE_BUTTON_ID, "disabled"),
        Output("download-inference", "disabled"),
        Input("inference-job", "data"),
        prevent_initial_call=True,
    )
    def show_job_state(state: JobState | None) -> tuple[str, str, bool, bool]:
        """Show the final status and enable Continue (CANCELLED) and Download (ended with results)."""
        if not state:
            return "Model not loaded", "status-pill status-muted", True, True
        pill_text, pill_class = _status_pill(RunStatus(state["status"]), state["error"])
        can_continue = state["status"] == RunStatus.CANCELLED.value
        can_download = state["status"] in ENDED_STATUSES and state["processed"] > 0
        return pill_text, pill_class, not can_continue, not can_download

    @app.callback(
        Output("inference-download-file", "data"),
        Input("download-inference", "n_clicks"),
        State("inference-job", "data"),
        prevent_initial_call=True,
    )
    def download_results(_n_clicks: int, state: JobState | None) -> dict[str, str | bool | None] | None:
        """Send a zip of the job's *_sr.tif results."""
        if not state:
            return None
        archive = job.zip_results(Path(state["run_dir"]))
        return dcc.send_bytes(archive, RESULTS_ZIP_NAME.format(job_id=state["job_id"]))
