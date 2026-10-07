import os
import subprocess
import sys

from dash import Input, Output, State, ctx, html, no_update


def _parse_upload(contents, filename):
    if not contents:
        return None
    return html.Div(
        [html.Span("✓", className="file-check"), html.Span(filename or "Uploaded file", className="file-name")],
        className="file-chip",
    )


def _empty_preview(message):
    return [
        html.Div("▧", className="preview-placeholder-icon"),
        html.Div(message, className="preview-placeholder-copy"),
    ]


def _upload_chips(contents, filenames):
    if not contents:
        return []

    if isinstance(contents, str):
        contents = [contents]

    if filenames is None:
        filenames = [None] * len(contents)
    elif isinstance(filenames, str):
        filenames = [filenames]

    return [_parse_upload(content, filename) for content, filename in zip(contents, filenames)]


def _directory_status(path_value):
    if not path_value or not path_value.strip():
        return False, "soft-input is-invalid", "Directory path is required.", "field-feedback field-feedback-invalid"

    normalized_path = os.path.expanduser(path_value.strip())
    if not os.path.isdir(normalized_path):
        return False, "soft-input is-invalid", "Directory does not exist.", "field-feedback field-feedback-invalid"

    return True, "soft-input is-valid", "Directory exists.", "field-feedback field-feedback-valid"


def _mac_choose_folder(prompt_text):
    apple_script = f'POSIX path of (choose folder with prompt "{prompt_text}")'
    result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def _mac_choose_file(prompt_text):
    apple_script = f'POSIX path of (choose file with prompt "{prompt_text}")'
    result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    selected = result.stdout.strip()
    return selected or None


def _choose_folder(prompt_text):
    if sys.platform == "darwin":
        return _mac_choose_folder(prompt_text)
    return None


def _choose_file(prompt_text):
    if sys.platform == "darwin":
        return _mac_choose_file(prompt_text)
    return None


def register_callbacks(app):
    @app.callback(
        Output("app-shell", "className"),
        Input("theme-toggle", "value"),
    )
    def toggle_theme(theme_value):
        if theme_value and "light" in theme_value:
            return "app-shell theme-light"
        return "app-shell theme-dark"

    @app.callback(
        Output("output-path", "value"),
        Input("choose-output", "n_clicks"),
        State("output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_training_output_directory(n_clicks, current_value):
        selected = _choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("inference-output-path", "value"),
        Input("choose-inference-output", "n_clicks"),
        State("inference-output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_inference_output_directory(n_clicks, current_value):
        selected = _choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("model-path", "value"),
        Input("choose-model", "n_clicks"),
        State("model-path", "value"),
        prevent_initial_call=True,
    )
    def choose_model_file(n_clicks, current_value):
        selected = _choose_file("Select model file")
        return selected if selected else current_value

    @app.callback(
        Output("train-upload-summary", "children"),
        Input("train-image-upload", "contents"),
        Input("mask-upload", "contents"),
        Input("train-small-image-upload", "contents"),
        State("train-image-upload", "filename"),
        State("mask-upload", "filename"),
        State("train-small-image-upload", "filename"),
    )
    def summarize_training_uploads(image_contents, mask_contents, small_contents, image_name, mask_name, small_name):
        items = []
        if image_contents:
            items.append(_parse_upload(image_contents, image_name))
        if mask_contents:
            items.append(_parse_upload(mask_contents, mask_name))
        if small_contents:
            items.append(_parse_upload(small_contents, small_name))
        return items or "No files selected yet."

    @app.callback(
        Output("finetune-upload-summary", "children"),
        Input("finetune-data-upload", "contents"),
        Input("finetune-small-image-upload", "contents"),
        State("finetune-data-upload", "filename"),
        State("finetune-small-image-upload", "filename"),
    )
    def summarize_finetune_uploads(data_contents, small_contents, data_filenames, small_filename):
        items = _upload_chips(data_contents, data_filenames)
        if small_contents:
            items.append(_parse_upload(small_contents, small_filename))
        return items or "No files selected yet."

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

        chips = _upload_chips(contents, filenames)
        if isinstance(contents, str):
            contents = [contents]
        return chips, html.Img(src=contents[0], className="preview-image"), "image-preview"

    @app.callback(
        Output("start-training", "disabled"),
        Output("output-path", "className"),
        Output("output-path-feedback", "children"),
        Output("output-path-feedback", "className"),
        Input("output-path", "value"),
        Input("fake-train-timer", "disabled"),
    )
    def validate_training_output_path(path_value, timer_disabled):
        is_valid, input_class_name, feedback, feedback_class_name = _directory_status(path_value)
        return (not is_valid) or (not timer_disabled), input_class_name, feedback, feedback_class_name

    @app.callback(
        Output("run-inference", "disabled"),
        Output("inference-output-path", "className"),
        Output("inference-output-path-feedback", "children"),
        Output("inference-output-path-feedback", "className"),
        Input("inference-output-path", "value"),
    )
    def validate_inference_output_path(path_value):
        is_valid, input_class_name, feedback, feedback_class_name = _directory_status(path_value)
        return not is_valid, input_class_name, feedback, feedback_class_name

    @app.callback(
        Output("fake-train-timer", "disabled"),
        Output("stop-training", "disabled"),
        Output("train-status", "children"),
        Output("train-status", "className"),
        Output("train-status-copy", "children"),
        Input("start-training", "n_clicks"),
        Input("stop-training", "n_clicks"),
        State("output-path", "value"),
        prevent_initial_call=True,
    )
    def toggle_training(start_clicks, stop_clicks, output_path):
        is_valid, _, _, _ = _directory_status(output_path)
        if ctx.triggered_id == "start-training" and not is_valid:
            return True, True, "Invalid path", "status-pill status-warning", "Choose an existing output directory before starting."
        if ctx.triggered_id == "start-training":
            return False, False, "Running", "status-pill status-success", "Training UI demo in progress"
        return True, True, "Stopped", "status-pill status-warning", "Training stopped by user"

    @app.callback(
        Output("train-progress", "value"),
        Input("fake-train-timer", "n_intervals"),
        State("fake-train-timer", "disabled"),
    )
    def update_fake_progress(n_intervals, disabled):
        if disabled:
            return no_update
        return min((n_intervals * 7) % 101, 100)

    @app.callback(
        Output("inference-status", "children"),
        Output("inference-status", "className"),
        Output("hr-preview", "children"),
        Output("hr-preview", "className"),
        Input("run-inference", "n_clicks"),
        State("inference-upload", "contents"),
        State("inference-output-path", "value"),
        prevent_initial_call=True,
    )
    def demo_inference(n_clicks, contents, output_path):
        is_valid, _, _, _ = _directory_status(output_path)
        if not is_valid:
            return "Invalid output path", "status-pill status-warning", no_update, no_update
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
