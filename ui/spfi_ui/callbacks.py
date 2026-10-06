from dash import Input, Output, State, ctx, html, no_update


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
        Output("fake-train-timer", "disabled"),
        Output("start-training", "disabled"),
        Output("stop-training", "disabled"),
        Output("train-status", "children"),
        Output("train-status", "className"),
        Output("train-status-copy", "children"),
        Output("stop-training", "className"),
        Output("resume-training", "className"),
        Output("fake-train-timer", "n_intervals"),
        Input("start-training", "n_clicks"),
        Input("stop-training", "n_clicks"),
        Input("resume-training", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_training(start_clicks, stop_clicks, resume_clicks):
        running_buttons = ("danger-action", "resume-action d-none")
        if ctx.triggered_id == "start-training":
            return (False, True, False, "Running", "status-pill status-success", "Training UI demo in progress",
                    *running_buttons, 0)
        if ctx.triggered_id == "resume-training":
            return (False, True, False, "Running", "status-pill status-success", "Training resumed from where it stopped",
                    *running_buttons, no_update)
        return (True, False, True, "Stopped", "status-pill status-warning", "Training stopped by user",
                "danger-action d-none", "resume-action", no_update)

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
