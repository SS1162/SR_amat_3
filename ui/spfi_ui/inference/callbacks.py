from dash import Input, Output, State, html, no_update

from ..callbacks import choose_file, choose_folder, output_directory_feedback
from ..components import summarize_uploads


def _empty_preview(message):
    return [
        html.Div("▧", className="preview-placeholder-icon"),
        html.Div(message, className="preview-placeholder-copy"),
    ]


def register_inference_callbacks(app):
    @app.callback(
        Output("model-path", "value"),
        Input("choose-model", "n_clicks"),
        State("model-path", "value"),
        prevent_initial_call=True,
    )
    def choose_model_file(n_clicks, current_value):
        selected = choose_file("Select model file")
        return selected if selected else current_value

    @app.callback(
        Output("inference-output-path", "value"),
        Input("choose-inference-output", "n_clicks"),
        State("inference-output-path", "value"),
        prevent_initial_call=True,
    )
    def choose_inference_output_directory(n_clicks, current_value):
        selected = choose_folder("Select Routing folder")
        return selected if selected else current_value

    @app.callback(
        Output("run-inference", "disabled"),
        Output("inference-output-path", "className"),
        Output("inference-output-path-feedback", "children"),
        Output("inference-output-path-feedback", "className"),
        Input("inference-output-path", "value"),
    )
    def validate_inference_output_path(path_value):
        is_valid, input_class, feedback, feedback_class = output_directory_feedback(path_value)
        return not is_valid, input_class, feedback, feedback_class

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

        first = contents[0] if isinstance(contents, list) else contents
        return summarize_uploads((contents, filenames)), html.Img(src=first, className="preview-image"), "image-preview"

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
        is_valid, _, _, _ = output_directory_feedback(output_path)
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
