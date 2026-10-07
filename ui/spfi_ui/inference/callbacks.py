from dash import Input, Output, State, html

from ..components import summarize_uploads


def _empty_preview(message):
    return [
        html.Div("▧", className="preview-placeholder-icon"),
        html.Div(message, className="preview-placeholder-copy"),
    ]


def register_inference_callbacks(app):
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
