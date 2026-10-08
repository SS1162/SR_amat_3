from collections import Counter

import pytest
from dash import Dash, dcc

from spfi_ui.training.callbacks import register_training_callbacks
from spfi_ui.training.page import finetune_tab, scratch_tab, training_controls, training_metrics, training_page


def walk(component):
    """Yields every Dash component in the tree rooted at ``component``."""
    if component is None or isinstance(component, (str, int, float)):
        return
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from walk(child)
        return
    yield component
    yield from walk(getattr(component, "children", None))


def ids(component):
    return [c.id for c in walk(component) if getattr(c, "id", None) is not None]


def by_id(component, element_id):
    matches = [c for c in walk(component) if getattr(c, "id", None) == element_id]
    assert len(matches) == 1, element_id
    return matches[0]


CONTROL_IDS = ["train-poll-timer", "output-path", "choose-output", "output-path-feedback", "run-name", "device-select",
               "start-training", "stop-training", "resume-training", "train-status", "train-progress",
               "train-status-copy"]
METRIC_IDS = [f"metric-{name}-{part}" for name in ("loss", "val", "iter", "elapsed") for part in ("value", "hint")]


class TestTrainingControls:
    @pytest.mark.parametrize("prefix", ["", "ft-", "x-"])
    def test_ids_are_prefixed(self, prefix):
        assert sorted(ids(training_controls(prefix))) == sorted(prefix + i for i in CONTROL_IDS)

    def test_defaults(self):
        panel = training_controls()
        assert by_id(panel, "run-name").value == "spfi_baseline_01"
        assert by_id(panel, "output-path").value == "runs"
        assert by_id(panel, "device-select").value == "auto"
        assert [o["value"] for o in by_id(panel, "device-select").options] == ["auto", "cuda", "cpu"]
        assert "Training controls" in [c.children for c in walk(panel) if type(c).__name__ == "H4"]
        assert by_id(panel, "choose-output").children == "Select Routing"
        assert by_id(panel, "output-path-feedback").children is None

    def test_custom_run_name_and_title(self):
        panel = training_controls("ft-", "my_run", "Fine-tuning controls")
        assert by_id(panel, "ft-run-name").value == "my_run"
        assert "Fine-tuning controls" in [c.children for c in walk(panel) if type(c).__name__ == "H4"]

    def test_initial_button_and_timer_state(self):
        panel = training_controls()
        timer = by_id(panel, "train-poll-timer")
        assert isinstance(timer, dcc.Interval)
        assert timer.disabled is True and timer.interval == 1000
        assert by_id(panel, "stop-training").disabled is True
        assert "d-none" in by_id(panel, "resume-training").className
        assert by_id(panel, "train-progress").value == 0
        assert by_id(panel, "train-status").children == "Idle"
        assert by_id(panel, "train-status-copy").children == "No active training run"


class TestTrainingMetrics:
    @pytest.mark.parametrize("prefix", ["", "ft-"])
    def test_ids(self, prefix):
        assert sorted(ids(training_metrics(prefix))) == sorted(prefix + i for i in METRIC_IDS)

    def test_initial_values_match_idle_render(self):
        from spfi_ui.training.callbacks import _render_metrics
        metrics = training_metrics()
        idle = _render_metrics({"state": "idle", "mode": None}, "scratch")
        shown = tuple(by_id(metrics, element_id).children for element_id in METRIC_IDS)
        assert shown == idle


class TestTabs:
    def test_scratch_tab_has_uploads_controls_and_metrics(self):
        found = set(ids(scratch_tab()))
        assert {"train-image-upload", "mask-upload", "train-small-image-upload", "train-upload-summary"} <= found
        assert set(CONTROL_IDS) <= found and set(METRIC_IDS) <= found

    def test_finetune_tab_has_weights_uploads_and_prefixed_ids(self):
        tab = finetune_tab()
        found = set(ids(tab))
        assert {"ft-weights", "choose-finetune-model", "finetune-data-upload", "finetune-mask-upload",
                "finetune-small-image-upload", "finetune-upload-summary"} <= found
        assert {f"ft-{i}" for i in CONTROL_IDS + METRIC_IDS} <= found
        assert by_id(tab, "ft-run-name").value == "spfi_finetune_01"

    def test_uploads_accept_multiple_files(self):
        page = training_page()
        for upload_id in ("train-image-upload", "mask-upload", "finetune-data-upload", "finetune-mask-upload"):
            assert by_id(page, upload_id).multiple is True
        for upload_id in ("train-small-image-upload", "finetune-small-image-upload"):
            assert not by_id(page, upload_id).multiple

    def test_training_page_ids_are_unique(self):
        duplicates = [i for i, n in Counter(ids(training_page())).items() if n > 1]
        assert duplicates == []

    def test_training_page_tabs(self):
        tabs = training_page().children[0]
        assert tabs.active_tab == "scratch"
        assert [t.tab_id for t in tabs.children] == ["scratch", "finetune"]


def test_every_callback_id_exists_in_layout():
    """Catches renamed ids: each Output/Input/State used by the training callbacks must be on the page."""
    app = Dash(__name__, suppress_callback_exceptions=True)
    register_training_callbacks(app)
    layout_ids = set(ids(training_page()))

    referenced = set()
    for output_key, callback in app.callback_map.items():
        referenced.update(dep["id"] for dep in callback["inputs"] + callback["state"])
        # multi-output keys look like "..a.prop...b.prop.."
        referenced.update(out.rsplit(".", 1)[0] for out in output_key.strip(".").split("..."))
    assert len(referenced) > 40

    assert referenced - layout_ids == set()
