from collections import Counter

from dash import Dash, dcc

from spfi_ui.training.callbacks import _render_metrics, register_training_callbacks
from spfi_ui.training.page import (
    base_model_card, data_card, training_controls, training_metrics, training_page,
)


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


def titles(component):
    return [c.children for c in walk(component) if type(c).__name__ == "H4"]


CONTROL_IDS = ["train-poll-timer", "output-path", "choose-output", "output-path-feedback",
               "run-name", "device-select", "start-training", "stop-training", "resume-training",
               "train-status", "train-progress", "train-status-copy"]
METRIC_IDS = [f"metric-{name}-{part}" for name in ("loss", "val", "iter", "elapsed") for part in ("value", "hint")]
UPLOAD_IDS = ["train-image-upload", "mask-upload", "train-small-image-upload", "train-upload-summary"]
BASE_MODEL_IDS = ["base-model", "choose-base-model", "base-model-hint"]


class TestTrainingControls:
    def test_ids(self):
        assert sorted(ids(training_controls())) == sorted(CONTROL_IDS)

    def test_defaults(self):
        panel = training_controls()
        assert by_id(panel, "run-name").value == "spfi_baseline_01"
        assert by_id(panel, "output-path").value == "runs"
        assert by_id(panel, "device-select").value == "auto"
        assert [o["value"] for o in by_id(panel, "device-select").options] == ["auto", "cuda", "cpu"]
        assert titles(panel) == ["Training controls"]
        assert by_id(panel, "choose-output").children == "Select Routing"
        assert by_id(panel, "output-path-feedback").children is None

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
    def test_ids(self):
        assert sorted(ids(training_metrics())) == sorted(METRIC_IDS)

    def test_initial_values_match_idle_render(self):
        metrics = training_metrics()
        idle = _render_metrics({"state": "idle", "mode": None})
        assert tuple(by_id(metrics, element_id).children for element_id in METRIC_IDS) == idle


class TestCards:
    def test_data_card(self):
        card = data_card()
        assert sorted(ids(card)) == sorted(UPLOAD_IDS)
        assert by_id(card, "train-image-upload").multiple is True
        assert by_id(card, "mask-upload").multiple is True
        assert not by_id(card, "train-small-image-upload").multiple

    def test_base_model_card_is_optional_and_empty(self):
        card = base_model_card()
        assert sorted(ids(card)) == sorted(BASE_MODEL_IDS)
        assert getattr(by_id(card, "base-model"), "value", None) is None
        assert by_id(card, "base-model-hint").children == "Training from scratch"
        kickers = [c.children for c in walk(card) if getattr(c, "className", None) == "card-kicker"]
        assert kickers == ["02 · FINE-TUNING · OPTIONAL"]
        assert titles(card) == ["Fine-tuning: continue from an existing model"]


class TestTrainingPage:
    def test_single_page_without_sub_tabs(self):
        assert not [c for c in walk(training_page()) if type(c).__name__ in ("Tabs", "Tab")]

    def test_contains_every_section(self):
        found = set(ids(training_page()))
        assert set(CONTROL_IDS + METRIC_IDS + UPLOAD_IDS + BASE_MODEL_IDS + ["train-mode-pill"]) <= found

    def test_mode_pill_starts_as_scratch(self):
        assert by_id(training_page(), "train-mode-pill").children == "From scratch"

    def test_ids_are_unique(self):
        duplicates = [i for i, n in Counter(ids(training_page())).items() if n > 1]
        assert duplicates == []

    def test_no_finetune_tab_leftovers(self):
        assert not [i for i in ids(training_page()) if i.startswith("ft-") or "finetune" in i]
        assert "Fine-tuning controls" not in titles(training_page())


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
    assert len(referenced) > 25

    assert referenced - layout_ids == set()
