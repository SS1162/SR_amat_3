from pathlib import Path
from types import SimpleNamespace

import pytest
from dash import Dash

from spfi_ui.training import callbacks
from spfi_ui.training.callbacks import (
    _describe_base_model, _format_elapsed, _render_metrics, _render_training, _training_mode,
    register_training_callbacks,
)


def snap(**overrides):
    base = {"mode": None, "val": None, "elapsed": 0.0, "state": "idle", "error": None,
            "iteration": 0, "loss": None, "ckpt_every": None, "run_dir": None}
    base.update(overrides)
    return base


STOP_SHOWN, RESUME_HIDDEN = "danger-action", "resume-action d-none"


# ------------------------------------------------------------ _training_mode

class TestTrainingMode:
    @pytest.mark.parametrize("base_model", [None, "", "   "])
    def test_empty_base_model_trains_from_scratch(self, base_model):
        assert _training_mode(base_model) == ("scratch", None)

    def test_base_model_makes_a_finetune(self):
        assert _training_mode("runs/a/best.pt") == ("finetune", "runs/a/best.pt")

    def test_base_model_is_stripped(self):
        assert _training_mode("  runs/a/best.pt \n") == ("finetune", "runs/a/best.pt")


class TestDescribeBaseModel:
    def test_scratch(self):
        assert _describe_base_model(None) == ("From scratch", "Training from scratch")
        assert _describe_base_model("  ") == ("From scratch", "Training from scratch")

    def test_finetune_shows_file_name(self):
        assert _describe_base_model("runs/a/checkpoints/spfi_x4_best.pt") == (
            "Fine-tuning", "Fine-tuning from spfi_x4_best.pt")


# ------------------------------------------------------------ _render_training

class TestRenderTraining:
    def test_idle(self):
        assert _render_training(snap()) == (
            True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Idle", "status-pill status-muted",
            "No active training run", 0)

    def test_unknown_state_renders_idle(self):
        assert _render_training(snap(state="weird", mode="scratch"))[5] == "Idle"

    def test_preparing(self):
        out = _render_training(snap(state="preparing", mode="scratch"))
        assert out == (False, True, False, STOP_SHOWN, RESUME_HIDDEN, "Preparing", "status-pill status-warning",
                       "From scratch · Saving data and building the model…", 0)

    def test_running_with_loss_and_progress(self):
        out = _render_training(snap(state="running", mode="scratch", iteration=260, loss=0.12345, ckpt_every=250))
        assert out[:5] == (False, True, False, STOP_SHOWN, RESUME_HIDDEN)
        assert out[5:7] == ("Running", "status-pill status-success")
        assert out[7] == "From scratch · Iteration 260 · loss 0.1235 · bar = progress to next checkpoint"
        assert out[8] == pytest.approx(4.0)

    def test_running_finetune_is_labelled(self):
        out = _render_training(snap(state="running", mode="finetune", iteration=3))
        assert out[7] == "Fine-tuning · Iteration 3 · bar = progress to next checkpoint"
        assert out[8] == 0

    def test_unknown_mode_has_no_label(self):
        out = _render_training(snap(state="running", mode=None, iteration=3))
        assert out[7] == "Iteration 3 · bar = progress to next checkpoint"

    def test_stopping(self):
        out = _render_training(snap(state="stopping", mode="scratch", iteration=125, ckpt_every=250))
        assert out == (False, True, True, STOP_SHOWN, RESUME_HIDDEN, "Stopping", "status-pill status-warning",
                       "Finishing the current iteration…", 50.0)

    def test_stopped_shows_resume_and_hides_stop(self):
        out = _render_training(snap(state="stopped", mode="finetune", iteration=10, loss=1.0, ckpt_every=250))
        assert out == (True, False, True, "danger-action d-none", "resume-action", "Stopped",
                       "status-pill status-warning", "Fine-tuning · Stopped at iteration 10 · loss 1.0000", 4.0)

    def test_finished(self):
        out = _render_training(snap(state="finished", mode="finetune", iteration=900, run_dir="runs/x"))
        assert out == (True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Finished", "status-pill status-success",
                       "Fine-tuning · Early stopping at iteration 900 · checkpoints in runs/x", 100)

    def test_error(self):
        out = _render_training(snap(state="error", mode="scratch", error="RuntimeError: boom"))
        assert out == (True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Error", "status-pill status-danger",
                       "RuntimeError: boom", 0)

    def test_error_argument_overrides_status_only(self):
        out = _render_training(snap(state="running", mode="scratch", iteration=5, ckpt_every=10),
                               error="Upload at least 2 training images")
        assert out[:5] == (False, True, False, STOP_SHOWN, RESUME_HIDDEN)
        assert out[5:] == ("Input required", "status-pill status-danger", "Upload at least 2 training images", 50.0)


# ------------------------------------------------------------ _format_elapsed

@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00"),
    (59.9, "00:59"),
    (61, "01:01"),
    (3599, "59:59"),
    (3600, "1:00:00"),
    (3725, "1:02:05"),
    (36000 + 61, "10:01:01"),
])
def test_format_elapsed(seconds, expected):
    assert _format_elapsed(seconds) == expected


# ------------------------------------------------------------ _render_metrics

IDLE_METRICS = ("—", "Available when a run begins", "—", "Available after the first checkpoint",
                "0", "No active training run", "00:00", "Training time, excluding pauses")


class TestRenderMetrics:
    def test_idle(self):
        assert _render_metrics(snap()) == IDLE_METRICS

    def test_preparing_without_trainer_data(self):
        out = _render_metrics(snap(state="preparing", mode="scratch"))
        assert out == ("—", "Latest training iteration", "—", "Available after the first checkpoint",
                       "0", "Preparing…", "00:00", "Training time, excluding pauses")

    @pytest.mark.parametrize("mode", ["scratch", "finetune"])
    def test_running_with_loss_and_validation(self, mode):
        out = _render_metrics(snap(state="running", mode=mode, iteration=260, ckpt_every=250,
                                   loss=0.123456, val=(0.0456, 28.123), elapsed=75))
        assert out == ("0.1235", "Latest training iteration",
                       "28.12 dB", "val loss 0.0456 · higher PSNR is better",
                       "260", "Next checkpoint at 500", "01:15", "Training time, excluding pauses")

    def test_next_checkpoint_on_exact_boundary(self):
        out = _render_metrics(snap(state="running", mode="finetune", iteration=250, ckpt_every=250))
        assert out[5] == "Next checkpoint at 500"

    def test_iteration_zero(self):
        out = _render_metrics(snap(state="running", mode="scratch", iteration=0, ckpt_every=250))
        assert out[4:6] == ("0", "Next checkpoint at 250")


# ------------------------------------------------------------ registered callbacks

class FakeApp:
    """Captures callbacks registered with ``@app.callback(...)``, keyed by function name."""

    def __init__(self):
        self.callbacks = {}

    def callback(self, *dependencies, **_kwargs):
        def decorator(fn):
            self.callbacks[fn.__name__] = (dependencies, fn)
            return fn
        return decorator

    def get(self, name):
        return self.callbacks[name][1]


class FakeRun:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.state = snap()

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.error:
            raise self.error

    def start(self, *args, **kwargs):
        self._call("start", *args, **kwargs)

    def stop(self):
        self._call("stop")

    def resume(self):
        self._call("resume")

    def snapshot(self):
        return self.state


@pytest.fixture
def app():
    fake = FakeApp()
    register_training_callbacks(fake)
    return fake


@pytest.fixture
def fake_run(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(callbacks, "RUN", run)
    return run


def trigger(monkeypatch, component_id):
    monkeypatch.setattr(callbacks, "ctx", SimpleNamespace(triggered_id=component_id))


UPLOAD_ARGS = (["img1", "img2"], ["a.png", "b.png"], ["m"], ["a.png"])
CLICKS = (1, 0, 0, 0)


def control(app, output_path="runs", run_name="exp1", device="auto", base_model=None):
    return app.get("control_training")(*CLICKS, *UPLOAD_ARGS, output_path, run_name, device, base_model)


class TestRegistration:
    def test_registers_all_expected_callbacks(self, app):
        assert set(app.callbacks) == {
            "choose_output_directory", "choose_base_model_file", "describe_base_model",
            "validate_output_path", "summarize_training_uploads", "control_training",
            "update_training_metrics",
        }

    def test_control_training_reads_base_model(self, app):
        deps, _ = app.callbacks["control_training"]
        assert deps[-1].component_id == "base-model"

    def test_registers_on_real_dash_app(self):
        dash_app = Dash(__name__, suppress_callback_exceptions=True)
        register_training_callbacks(dash_app)
        assert len(dash_app.callback_map) == 7

    def test_no_finetune_tab_ids_left(self):
        dash_app = Dash(__name__, suppress_callback_exceptions=True)
        register_training_callbacks(dash_app)
        assert not any("ft-" in key or "finetune" in key for key in dash_app.callback_map)


class TestUploadSummary:
    def test_empty(self, app):
        assert app.get("summarize_training_uploads")(None, None, None, None, None, None) == "No files selected yet."

    def test_images_and_masks(self, app):
        chips = app.get("summarize_training_uploads")(["x", "y"], "m", None, ["a.png", "b.png"], "a.png", None)
        assert len(chips) == 3

    def test_includes_small_image(self, app):
        chips = app.get("summarize_training_uploads")(["x"], None, "s", ["a.png"], None, "small.png")
        assert len(chips) == 2


class TestChoosePaths:
    def test_choose_folder_replaces_value(self, app, monkeypatch):
        prompts = []
        monkeypatch.setattr(callbacks, "choose_folder", lambda prompt: prompts.append(prompt) or "/data/runs")
        assert app.get("choose_output_directory")(1, "runs") == "/data/runs"
        assert prompts == ["Select Routing folder"]

    def test_cancelled_folder_keeps_value(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "choose_folder", lambda prompt: None)
        assert app.get("choose_output_directory")(1, "runs") == "runs"

    def test_choose_base_model(self, app, monkeypatch):
        prompts = []
        monkeypatch.setattr(callbacks, "choose_file", lambda prompt: prompts.append(prompt) or "/m/best.pt")
        assert app.get("choose_base_model_file")(1, None) == "/m/best.pt"
        assert prompts == ["Select base model file"]

    def test_cancelled_base_model_keeps_value(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "choose_file", lambda prompt: None)
        assert app.get("choose_base_model_file")(1, "old.pt") == "old.pt"

    def test_choosers_dont_fire_on_page_load(self):
        dash_app = Dash(__name__, suppress_callback_exceptions=True)
        register_training_callbacks(dash_app)
        prevent = {cb["output"]: cb.get("prevent_initial_call") for cb in dash_app._callback_list}
        assert prevent["output-path.value"] is True
        assert prevent["base-model.value"] is True


class TestDescribeBaseModelCallback:
    def test_updates_pill_and_hint(self, app):
        fn = app.get("describe_base_model")
        assert fn(None) == ("From scratch", "Training from scratch")
        assert fn("/m/best.pt") == ("Fine-tuning", "Fine-tuning from best.pt")


VALID, INVALID = "field-feedback field-feedback-valid", "field-feedback field-feedback-invalid"


class TestValidateOutputPath:
    def test_existing_directory_is_valid(self, app, tmp_path):
        assert app.get("validate_output_path")(str(tmp_path)) == ("soft-input is-valid", "Directory exists.", VALID)

    def test_new_directory_is_valid(self, app, tmp_path):
        assert app.get("validate_output_path")(str(tmp_path / "new")) == (
            "soft-input is-valid", "Directory will be created when training starts.", VALID)

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_empty_path_is_invalid(self, app, value):
        assert app.get("validate_output_path")(value) == (
            "soft-input is-invalid", "Directory path is required.", INVALID)

    def test_file_path_is_invalid(self, app, tmp_path):
        file_path = tmp_path / "f.txt"
        file_path.write_text("x")
        assert app.get("validate_output_path")(str(file_path)) == (
            "soft-input is-invalid", "Path points to a file, not a directory.", INVALID)

    def test_missing_parent_is_invalid(self, app, tmp_path):
        assert app.get("validate_output_path")(str(tmp_path / "a" / "b")) == (
            "soft-input is-invalid", "Parent directory does not exist.", INVALID)


class TestControlTraining:
    def test_start_without_base_model_trains_from_scratch(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "start-training")
        out = control(app, device="cuda")
        assert fake_run.calls == [("start", (*UPLOAD_ARGS, Path("runs") / "exp1", "cuda"),
                                   {"mode": "scratch", "weights": None})]
        assert out == _render_training(fake_run.state)

    @pytest.mark.parametrize("base_model", ["", "   "])
    def test_blank_base_model_trains_from_scratch(self, app, fake_run, monkeypatch, base_model):
        trigger(monkeypatch, "start-training")
        control(app, base_model=base_model)
        assert fake_run.calls[0][2] == {"mode": "scratch", "weights": None}

    def test_start_with_base_model_fine_tunes(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "start-training")
        control(app, run_name="ft1", device="cpu", base_model=" w.pt ")
        name, args, kwargs = fake_run.calls[0]
        assert name == "start"
        assert args[4] == Path("runs") / "ft1"
        assert kwargs == {"mode": "finetune", "weights": "w.pt"}

    def test_start_auto_device_becomes_none(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "start-training")
        control(app, device="auto")
        assert fake_run.calls[0][1][5] is None

    @pytest.mark.parametrize("output_path, run_name", [("", "x"), ("runs", ""), (None, "x"), ("runs", None)])
    def test_start_requires_output_and_name(self, app, fake_run, monkeypatch, output_path, run_name):
        trigger(monkeypatch, "start-training")
        out = control(app, output_path=output_path, run_name=run_name)
        assert fake_run.calls == []
        assert out[5:8] == ("Input required", "status-pill status-danger",
                            "Output directory and run name are required")

    def test_run_value_error_is_shown(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "RUN", FakeRun(error=ValueError("Checkpoint not found: w.pt")))
        trigger(monkeypatch, "start-training")
        out = control(app, base_model="w.pt")
        assert out[5:8] == ("Input required", "status-pill status-danger", "Checkpoint not found: w.pt")

    def test_non_value_error_propagates(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "RUN", FakeRun(error=RuntimeError("bug")))
        trigger(monkeypatch, "stop-training")
        with pytest.raises(RuntimeError):
            control(app)

    def test_stop(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "stop-training")
        control(app)
        assert fake_run.calls == [("stop", (), {})]

    def test_resume(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "resume-training")
        control(app)
        assert fake_run.calls == [("resume", (), {})]

    def test_resume_error_is_shown(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "RUN", FakeRun(error=ValueError("There is no stopped run to resume")))
        trigger(monkeypatch, "resume-training")
        assert control(app)[7] == "There is no stopped run to resume"

    @pytest.mark.parametrize("triggered", ["train-poll-timer", None])
    def test_poll_only_renders(self, app, fake_run, monkeypatch, triggered):
        trigger(monkeypatch, triggered)
        fake_run.state = snap(state="running", mode="finetune", iteration=5, ckpt_every=10)
        out = control(app)
        assert fake_run.calls == []
        assert out == _render_training(fake_run.state)


class TestUpdateMetrics:
    @pytest.mark.parametrize("mode", ["scratch", "finetune"])
    def test_renders_current_run(self, app, fake_run, mode):
        fake_run.state = snap(state="running", mode=mode, iteration=3, ckpt_every=10, loss=0.5)
        out = app.get("update_training_metrics")(1, False)
        assert out == _render_metrics(fake_run.state)
        assert out[0] == "0.5000"

    def test_idle(self, app, fake_run):
        assert app.get("update_training_metrics")(0, True) == IDLE_METRICS
