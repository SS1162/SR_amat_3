from pathlib import Path
from types import SimpleNamespace

import pytest
from dash import Dash

from spfi_ui.training import callbacks
from spfi_ui.training.callbacks import (
    ACTIVE_STATES, _format_elapsed, _render_metrics, _render_training, register_training_callbacks,
)


def snap(**overrides):
    base = {"mode": None, "val": None, "elapsed": 0.0, "state": "idle", "error": None,
            "iteration": 0, "loss": None, "ckpt_every": None, "run_dir": None}
    base.update(overrides)
    return base


STOP_SHOWN, RESUME_HIDDEN = "danger-action", "resume-action d-none"


# ------------------------------------------------------------ _render_training

class TestRenderTraining:
    def test_idle(self):
        assert _render_training(snap(), "scratch") == (
            True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Idle", "status-pill status-muted",
            "No active training run", 0)

    def test_unknown_state_renders_idle(self):
        assert _render_training(snap(state="weird", mode="scratch"), "scratch")[5] == "Idle"

    def test_preparing(self):
        out = _render_training(snap(state="preparing", mode="scratch"), "scratch")
        assert out == (False, True, False, STOP_SHOWN, RESUME_HIDDEN, "Preparing", "status-pill status-warning",
                       "Saving data and building the model…", 0)

    def test_running_with_loss_and_progress(self):
        out = _render_training(snap(state="running", mode="scratch", iteration=260, loss=0.12345,
                                    ckpt_every=250), "scratch")
        assert out[:5] == (False, True, False, STOP_SHOWN, RESUME_HIDDEN)
        assert out[5:7] == ("Running", "status-pill status-success")
        assert out[7] == "Iteration 260 · loss 0.1235 · bar = progress to next checkpoint"
        assert out[8] == pytest.approx(4.0)

    def test_running_without_loss_or_ckpt_every(self):
        out = _render_training(snap(state="running", mode="scratch", iteration=3), "scratch")
        assert out[7] == "Iteration 3 · bar = progress to next checkpoint"
        assert out[8] == 0

    def test_stopping(self):
        out = _render_training(snap(state="stopping", mode="scratch", iteration=125, ckpt_every=250), "scratch")
        assert out == (False, True, True, STOP_SHOWN, RESUME_HIDDEN, "Stopping", "status-pill status-warning",
                       "Finishing the current iteration…", 50.0)

    def test_stopped_shows_resume_and_hides_stop(self):
        out = _render_training(snap(state="stopped", mode="scratch", iteration=10, loss=1.0, ckpt_every=250),
                               "scratch")
        assert out == (True, False, True, "danger-action d-none", "resume-action", "Stopped",
                       "status-pill status-warning", "Stopped at iteration 10 · loss 1.0000", 4.0)

    def test_finished(self):
        out = _render_training(snap(state="finished", mode="finetune", iteration=900, run_dir="runs/x"),
                               "finetune")
        assert out == (True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Finished", "status-pill status-success",
                       "Early stopping at iteration 900 · checkpoints in runs/x", 100)

    def test_error(self):
        out = _render_training(snap(state="error", mode="scratch", error="RuntimeError: boom"), "scratch")
        assert out == (True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Error", "status-pill status-danger",
                       "RuntimeError: boom", 0)

    @pytest.mark.parametrize("state", ACTIVE_STATES)
    @pytest.mark.parametrize("owner, panel, label", [
        ("finetune", "scratch", "fine-tuning"),
        ("scratch", "finetune", "from-scratch"),
    ])
    def test_other_panel_active_is_busy(self, state, owner, panel, label):
        out = _render_training(snap(state=state, mode=owner, iteration=5, ckpt_every=10), panel)
        assert out == (False, True, True, STOP_SHOWN, RESUME_HIDDEN, "Busy", "status-pill status-muted",
                       f"A {label} run is active; stop it to start here", 0)

    @pytest.mark.parametrize("state", ["stopped", "finished", "error", "idle"])
    def test_other_panel_inactive_is_idle(self, state):
        out = _render_training(snap(state=state, mode="finetune", error="x"), "scratch")
        assert out == (True, False, True, STOP_SHOWN, RESUME_HIDDEN, "Idle", "status-pill status-muted",
                       "No active training run", 0)

    def test_error_argument_overrides_status_only(self):
        out = _render_training(snap(state="running", mode="scratch", iteration=5, ckpt_every=10), "scratch",
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
        assert _render_metrics(snap(), "scratch") == IDLE_METRICS

    def test_other_mode_shows_idle(self):
        assert _render_metrics(snap(state="running", mode="finetune", iteration=50), "scratch") == IDLE_METRICS

    def test_preparing_without_trainer_data(self):
        out = _render_metrics(snap(state="preparing", mode="scratch"), "scratch")
        assert out == ("—", "Latest training iteration", "—", "Available after the first checkpoint",
                       "0", "Preparing…", "00:00", "Training time, excluding pauses")

    def test_running_with_loss_and_validation(self):
        out = _render_metrics(snap(state="running", mode="scratch", iteration=260, ckpt_every=250,
                                   loss=0.123456, val=(0.0456, 28.123), elapsed=75), "scratch")
        assert out == ("0.1235", "Latest training iteration",
                       "28.12 dB", "val loss 0.0456 · higher PSNR is better",
                       "260", "Next checkpoint at 500", "01:15", "Training time, excluding pauses")

    def test_next_checkpoint_on_exact_boundary(self):
        out = _render_metrics(snap(state="running", mode="finetune", iteration=250, ckpt_every=250), "finetune")
        assert out[5] == "Next checkpoint at 500"

    def test_iteration_zero(self):
        out = _render_metrics(snap(state="running", mode="scratch", iteration=0, ckpt_every=250), "scratch")
        assert out[4:6] == ("0", "Next checkpoint at 250")


# ------------------------------------------------------------ registered callbacks

class FakeApp:
    """Captures callbacks registered with ``@app.callback(...)``."""

    def __init__(self):
        self.callbacks = {}

    def callback(self, *dependencies):
        def decorator(fn):
            first_output = dependencies[0]
            self.callbacks[(first_output.component_id, fn.__name__)] = (dependencies, fn)
            return fn
        return decorator

    def get(self, output_id, name):
        return self.callbacks[(output_id, name)][1]


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


CONTROL_ARGS = (1, 0, 0, 0, ["img1", "img2"], ["a.png", "b.png"], ["m"], ["a.png"])


class TestRegistration:
    def test_registers_all_expected_callbacks(self, app):
        assert set(app.callbacks) == {
            ("train-upload-summary", "summarize_training_uploads"),
            ("finetune-upload-summary", "summarize_finetune_uploads"),
            ("train-poll-timer", "control_training"),
            ("ft-train-poll-timer", "control_training"),
            ("metric-loss-value", "update_training_metrics"),
            ("ft-metric-loss-value", "update_training_metrics"),
        }

    def test_finetune_panel_reads_weights_state(self, app):
        deps, _ = app.callbacks[("ft-train-poll-timer", "control_training")]
        assert deps[-1].component_id == "ft-weights"
        scratch_deps, _ = app.callbacks[("train-poll-timer", "control_training")]
        assert all(dep.component_id != "ft-weights" for dep in scratch_deps)

    def test_registers_on_real_dash_app(self):
        dash_app = Dash(__name__, suppress_callback_exceptions=True)
        register_training_callbacks(dash_app)
        assert len(dash_app.callback_map) == 6


class TestUploadSummaries:
    def test_training_summary(self, app):
        fn = app.get("train-upload-summary", "summarize_training_uploads")
        assert fn(None, None, None, None) == "No files selected yet."
        chips = fn(["x", "y"], "m", ["a.png", "b.png"], "a.png")
        assert len(chips) == 3

    def test_finetune_summary(self, app):
        fn = app.get("finetune-upload-summary", "summarize_finetune_uploads")
        assert fn(None, None, None, None) == "No files selected yet."
        assert len(fn(["x"], None, ["a.png"], None)) == 1


class TestControlTraining:
    def test_start_scratch(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "start-training")
        fn = app.get("train-poll-timer", "control_training")

        out = fn(*CONTROL_ARGS, "runs", "exp1", "cuda")

        assert fake_run.calls == [("start", (["img1", "img2"], ["a.png", "b.png"], ["m"], ["a.png"],
                                             Path("runs") / "exp1", "cuda"),
                                   {"mode": "scratch", "weights": None})]
        assert out == _render_training(fake_run.state, "scratch")

    def test_start_auto_device_becomes_none(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "start-training")
        app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "exp1", "auto")
        assert fake_run.calls[0][1][5] is None

    def test_start_finetune_passes_weights(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "ft-start-training")
        fn = app.get("ft-train-poll-timer", "control_training")
        fn(*CONTROL_ARGS, "runs", "ft1", "cpu", "w.pt")
        name, args, kwargs = fake_run.calls[0]
        assert name == "start"
        assert args[4] == Path("runs") / "ft1"
        assert kwargs == {"mode": "finetune", "weights": "w.pt"}

    @pytest.mark.parametrize("output_path, run_name", [("", "x"), ("runs", ""), (None, "x"), ("runs", None)])
    def test_start_requires_output_and_name(self, app, fake_run, monkeypatch, output_path, run_name):
        trigger(monkeypatch, "start-training")
        out = app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, output_path, run_name, "auto")
        assert fake_run.calls == []
        assert out[5:8] == ("Input required", "status-pill status-danger",
                            "Output directory and run name are required")

    def test_run_value_error_is_shown(self, app, monkeypatch):
        run = FakeRun(error=ValueError("A training run is already active"))
        monkeypatch.setattr(callbacks, "RUN", run)
        trigger(monkeypatch, "start-training")
        out = app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto")
        assert out[5:8] == ("Input required", "status-pill status-danger", "A training run is already active")

    def test_non_value_error_propagates(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "RUN", FakeRun(error=RuntimeError("bug")))
        trigger(monkeypatch, "stop-training")
        with pytest.raises(RuntimeError):
            app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto")

    def test_stop(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "stop-training")
        app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto")
        assert fake_run.calls == [("stop", (), {})]

    def test_resume(self, app, fake_run, monkeypatch):
        trigger(monkeypatch, "ft-resume-training")
        app.get("ft-train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto", None)
        assert fake_run.calls == [("resume", (), {})]

    def test_resume_error_is_shown(self, app, monkeypatch):
        monkeypatch.setattr(callbacks, "RUN", FakeRun(error=ValueError("There is no stopped run to resume")))
        trigger(monkeypatch, "resume-training")
        out = app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto")
        assert out[7] == "There is no stopped run to resume"

    @pytest.mark.parametrize("triggered", ["train-poll-timer", None, "ft-start-training"])
    def test_poll_or_foreign_trigger_only_renders(self, app, fake_run, monkeypatch, triggered):
        trigger(monkeypatch, triggered)
        fake_run.state = snap(state="running", mode="scratch", iteration=5, ckpt_every=10)
        out = app.get("train-poll-timer", "control_training")(*CONTROL_ARGS, "runs", "x", "auto")
        assert fake_run.calls == []
        assert out == _render_training(fake_run.state, "scratch")


class TestUpdateMetrics:
    def test_scratch_panel(self, app, fake_run):
        fake_run.state = snap(state="running", mode="scratch", iteration=3, ckpt_every=10, loss=0.5)
        out = app.get("metric-loss-value", "update_training_metrics")(1, False)
        assert out == _render_metrics(fake_run.state, "scratch")
        assert out[0] == "0.5000"

    def test_finetune_panel_ignores_scratch_run(self, app, fake_run):
        fake_run.state = snap(state="running", mode="scratch", iteration=3, ckpt_every=10, loss=0.5)
        assert app.get("ft-metric-loss-value", "update_training_metrics")(1, False) == IDLE_METRICS
