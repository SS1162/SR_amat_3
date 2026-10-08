import threading
from pathlib import Path

import pytest
from PIL import Image

from spfi_ui.training import runner
from spfi_ui.training.runner import (
    MIN_IMAGES, TrainingRun, _as_lists, _normalize_run_dir, _save_uploads, _validate_run_dir,
)

from .helpers import FakeTrainer, png_data_url, wait_for


def _uploads(*names):
    return [png_data_url() for _ in names], list(names)


# ---------------------------------------------------------------- _as_lists

class TestAsLists:
    def test_empty_contents_give_empty_lists(self):
        assert _as_lists(None, None) == ([], [])
        assert _as_lists([], ["a.png"]) == ([], [])
        assert _as_lists("", "a.png") == ([], [])

    def test_single_string_is_wrapped(self):
        assert _as_lists("data", "a.png") == (["data"], ["a.png"])

    def test_lists_are_copied(self):
        contents, names = ["x", "y"], ["a.png", "b.png"]
        out_contents, out_names = _as_lists(contents, names)
        assert (out_contents, out_names) == (contents, names)
        assert out_contents is not contents and out_names is not names

    def test_missing_filenames_become_none(self):
        assert _as_lists(("x", "y"), None) == (["x", "y"], [None, None])


# ------------------------------------------------- _normalize_run_dir / _validate_run_dir

class TestRunDir:
    def test_absolute_path_is_kept(self, tmp_path):
        assert _normalize_run_dir(tmp_path / "x") == tmp_path / "x"

    def test_relative_path_is_under_repo_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
        assert _normalize_run_dir(Path("runs") / "x") == tmp_path / "runs" / "x"

    def test_new_run_in_existing_output_dir(self, tmp_path):
        assert _validate_run_dir(tmp_path / "run") == tmp_path / "run"

    def test_existing_run_dir(self, tmp_path):
        (tmp_path / "run").mkdir()
        assert _validate_run_dir(tmp_path / "run") == tmp_path / "run"

    def test_output_dir_may_be_created_one_level_deep(self, tmp_path):
        assert _validate_run_dir(tmp_path / "new_output" / "run") == tmp_path / "new_output" / "run"

    def test_relative_is_normalized(self, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
        assert _validate_run_dir(Path("runs") / "x") == tmp_path / "runs" / "x"

    def test_run_dir_is_a_file(self, tmp_path):
        (tmp_path / "run").write_text("x")
        with pytest.raises(ValueError, match="Output path points to a file"):
            _validate_run_dir(tmp_path / "run")

    def test_output_dir_is_a_file(self, tmp_path):
        (tmp_path / "out").write_text("x")
        with pytest.raises(ValueError, match="Output directory points to a file"):
            _validate_run_dir(tmp_path / "out" / "run")

    def test_missing_grandparent(self, tmp_path):
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            _validate_run_dir(tmp_path / "a" / "b" / "run")

    def test_start_rejects_invalid_run_dir_before_saving(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            training_run.start(contents, names, None, None, tmp_path / "a" / "b" / "run", "cpu")
        assert not (tmp_path / "a").exists()
        assert training_run.state == "idle"


# ------------------------------------------------------------ _save_uploads

class TestSaveUploads:
    def test_saves_each_upload_as_png_named_by_stem(self, tmp_path):
        contents, names = _uploads("a.png", "b.jpg")
        folder = tmp_path / "nested" / "images"

        stems = _save_uploads(contents, names, folder)

        assert stems == ["a", "b"]
        assert sorted(p.name for p in folder.iterdir()) == ["a.png", "b.png"]
        with Image.open(folder / "b.png") as img:
            assert img.format == "PNG" and img.size == (8, 8)

    def test_missing_filename_gets_indexed_default(self, tmp_path):
        stems = _save_uploads([png_data_url(), png_data_url()], [None, "x.png"], tmp_path)
        assert stems == ["upload_0000", "x"]
        assert (tmp_path / "upload_0000.png").is_file()

    def test_duplicate_stems_are_rejected(self, tmp_path):
        contents, names = _uploads("a.png", "a.jpg")
        with pytest.raises(ValueError, match="share the name 'a'"):
            _save_uploads(contents, names, tmp_path)

    @pytest.mark.parametrize("content", [
        "no-comma-here",                       # IndexError on split
        "data:image/png;base64,@@@notb64",     # binascii.Error (a ValueError)
        "data:text/plain;base64,aGVsbG8=",     # valid base64, not an image
    ])
    def test_unreadable_content_raises_value_error(self, tmp_path, content):
        with pytest.raises(ValueError, match="Could not read 'bad.png' as an image"):
            _save_uploads([content], ["bad.png"], tmp_path)

    def test_empty_input_creates_folder_only(self, tmp_path):
        folder = tmp_path / "empty"
        assert _save_uploads([], [], folder) == []
        assert folder.is_dir()


# ------------------------------------------------------- TrainingRun.start validation

class TestStartValidation:
    def test_initial_snapshot_is_idle(self):
        snap = TrainingRun().snapshot()
        assert snap == {
            "mode": None, "val": None, "elapsed": 0.0, "state": "idle", "error": None,
            "iteration": 0, "loss": None, "ckpt_every": None, "run_dir": None,
        }

    @pytest.mark.parametrize("count", range(MIN_IMAGES))
    def test_requires_min_images(self, tmp_path, training_run, count):
        contents, names = _uploads(*[f"{i}.png" for i in range(count)])
        with pytest.raises(ValueError, match=f"at least {MIN_IMAGES}"):
            training_run.start(contents, names, None, None, tmp_path / "run", "cpu")
        assert not (tmp_path / "run").exists()
        assert training_run.state == "idle"

    def test_single_string_upload_counts_as_one_image(self, tmp_path, training_run):
        with pytest.raises(ValueError, match="at least"):
            training_run.start(png_data_url(), "a.png", None, None, tmp_path / "run", "cpu")

    def test_existing_data_dir_is_rejected(self, tmp_path, training_run):
        (tmp_path / "run" / "data").mkdir(parents=True)
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="already has data"):
            training_run.start(contents, names, None, None, tmp_path / "run", "cpu")

    def test_finetune_requires_weights(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="Choose the checkpoint"):
            training_run.start(contents, names, None, None, tmp_path / "run", "cpu", mode="finetune")

    def test_finetune_weights_must_exist(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="Checkpoint not found"):
            training_run.start(contents, names, None, None, tmp_path / "run", "cpu",
                               mode="finetune", weights=str(tmp_path / "missing.pt"))

    def test_finetune_weights_must_be_a_file(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="Checkpoint not found"):
            training_run.start(contents, names, None, None, tmp_path / "run", "cpu",
                               mode="finetune", weights=str(tmp_path))

    def test_bad_image_cleans_up_data_dir(self, tmp_path, training_run):
        contents = [png_data_url(), "garbage"]
        with pytest.raises(ValueError, match="Could not read"):
            training_run.start(contents, ["a.png", "b.png"], None, None, tmp_path / "run", "cpu")
        assert not (tmp_path / "run" / "data").exists()
        assert training_run.state == "idle"

    def test_unmatched_masks_are_rejected_and_cleaned_up(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        masks, mask_names = _uploads("a.png", "z.png", "y.png")
        with pytest.raises(ValueError, match="Masks without a matching image: y, z"):
            training_run.start(contents, names, masks, mask_names, tmp_path / "run", "cpu")
        assert not (tmp_path / "run" / "data").exists()

    def test_bad_mask_cleans_up_data_dir(self, tmp_path, training_run):
        contents, names = _uploads("a.png", "b.png")
        with pytest.raises(ValueError, match="Could not read"):
            training_run.start(contents, names, ["garbage"], ["a.png"], tmp_path / "run", "cpu")
        assert not (tmp_path / "run" / "data").exists()

    def test_relative_run_dir_resolves_under_repo_root(self, monkeypatch, tmp_path, training_run, fake_train):
        monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
        contents, names = _uploads("a.png", "b.png")

        training_run.start(contents, names, None, None, "runs/rel", "cpu")

        assert training_run.run_dir == tmp_path / "runs" / "rel"
        assert (tmp_path / "runs" / "rel" / "data" / "images" / "a.png").is_file()

    def test_repo_root_points_at_repository(self):
        assert (runner.REPO_ROOT / "algo").is_dir()
        assert (runner.REPO_ROOT / "ui" / "spfi_ui").is_dir()


# ------------------------------------------------------- TrainingRun lifecycle

class TestLifecycle:
    def _start(self, run, tmp_path, **kwargs):
        contents, names = _uploads("a.png", "b.png")
        run.start(contents, names, kwargs.pop("masks", None), kwargs.pop("mask_names", None),
                  tmp_path / "run", kwargs.pop("device", "cpu"), **kwargs)

    def test_start_saves_data_and_runs_trainer(self, tmp_path, training_run, fake_train):
        masks, mask_names = _uploads("a.png")
        self._start(training_run, tmp_path, masks=masks, mask_names=mask_names)

        wait_for(lambda: training_run.state == "running")
        trainer = fake_train.instances[0]
        assert trainer.entered.wait(5)
        data_dir = tmp_path / "run" / "data"
        assert sorted(p.name for p in (data_dir / "images").iterdir()) == ["a.png", "b.png"]
        assert [p.name for p in (data_dir / "masks").iterdir()] == ["a.png"]
        assert trainer.kwargs == {
            "emps_dir": data_dir, "ckpt_dir": tmp_path / "run" / "checkpoints", "device": "cpu",
            "init": "scratch", "weights": None, "run_dir": tmp_path / "run",
        }
        snap = training_run.snapshot()
        assert snap["mode"] == "scratch"
        assert snap["ckpt_every"] == 10
        assert snap["iteration"] == 1
        assert snap["loss"] == 0.5
        assert snap["run_dir"] == str(tmp_path / "run")

    def test_start_while_active_is_rejected(self, tmp_path, training_run, fake_train):
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "running")
        contents, names = _uploads("c.png", "d.png")
        with pytest.raises(ValueError, match="already active"):
            training_run.start(contents, names, None, None, tmp_path / "other", "cpu")
        assert not (tmp_path / "other").exists()

    def test_finetune_passes_weights_to_trainer(self, tmp_path, training_run, fake_train):
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"x")
        self._start(training_run, tmp_path, mode="finetune", weights=str(weights))

        wait_for(lambda: fake_train.instances)
        kwargs = fake_train.instances[0].kwargs
        assert kwargs["init"] == "finetune"
        assert kwargs["weights"] == weights
        assert training_run.mode == "finetune"

    def test_scratch_ignores_weights(self, tmp_path, training_run, fake_train):
        self._start(training_run, tmp_path, weights="ignored.pt")
        wait_for(lambda: fake_train.instances)
        assert fake_train.instances[0].kwargs["weights"] is None

    def test_natural_finish(self, tmp_path, training_run, fake_train):
        fake_train.block = False
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "finished")
        assert training_run.error is None
        assert training_run.snapshot()["elapsed"] >= 0.0

    def test_stop_then_resume_then_finish(self, tmp_path, training_run, fake_train):
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "running")
        trainer = fake_train.instances[0]
        assert trainer.entered.wait(5)

        training_run.stop()
        assert training_run.state in ("stopping", "stopped")
        wait_for(lambda: training_run.state == "stopped")
        elapsed_after_stop = training_run.snapshot()["elapsed"]
        assert elapsed_after_stop >= 0.0   # Windows monotonic clock ticks every ~15 ms
        assert training_run._segment_start is None

        trainer.entered.clear()
        training_run.resume()
        assert training_run.state == "running"
        assert trainer.entered.wait(5)
        assert trainer.train_calls == 2
        assert fake_train.instances == [trainer]   # resume reuses the same trainer

        trainer.release.set()
        wait_for(lambda: training_run.state == "finished")
        assert training_run.snapshot()["elapsed"] >= elapsed_after_stop

    def test_stop_during_preparing_ends_stopped_without_training(self, tmp_path, training_run, fake_train,
                                                                 monkeypatch):
        gate = threading.Event()
        original_init = FakeTrainer.__init__

        def slow_init(self, **kwargs):
            gate.wait(5)
            original_init(self, **kwargs)

        monkeypatch.setattr(FakeTrainer, "__init__", slow_init)
        self._start(training_run, tmp_path)
        assert training_run.state == "preparing"

        training_run.stop()
        assert training_run.state == "stopping"
        gate.set()

        wait_for(lambda: training_run.state == "stopped")
        assert fake_train.instances[0].train_calls == 0
        assert training_run.snapshot()["elapsed"] == 0.0

    def test_stop_is_noop_when_not_active(self, training_run):
        training_run.stop()
        assert training_run.state == "idle"
        for state in ("stopped", "finished", "error"):
            training_run.state = state
            training_run.stop()
            assert training_run.state == state

    def test_resume_requires_stopped_run(self, training_run):
        with pytest.raises(ValueError, match="no stopped run"):
            training_run.resume()
        training_run.state = "stopped"   # stopped but no trainer
        with pytest.raises(ValueError, match="no stopped run"):
            training_run.resume()

    def test_trainer_init_error_sets_error_state(self, tmp_path, training_run, fake_train):
        fake_train.init_error = RuntimeError("no GPU")
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "error")
        assert training_run.error == "RuntimeError: no GPU"
        assert training_run.trainer is None
        assert training_run.snapshot()["iteration"] == 0

    def test_train_error_sets_error_state(self, tmp_path, training_run, fake_train):
        fake_train.train_error = ValueError("nan loss")
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "error")
        assert training_run.error == "ValueError: nan loss"
        assert training_run._segment_start is None

    def test_can_start_new_run_after_finish(self, tmp_path, training_run, fake_train):
        fake_train.block = False
        self._start(training_run, tmp_path)
        wait_for(lambda: training_run.state == "finished")
        training_run._thread.join(5)

        contents, names = _uploads("c.png", "d.png")
        training_run.start(contents, names, None, None, tmp_path / "run2", "cpu")
        wait_for(lambda: training_run.state == "finished" and training_run.run_dir == tmp_path / "run2")
        assert len(fake_train.instances) == 2
        assert training_run.error is None


# ------------------------------------------------------------- snapshot

class TestSnapshot:
    def test_reports_trainer_fields(self):
        run = TrainingRun()

        class Trainer:
            iteration = 42
            last_loss = 3      # tensors / ints are converted with float()
            last_val = (0.1, 30.5)

        run.trainer = Trainer()
        run.run_dir = "some/dir"
        snap = run.snapshot()
        assert snap["iteration"] == 42
        assert snap["loss"] == 3.0 and isinstance(snap["loss"], float)
        assert snap["val"] == (0.1, 30.5)
        assert snap["run_dir"] == "some/dir"

    def test_elapsed_includes_running_segment(self, monkeypatch):
        run = TrainingRun()
        run._elapsed = 5.0
        run._segment_start = 100.0
        monkeypatch.setattr(runner.time, "monotonic", lambda: 103.0)
        assert run.snapshot()["elapsed"] == 8.0

    def test_module_level_singleton(self):
        assert isinstance(runner.RUN, TrainingRun)
