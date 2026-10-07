import sys
import types

import pytest

from spfi_ui.training import runner

from .helpers import FakeTrainer


@pytest.fixture
def fake_train(monkeypatch):
    """Installs a fake ``algo.train`` module so TrainingRun._run imports FakeTrainer instead of torch."""
    FakeTrainer.instances = []
    FakeTrainer.init_error = None
    FakeTrainer.train_error = None
    FakeTrainer.block = True
    module = types.ModuleType("algo.train")
    module.CKPT_EVERY = 10
    module.SPFITrainer = FakeTrainer
    package = types.ModuleType("algo")
    package.train = module
    monkeypatch.setitem(sys.modules, "algo", package)
    monkeypatch.setitem(sys.modules, "algo.train", module)
    return FakeTrainer


@pytest.fixture
def training_run():
    """A fresh TrainingRun whose background thread is stopped after the test."""
    run = runner.TrainingRun()
    yield run
    run._stop_event.set()
    for trainer in FakeTrainer.instances:
        trainer.release.set()
    if run._thread is not None:
        run._thread.join(timeout=5)
