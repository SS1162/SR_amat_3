"""Shared test helpers for the training tab."""

import base64
import io
import threading
import time

from PIL import Image


def png_data_url(size=(8, 8), color=(255, 0, 0)):
    """A dcc.Upload-style data URL holding a small PNG."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def wait_for(predicate, timeout=5.0):
    """Polls ``predicate`` until it is true; fails the test on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


class FakeTrainer:
    """Mimics the SPFITrainer surface that TrainingRun uses: iteration, last_loss, last_val, train()."""

    instances = []
    init_error = None    # raise this from __init__ when set
    train_error = None   # raise this from train() when set
    block = True         # train() waits for stop_event (or `release`) instead of returning at once

    def __init__(self, **kwargs):
        if FakeTrainer.init_error is not None:
            raise FakeTrainer.init_error
        self.kwargs = kwargs
        self.iteration = 0
        self.last_loss = None
        self.last_val = None
        self.train_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        FakeTrainer.instances.append(self)

    def train(self, stop_event=None):
        self.train_calls += 1
        self.iteration += 1
        self.last_loss = 0.5
        self.entered.set()
        if FakeTrainer.train_error is not None:
            raise FakeTrainer.train_error
        if FakeTrainer.block:
            while not (stop_event.is_set() or self.release.is_set()):
                stop_event.wait(0.01)
        self.release.clear()
