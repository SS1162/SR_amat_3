"""Tests for the training-recipe options: batch_size, the learning-rate schedule,
EMA weights, flip/rotation augmentation and the image cache."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from algo import train

N_IMAGES = 10


@pytest.fixture
def emps_dir(tmp_path: Path) -> Path:
    """A tiny EMPS-like folder: images/<id>.png, grayscale."""
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    for i in range(N_IMAGES):
        arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(tmp_path / "images" / f"img{i:02d}.png")
    return tmp_path


def make_trainer(emps_dir: Path, tmp_path: Path, **kwargs) -> train.SPFITrainer:
    return train.SPFITrainer(emps_dir=emps_dir, ckpt_dir=tmp_path / "ckpt", device="cpu",
                             embed_dim=8, **kwargs)


# ---- learning-rate schedule ----

def test_constant_schedule_is_the_original_recipe() -> None:
    assert all(train.lr_at(i, 1e-4) == 1e-4 for i in (1, 10, 10_000))


def test_warmup_is_linear() -> None:
    assert train.lr_at(50, 2e-4, "cosine", warmup_iters=100, total_iters=1000) == pytest.approx(1e-4)
    assert train.lr_at(100, 2e-4, "cosine", warmup_iters=100, total_iters=1000) == pytest.approx(2e-4)


def test_cosine_decays_to_the_floor_and_stays_there() -> None:
    lr = lambda i: train.lr_at(i, 2e-4, "cosine", warmup_iters=100, total_iters=1000, min_lr_ratio=0.01)
    assert lr(550) == pytest.approx(2e-6 + (2e-4 - 2e-6) * 0.5)   # halfway through the decay
    assert lr(1000) == pytest.approx(2e-6)
    assert lr(1500) == pytest.approx(2e-6)
    rates = [lr(i) for i in range(100, 1001)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_cosine_needs_total_iters(emps_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="total_iters"):
        make_trainer(emps_dir, tmp_path, lr_schedule="cosine")


def test_trainer_applies_the_schedule(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path, lr=2e-4, lr_schedule="cosine", warmup_iters=2, total_iters=6)
    trainer.train(max_iters=4)
    expected = train.lr_at(4, 2e-4, "cosine", 2, 6)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected)


# ---- batch size ----

def test_batch_size_draws_that_many_patches_per_update(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path, batch_size=3)
    reference = random.Random()
    reference.setstate(trainer._train_rng.getstate())
    trainer.train(max_iters=2)
    for _ in range(6):
        trainer.train_ds.sample(reference)
    assert trainer._train_rng.getstate() == reference.getstate()


def test_batch_size_one_matches_the_original_loop(emps_dir: Path, tmp_path: Path) -> None:
    """With the defaults, the updates are exactly those of the original single-patch loop."""
    a = make_trainer(emps_dir, tmp_path / "a")
    a.train(max_iters=3)

    b = make_trainer(emps_dir, tmp_path / "b")
    for _ in range(3):
        lr_patch, hr_patch = b.train_ds.sample(b._train_rng)
        b.optimizer.zero_grad()
        train.total_loss(b.model(lr_patch.unsqueeze(0)), hr_patch.unsqueeze(0)).backward()
        b.optimizer.step()
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)


# ---- augmentation ----

def test_dihedral_gives_the_eight_distinct_transforms() -> None:
    img = torch.arange(16.0).view(1, 4, 4)
    outs = [train.dihedral(img, k) for k in range(8)]
    assert torch.equal(outs[0], img)
    assert len({tuple(o.flatten().tolist()) for o in outs}) == 8


def test_augment_off_does_not_touch_the_rng(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path)
    r1, r2 = random.Random(5), random.Random(5)
    lr1, hr1 = trainer.train_ds.sample(r1)
    lr2, hr2 = trainer.train_ds.sample(r2, augment=False)
    assert torch.equal(hr1, hr2) and r1.getstate() == r2.getstate()


# ---- EMA ----

def test_ema_weights_are_validated_and_exported(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path, lr=1e-2, ema_decay=0.9)
    trainer.train(max_iters=5)
    assert trainer.eval_model is trainer.ema_model
    trainer._save_checkpoint("final")
    ckpt = torch.load(tmp_path / "ckpt" / "spfi_x4_final.pt", weights_only=True)
    for name, value in trainer.ema_model.state_dict().items():
        assert torch.equal(ckpt["model_state"][name], value)
    raw = trainer.model.state_dict()
    assert all(torch.equal(ckpt["train_model_state"][n], raw[n]) for n in raw)
    assert any(not torch.equal(ckpt["model_state"][n], raw[n]) for n in raw)
    # the exported weights load into a fresh model, as in the app
    train.build_spfi(scale=train.SCALE, embed_dim=8).load_state_dict(ckpt["model_state"])


def test_no_ema_by_default(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path)
    assert trainer.ema_model is None and trainer.eval_model is trainer.model
    trainer._save_checkpoint("best")
    assert "train_model_state" not in torch.load(tmp_path / "ckpt" / "spfi_x4_best.pt", weights_only=True)


# ---- image cache ----

def test_cache_gives_the_same_patches(emps_dir: Path, tmp_path: Path) -> None:
    plain = make_trainer(emps_dir, tmp_path / "a")
    cached = make_trainer(emps_dir, tmp_path / "b", cache_images=True)
    r1, r2 = random.Random(3), random.Random(3)
    for _ in range(5):
        (l1, h1), (l2, h2) = plain.train_ds.sample(r1, True), cached.train_ds.sample(r2, True)
        assert torch.equal(l1, l2) and torch.equal(h1, h2)
    assert len(cached.train_ds._cache) > 0


def test_lr_at_rejects_unknown_schedule() -> None:
    with pytest.raises(ValueError, match="unknown lr schedule"):
        train.lr_at(1, 1e-4, "step")
    assert not math.isnan(train.lr_at(1, 1e-4, "cosine", total_iters=1))
