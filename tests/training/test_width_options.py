"""Tests for the trainer options used by fixed-budget width comparisons:
``embed_dim``, ``max_iters`` and ``split_seed`` / ``split_files``."""

from __future__ import annotations

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
    return train.SPFITrainer(emps_dir=emps_dir, ckpt_dir=tmp_path / "ckpt", device="cpu", **kwargs)


def test_split_files_matches_original_split() -> None:
    """split_files reproduces the split the trainer used before it was factored out."""
    stems = [f"img{i:03d}" for i in range(50)]
    expected = sorted(stems)
    random.Random(3).shuffle(expected)
    assert train.split_files(reversed(stems), 3) == (expected[:40], expected[40:])


def test_embed_dim_sets_model_width_and_checkpoint(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path, embed_dim=8)
    assert trainer.model.embed_dim == 8
    trainer._save_checkpoint("best")
    ckpt = torch.load(tmp_path / "ckpt" / "spfi_x4_best.pt", weights_only=True)
    assert ckpt["hparams"] == {"scale": train.SCALE, "embed_dim": 8}


def test_default_embed_dim_unchanged(emps_dir: Path, tmp_path: Path) -> None:
    assert make_trainer(emps_dir, tmp_path).model.embed_dim == train.EMBED_DIM


def test_max_iters_stops_exactly(emps_dir: Path, tmp_path: Path) -> None:
    trainer = make_trainer(emps_dir, tmp_path, embed_dim=8)
    trainer.train(max_iters=3)
    assert trainer.iteration == 3
    trainer.train(max_iters=5)   # resuming continues up to the new total
    assert trainer.iteration == 5


def test_split_seed_keeps_val_images_across_training_seeds(emps_dir: Path, tmp_path: Path) -> None:
    a = make_trainer(emps_dir, tmp_path, embed_dim=8, seed=0, split_seed=0)
    b = make_trainer(emps_dir, tmp_path, embed_dim=8, seed=1, split_seed=0)
    c = make_trainer(emps_dir, tmp_path, embed_dim=8, seed=1)   # split follows seed
    assert a.val_ds.filenames == b.val_ds.filenames
    assert c.val_ds.filenames == train.split_files(a.val_ds.filenames + a.train_ds.filenames, 1)[1]
