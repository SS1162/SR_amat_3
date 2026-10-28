"""Eval viewer: sample selection, cropping, cache and the bicubic/model metric fairness."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from eval_viewer import data as ev_data
from eval_viewer import evaluate as ev
from eval_viewer.evaluate import SampleSpec


class BicubicModel(nn.Module):
    """Stand-in for SPFI: plain bicubic x4, so its metrics must equal the bicubic reference."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i, size in enumerate([(70, 66), (90, 90), (64, 81), (100, 72), (88, 88)]):
        arr = (rng.random(size[::-1]) * 255).astype(np.uint8)
        Image.fromarray(arr).convert("RGB").save(root / "images" / f"img{i}.png")
    (root / "test.csv").write_text("img1\nimg3\n")
    return root


@pytest.fixture
def cache(tmp_path: Path, monkeypatch) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(ev, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(ev_data, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(ev, "_load_model", lambda path, device: BicubicModel())
    return cache_dir


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "fake.pt"
    torch.save({"hparams": {"scale": 4, "embed_dim": 8}, "iteration": 7}, path)
    return path


def test_selection_is_deterministic_and_bounded(dataset):
    spec = SampleSpec(dataset, n=3, seed=1, crop="full")
    assert ev.select_images(spec) == ev.select_images(spec)
    assert len(ev.select_images(spec)) == 3
    assert len(ev.select_images(SampleSpec(dataset, n=50, seed=1, crop="full"))) == 5


def test_test_subset_uses_test_csv(dataset):
    stems = [p.stem for p in ev.select_images(SampleSpec(dataset, n=10, seed=0, crop="full", subset="test"))]
    assert stems == ["img1", "img3"]


@pytest.mark.parametrize("crop, expected", [("full", (64, 96)), ("64", (64, 64)), ("256", (64, 96))])
def test_center_crop_sizes_are_multiples_of_8(crop, expected):
    out = ev.center_crop(torch.zeros(1, 70, 100), crop)
    assert tuple(out.shape[1:]) == expected


def test_blur_sigma_in_training_range():
    s = ev.blur_sigma("abc")
    assert ev.SIGMA_MIN <= s <= ev.SIGMA_MAX
    assert s == ev.blur_sigma("abc")


def test_evaluate_writes_cache_and_metrics_match_bicubic(dataset, cache, checkpoint):
    spec = SampleSpec(dataset, n=4, seed=0, crop="64")
    out = ev.evaluate_model(spec, checkpoint)

    assert ev.is_evaluated(spec, checkpoint)
    rows = ev_data.load_rows(spec, checkpoint, None)
    assert len(rows) == 4
    for r in rows:
        assert (out / "sr" / f"{r['stem']}.png").is_file()
        # Same HR, same LR, same upsampling -> identical PSNR: model and bicubic are measured alike.
        assert r["psnr"] == pytest.approx(r["bicubic_psnr"], abs=1e-3)
        assert r["d_psnr"] == pytest.approx(0, abs=1e-3)
    assert ev_data.summarize(rows)["n"] == 4
    assert ev_data.model_meta(spec, checkpoint)["iteration"] == 7


def test_base_comparison_rows(dataset, cache, checkpoint, tmp_path):
    other = tmp_path / "other.pt"
    other.write_bytes(checkpoint.read_bytes())
    spec = SampleSpec(dataset, n=2, seed=0, crop="64")
    ev.evaluate_model(spec, checkpoint)
    assert "d_base" not in ev_data.load_rows(spec, checkpoint, other)[0]   # base not evaluated yet
    ev.evaluate_model(spec, other)
    row = ev_data.load_rows(spec, checkpoint, other)[0]
    assert row["d_base"] == pytest.approx(0, abs=1e-6)
    stats = ev_data.summarize(ev_data.load_rows(spec, checkpoint, other))
    assert "d_base_mean" in stats


def test_stop_event_interrupts(dataset, cache, checkpoint):
    import threading
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        ev.evaluate_model(SampleSpec(dataset, n=2, seed=0, crop="64"), checkpoint, stop=stop)


def test_load_views_shapes(dataset, cache, checkpoint):
    spec = SampleSpec(dataset, n=1, seed=0, crop="64")
    ev.evaluate_model(spec, checkpoint)
    stem = ev_data.load_rows(spec, checkpoint, None)[0]["stem"]
    views = ev_data.load_views(spec, stem, checkpoint, None)
    assert {v.shape for v in views.values()} == {(64, 64)}
    assert ev_data.image_url(spec, "sr", stem, checkpoint).startswith("/cache/")
