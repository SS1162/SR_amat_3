"""Unit test for the checkpoint format written by SPFITrainer._save_checkpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from algo import train

ITERATION = 7
BEST_VAL_LOSS = 0.5
LEARNING_RATE = 0.1


def test_save_checkpoint_writes_hparams(tmp_path: Path) -> None:
    """The checkpoint stores "hparams" with the scale and embed_dim inference needs."""
    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(
        ckpt_dir=tmp_path,
        iteration=ITERATION,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=LEARNING_RATE),
        best_val_loss=BEST_VAL_LOSS,
        embed_dim=train.EMBED_DIM,
    )

    train.SPFITrainer._save_checkpoint(trainer, "best")

    checkpoint = torch.load(tmp_path / "spfi_x4_best.pt", weights_only=True)
    assert checkpoint["hparams"] == {"scale": train.SCALE, "embed_dim": train.EMBED_DIM}
    assert checkpoint["iteration"] == ITERATION
