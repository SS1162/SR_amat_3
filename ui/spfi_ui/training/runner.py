"""Runs SPFITrainer in a background thread so Dash callbacks stay responsive.

Flow: the uploaded images/masks are written to ``<run_dir>/data/{images,masks}``
as PNG, then ``SPFITrainer(emps_dir=<run_dir>/data, ckpt_dir=<run_dir>/checkpoints)``
trains on them. Masks are saved for later use; the trainer does not read them yet.
"""

import base64
import io
import shutil
import sys
import threading
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_IMAGES = 2  # train.py splits 80/20, so both splits need at least one image


def _as_lists(contents, filenames):
    if not contents:
        return [], []
    if isinstance(contents, str):
        return [contents], [filenames]
    return list(contents), list(filenames or [None] * len(contents))


def _save_uploads(contents, filenames, folder: Path) -> list[str]:
    """Decode dcc.Upload data URLs and save each one as <stem>.png. Returns the stems."""
    folder.mkdir(parents=True, exist_ok=True)
    stems = []
    for index, (content, filename) in enumerate(zip(contents, filenames)):
        name = filename or f"upload_{index:04d}.png"
        stem = Path(name).stem
        if stem in stems:
            raise ValueError(f"Two uploaded files share the name '{stem}'")
        try:
            data = base64.b64decode(content.split(",", 1)[1])
            with Image.open(io.BytesIO(data)) as img:
                img.save(folder / f"{stem}.png")
        except (IndexError, ValueError, UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read '{name}' as an image") from exc
        stems.append(stem)
    return stems


class TrainingRun:
    """A single training run at a time.

    state: idle -> preparing -> running -> (stopping ->) stopped | finished | error
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self.trainer = None
        self.state = "idle"
        self.error = None
        self.run_dir = None
        self.mode = None              # "scratch" or "finetune" for the current/last run
        self.ckpt_every = None
        self._elapsed = 0.0          # seconds spent in finished train() segments
        self._segment_start = None   # time.monotonic() when the current segment began

    def start(self, image_contents, image_names, mask_contents, mask_names, run_dir, device,
              mode="scratch", weights=None):
        """mode: "scratch" or "finetune" (an algo.train INIT_REGISTRY key); weights: checkpoint for finetune."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("A training run is already active")
            if mode == "finetune":
                if not weights:
                    raise ValueError("Choose the checkpoint (.pt) to fine-tune from")
                weights = Path(weights)
                if not weights.is_file():
                    raise ValueError(f"Checkpoint not found: {weights}")

            images, image_names = _as_lists(image_contents, image_names)
            masks, mask_names = _as_lists(mask_contents, mask_names)
            if len(images) < MIN_IMAGES:
                raise ValueError(f"Upload at least {MIN_IMAGES} training images")

            run_dir = Path(run_dir)
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir   # "runs" = <repo>/runs, where the TensorBoard page looks
            data_dir = run_dir / "data"
            if data_dir.exists():
                raise ValueError(f"'{run_dir}' already has data; choose another run name")

            try:
                image_stems = _save_uploads(images, image_names, data_dir / "images")
                if masks:
                    mask_stems = _save_uploads(masks, mask_names, data_dir / "masks")
                    unmatched = sorted(set(mask_stems) - set(image_stems))
                    if unmatched:
                        raise ValueError(f"Masks without a matching image: {', '.join(unmatched)}")
            except Exception:
                shutil.rmtree(data_dir, ignore_errors=True)  # created above, so safe to remove
                raise

            self.trainer = None
            self.error = None
            self.run_dir = run_dir
            self.mode = mode
            self._elapsed, self._segment_start = 0.0, None
            self.state = "preparing"
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                kwargs={"data_dir": data_dir, "ckpt_dir": run_dir / "checkpoints", "device": device,
                        "init": mode, "weights": weights if mode == "finetune" else None},
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        with self._lock:
            if self.state in ("preparing", "running"):
                self.state = "stopping"
                self._stop_event.set()

    def resume(self):
        with self._lock:
            if self.state != "stopped" or self.trainer is None:
                raise ValueError("There is no stopped run to resume")
            self.state = "running"
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._train, daemon=True)
            self._thread.start()

    def _run(self, data_dir, ckpt_dir, device, init, weights):
        try:
            from algo.train import CKPT_EVERY, SPFITrainer  # torch is heavy; import only when training

            self.ckpt_every = CKPT_EVERY
            trainer = SPFITrainer(emps_dir=data_dir, ckpt_dir=ckpt_dir, device=device,
                                  init=init, weights=weights, run_dir=data_dir.parent)
        except Exception as exc:
            self._finish(error=exc)
            return
        self.trainer = trainer
        with self._lock:
            if self._stop_event.is_set():
                self.state = "stopped"
                return
            self.state = "running"
        self._train()

    def _train(self):
        self._segment_start = time.monotonic()
        try:
            self.trainer.train(stop_event=self._stop_event)
        except Exception as exc:
            self._finish(error=exc)
            return
        self._finish()

    def _finish(self, error=None):
        with self._lock:
            if self._segment_start is not None:
                self._elapsed += time.monotonic() - self._segment_start
                self._segment_start = None
            if error is not None:
                self.state = "error"
                self.error = f"{type(error).__name__}: {error}"
            elif self._stop_event.is_set():
                self.state = "stopped"
            else:
                self.state = "finished"

    def snapshot(self):
        trainer = self.trainer
        last_loss = getattr(trainer, "last_loss", None)
        segment_start = self._segment_start
        elapsed = self._elapsed + (time.monotonic() - segment_start if segment_start is not None else 0.0)
        return {
            "mode": self.mode,
            "val": getattr(trainer, "last_val", None),
            "elapsed": elapsed,
            "state": self.state,
            "error": self.error,
            "iteration": trainer.iteration if trainer is not None else 0,
            "loss": float(last_loss) if last_loss is not None else None,
            "ckpt_every": self.ckpt_every,
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }


RUN = TrainingRun()
