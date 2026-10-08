# Inference: required changes outside the inference core

The inference core (`inference/`) is implemented against the target contracts below.
These are the gaps between the current training/model/app code and what inference
needs. Items 1–8 are still open.

## Done
- **`load_image`** (`inference/image_io.py`) reads the image directly from its original
  location with Pillow. It copies nothing to disk.
  - Grayscale images keep their raw dtype; for example, a 16-bit TIFF stays uint16.
  - Colour and palette images are converted to 8-bit grayscale (`"L"`).
- **`save_image`** (`inference/image_io.py`) clips the float32 image to [0, 1], scales it
  to uint16 and writes a 16-bit TIFF (mode `I;16`) with Pillow. No new dependency was
  needed.
  - An existing file is overwritten.
  - The output folder must already exist; the UI is responsible for that.

## 1. Checkpoint must store model hyperparameters
- **Current state:** `SPFITrainer._save_checkpoint` saves
  `{"iteration", "model_state", "optimizer_state", "best_val_loss"}`. It saves no
  architecture hyperparameters.
- **Required change:** also save `"hparams": {"scale": int, "embed_dim": int}` next to
  `"model_state"`.
- **File:** `algo/train.py`
- **Why:** `inference/runner.py::_load_model` builds the model with
  `build_spfi(scale=..., embed_dim=...)` read from `ckpt["hparams"]`. There's no fallback
  to hardcoded values. Checkpoints in the old format are rejected with `ModelLoadError`.

## 2. Implement `normalize_image`
- **Current state:** `normalize_image` is a stub that raises `NotImplementedError`.
- **Required change:** implement it to turn a raw grayscale image (uint8 / uint16 / ...)
  into a 2D float32 array in [0, 1].
- **File:** `algo/preprocessing.py`
- **Why:** `inference/dataset.py` runs `load_image -> normalize_image -> to_tensor`.
  Inference can't process any image until this exists.

## 3. Training must use `normalize_image`
- **Current state:** `EMPSDataset._load_gray_tensor` uses PIL `convert("L")` + `to_tensor`.
  `convert("L")` clips 16-bit data to 8-bit.
- **Required change:** load the raw array and call `normalize_image` before `to_tensor`.
- **File:** `algo/train.py`
- **Why:** training and inference must normalize identically. Otherwise the model sees
  differently scaled inputs at inference time, and 16-bit SEM detail is lost in training.

## 4. Open question: how is the true bit depth determined?
- **Current state:** not decided.
- **Required change:** the team decides how `normalize_image` knows the real bit depth.
  For example, 12-bit data stored in uint16 should not be divided by 65535.
- **File:** `algo/preprocessing.py`
- **Why:** normalization depends on it, and a wrong divisor silently darkens images.

## 5. Implement `resolve_inputs`
- **Current state:** `resolve_inputs` raises `NotImplementedError`. `load_image` and
  `save_image` are implemented with Pillow; `save_image` writes a 16-bit TIFF.
- **Required change:** implement `resolve_inputs` according to its docstring: paths at
  their original location, no copying, sorted deterministically.
- **File:** `inference/image_io.py`
- **Why:** `start_inference` can't find any input images without it.

## 6. The repo root must be importable from the Dash process
- **Current state:** `ui/app.py` imports `spfi_ui` relative to `ui/`, so the app is run
  from `ui/`. The repo root is not on `sys.path`.
- **Required change:** start the app so that the repo root is on `sys.path`, for
  example by running from the root or setting `PYTHONPATH`.
- **File:** app startup (`ui/app.py` / launch command)
- **Why:** `inference` imports `algo.model` and `algo.preprocessing`. DataLoader workers
  use the `spawn` context and re-import `inference.dataset`, which only works when the
  root is importable.

## 7. Unexpected errors propagate out of `start_inference`
- **Current state:** defined failures (`InferenceError` subclasses) produce a FAILED
  event and a returned result. Anything else, such as a programming error, propagates
  without a terminal event.
- **Required change:** the Dash layer handles exceptions at the background-callback level.
- **File:** Dash inference callbacks (`ui/spfi_ui/callbacks.py`)
- **Why:** without this the UI would never receive a terminal event for such a run.

## 8. `pytorch_wavelets` is required to build the real model
- **Current state:** `pytorch_wavelets` isn't installed in the current environment.
  `build_spfi` raises `DTCWTUnavailable` from `algo/dtcwt.py`.
- **Required change:** install it wherever inference runs, and record it in the
  project's dependency list once one exists.
- **File:** environment / dependency list
- **Why:** with it missing, every run fails at model load (`ModelLoadError`).
