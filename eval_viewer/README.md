# SR Eval Viewer

A developer tool for looking at what a checkpoint does to the images, not only at its metrics.
**Inference only: it never trains.**

## Run

```bash
pip install -r requirements.txt          # same as the main app
python -m eval_viewer.app                 # from the repo root
```

Open http://127.0.0.1:8051

## How it works

1. **Sample**: N random images from the dataset (fixed `seed`, so every model sees the same
   images), center-cropped to the chosen size (`256` = the training patch, the default).
2. **LR input**: the training degradation `algo.train.degrade` (Gaussian blur with a σ fixed
   per file, then bicubic ↓4), so the model sees inputs like those it was trained on.
3. **Outputs**: bicubic ↑4 and the model's SR. PSNR / SSIM come from the training code, so the
   numbers match the training logs.
4. **Cache**: everything goes to `eval_viewer/eval_cache/<sample>/` (git-ignored). Each
   (sample, model) is computed once. Switching models or sorting is then instant.

CPU speed: ~2 s per image at crop 256, ~20 s at full size (the attention cost grows faster than the area).

## Screen

- **Model / Compare with (base)**: checkpoints found in `eval_viewer/models/` and
  `runs/*/checkpoints/`. The base model is `eval_viewer/models/base/spfi_x4_base.pt`
  (`*.pt` is git-ignored, so share it separately).
- **Evaluate**: computes whatever is missing for the selected model and base.
- **Stats**: mean / median / std PSNR, SSIM, gain over bicubic and over base, % of images improved.
- **Charts**: PSNR distribution, ΔPSNR histogram, model vs bicubic scatter (points below
  y = x are images where the model hurts), ΔPSNR vs blur σ. Click a point to open that image.
- **Grid**: HR | model output, a small bicubic thumbnail, PSNR / SSIM, ▲/▼ badges. Sort
  (worst first by default), filter (worse than bicubic / base, σ range), export CSV.
- **Detail** (click a cell): LR, bicubic, model, base, HR and the |error| maps, with
  synchronized zoom. Prev / Next follow the grid order.

## CLI (precompute without the UI)

```bash
python -m eval_viewer.evaluate --model eval_viewer/models/base/spfi_x4_base.pt \
    --model runs/<run>/checkpoints/spfi_x4_best.pt --data <EMPS dir> --n 100 --crop 256
```

The dataset defaults to `~/Downloads/NanoVision-spfi/datasets/emps/EMPS`. You can override
it with `SR_EVAL_DATA` or in the UI.
