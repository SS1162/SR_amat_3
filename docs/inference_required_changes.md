# Inference: required changes outside the inference core

The inference core (`inference/`) is implemented against the target contracts below.
All items from the original list are done; the notes at the end describe known
limitations and what must change before a server deployment.

## Done
- **`load_image`** (now `algo/image_loading.py`, shared by training and inference) reads
  the image directly from its original location with Pillow. It copies nothing to disk.
  - Grayscale images keep their raw dtype; for example, a 16-bit TIFF stays uint16.
  - Colour and palette images are converted to 8-bit grayscale (`"L"`).
- **`save_image`** (`inference/image_io.py`) clips the float32 image to [0, 1], scales it
  to uint16 and writes a 16-bit TIFF (mode `I;16`) with Pillow.
  - An existing file is overwritten.
  - The output folder must already exist; the UI creates it.
- **1. Checkpoint stores model hyperparameters.** `SPFITrainer._save_checkpoint`
  (`algo/train.py`) also saves `"hparams": {"scale": SCALE, "embed_dim": EMBED_DIM}`.
  Old checkpoints have no `hparams` and are rejected with `ModelLoadError`; they must be
  retrained (no conversion script in the repo). `_load_model` also requires both values
  to be ints (not bool).
- **2. `normalize_image`** (`algo/preprocessing.py`) maps uint8 / 255 and uint16 / 65535
  to a 2D float32 array in [0, 1]. Any other dtype or a non-2D input raises `ValueError`
  (surfacing as `ImageLoadError` through the dataset).
- **3. Training uses `normalize_image`.** `EMPSDataset._load_gray_tensor` runs
  `load_image -> normalize_image -> to_tensor`, exactly like inference. A scan of the EMPS
  training images found only RGB PNGs (-> uint8 after `"L"`), so training input is unchanged.
- **4. Bit depth: decided — normalization by dtype.** See the limitation below.
- **5. `resolve_inputs`** (`inference/image_io.py`): a file yields `[path]`; a folder yields
  the files directly inside it (no recursion), with a supported extension (case-insensitive),
  skipping hidden files, in natural order (`img2` before `img10`).
- **6. Repo root importable.** `ui/app.py` puts the repo root on `sys.path` before any other
  import; spawned processes (background job, DataLoader workers) inherit it.
- **7. Unexpected errors.** The Inference tab's job wrapper (`ui/spfi_ui/inference/job.py::run_job`)
  catches any exception from `start_inference`, logs it at ERROR with traceback and returns a
  FAILED state, so the UI always gets a terminal state.
- **8. `pytorch_wavelets`** (and `PyWavelets`) are listed in `requirements.txt`.

## How the Inference tab runs a job
- Run / Continue is one Dash background callback (`DiskcacheManager`, cache in `.cache/dash`).
  The main app process never loads torch or the model; that happens only in the job process.
- Run saves the uploads to `tmp/inference/<job_id>/input/`, results go to `.../output/`,
  and "Download results" zips the `*_sr.tif` files.
- Stop sets a per-job flag in the diskcache; the job stops after its current image
  (CANCELLED), and Continue resumes the same job from `next_index`.

## Notes and known limitations
- **12-bit data in uint16 appears dark.** Normalization is by dtype, so 12-bit SEM data
  stored in uint16 uses only 1/16 of the range. Revisit if such data is used.
- **Training vs inference runners differ.** Training uses a thread-based runner in the Dash
  process; inference uses a background callback. When moving to a shared server, training
  must migrate to the same model.
- **Uploads are held in memory** by `dcc.Upload` (base64 in the browser and in the request).
  For the server phase, replace it with a chunked upload to disk.
- **Uploads are copied to `tmp/`** because the server has no other access to the user's files.
  This is not the "no copying" case of `resolve_inputs`, which still reads from where it is given.
- **Cleanup** of `tmp/inference/` runs at the start of each Run (24 h retention, never the
  current job). A server deployment may need a scheduled cleanup instead.
- **Before a server deployment:** Continue and Download take `run_dir` from the job store in
  the browser. They must instead derive `run_dir` from a validated `job_id`
  (`INFERENCE_RUNS_DIR / job_id`, checked to be a uuid4 hex that exists).
- **For the validation owner** (the UI does no input validation yet):
  - Ends as FAILED: missing model path, model path that is not a checkpoint, a checkpoint
    without `hparams` / with a different architecture, no uploaded files, an unreadable image.
  - Silently succeeds: "Single image" mode with several files uploaded — all of them are processed.
  - The "Destination directory" field is not used by inference; results are downloaded as a zip.
