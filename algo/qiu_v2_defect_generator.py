"""Qiu-v2 synthetic defect generator for grayscale SEM images, as one file.

Inserts new defects into defect-free images. Each new defect takes the shape of a real defect mask
(scaled and elastically deformed) and its pixels are generated, with no training, by the
single-image diffusion sampler of Qiu et al. from the patches of the real defect images.

Method: Haojun Qiu, Kiriakos N. Kutulakos, David B. Lindell, "Efficient and Training-Free
Single-Image Diffusion Models", CVPR 2026. Their official code (MIT License) is used unchanged,
installed as the package efficient_sid (see Install). This file adds: the abnormal patch bank,
filling only the defect region (RePaint-style blend), the mask and brightness channels, new
shapes, and the edge halo.

Install (Python 3.10+; a CUDA GPU is much faster, a CPU works for small runs):

    pip install torch numpy scipy pillow
    pip install "git+https://github.com/haojunqiu/efficient-SID-code"

    (no git: download https://github.com/haojunqiu/efficient-SID-code/archive/refs/heads/main.zip,
    unzip, and run `pip install ./efficient-SID-code-main`)

Input folders:
    defects/   real defect images (grayscale or RGB, read as gray)
    masks/     one binary mask per defect image, same file name stem (white = defect)
    normals/   defect-free images, the canvases the new defects go into

Usage:
    python qiu_v2_defect_generator.py --defects defects --masks masks --normals normals \
        --out synthetic --ratio 5

Writes <out>/images/*.png, <out>/masks/*.png (the new defect masks), synthetic.csv, samples.csv
(one row per attempt: source shape, canvas, position, contrast) and done.json.
"""
import argparse
import csv
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

try:
    import omegaconf  # noqa: F401
except ImportError:
    # efficient_sid only reads omegaconf.MISSING here, as a dataclass default.
    _omegaconf = types.ModuleType("omegaconf")
    _omegaconf.MISSING = "???"
    sys.modules["omegaconf"] = _omegaconf

from efficient_sid.closed_form_denoiser import ClosedFormDenoiserConfig, make_closed_form_denoiser  # noqa: E402
from efficient_sid.image_denoiser import ImageDenoiser, ImageDenoiserConfig  # noqa: E402
from efficient_sid.pyramid import PyramidConfig, PyramidProcessor  # noqa: E402
from efficient_sid.scheduler import FlowLinearConfig, FlowLinearScheduler  # noqa: E402
from efficient_sid.utils import (load_image_as_tensor, normalize_to_neg_one_to_one,  # noqa: E402
                                 unnormalize_to_zero_to_one)

# ---- Qiu et al. settings (configs/uncond/pixel_exact.yaml) ----
RELATIVE_SCALE = 0.5
MIN_SHORT_SIDE = 24
P = 15                     # patch size
DENOISER_TYPE = "exact"
BOTTLENECK_DTYPE = "float32"
NUM_STEPS = 10
SIGMA_MIN = 1e-4
ETA = 0.0
MAX_SCORE_BYTES = 128 * 2 ** 20   # peak query x dataset score matrix; chunking changes only memory

# ---- generator settings ----
OVERLAP_MIN = 0.5          # a bank patch is >= 50% defect (or centred on a defect smaller than a patch)
MASK_THRESHOLD = 127
MASK_WEIGHT = 1.0          # mask channel: MASK_WEIGHT * (+1 defect / -1 background)
MAX_START = 2              # never start coarser than this pyramid scale
MIN_PX = 4                 # start at the coarsest scale where the defect's short side is >= MIN_PX px
CAP = 200                  # per defect image, CAP patches also get their 8 flips/rotations
RATIO = 5                  # synthetic images per real defect image
FOLD_RHO = 0.1             # texture inside the defect
BRIGHT_WEIGHT = 1.0        # third channel: source contrast / 128 times this
DARK_MAX = 0.0             # defects with a lower contrast are not used as shapes (they stay in the bank)
SCALE_MIN, SCALE_MAX = 0.9, 1.3
ELASTIC = 0.3              # largest elastic shift, as a share of the shape's short side
THIN, THIN_LEN = 6.0, 30   # parts with half width <= THIN px and length >= THIN_LEN px (scratches): only scaled
OWN_IN_BANK = True         # the shape's own image stays in the bank (--no-own leaves it out)
EDGE_BLUR, EDGE_WEIGHT = 1.0, 1.0
SHADOW = 4.0               # gray levels, soft darker ring 2..4 px outside the edge
MIN_RESULT_CONTRAST = 20   # a fainter result is drawn again, at most MAX_TRIES times, else dropped
MAX_TRIES = 3
SEED = 20261018
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Qiu et al. building blocks ----------------

def num_scales(short_side):
    n = 1
    while MIN_SHORT_SIDE * 2 ** (n - 1) < short_side:
        n += 1
    return n


def pyramid_processor(n_scales):
    return PyramidProcessor(PyramidConfig(relative_scale=RELATIVE_SCALE, num_scales=n_scales,
                                          min_short_side=MIN_SHORT_SIDE))


def scheduler(device=None):
    return FlowLinearScheduler(FlowLinearConfig(num_steps=NUM_STEPS, sigma_min=SIGMA_MIN, eta=ETA), device)


def denoiser(dataset, num_queries):
    """Qiu's image denoiser over a patch dataset (N, D), on the dataset's device."""
    chunks = max(1, math.ceil(num_queries * len(dataset) * 4 / MAX_SCORE_BYTES))
    pc = ClosedFormDenoiserConfig(type=DENOISER_TYPE, bottleneck_dtype=BOTTLENECK_DTYPE, query_chunks=chunks)
    cfg = ImageDenoiserConfig(patch_size=P, dataset_stride=1, query_stride=1, fold_rho=FOLD_RHO, patch_denoiser=pc)
    return ImageDenoiser(cfg, make_closed_form_denoiser(pc, dataset), dtype=torch.float32).to(dataset.device)


# ---------------- abnormal patch bank ----------------

def mask_pyramid(processor, mask, n_scales):
    """A (H, W) bool mask downscaled like an image, one (h, w) map in [0, 1] per scale."""
    return [m[0].clamp(0, 1) for m in processor.build_gaussian_pyramid(mask.float()[None], num_scales=n_scales)]


def abnormal_positions(mask, patch_size, overlap_min):
    """Top-left corners of abnormal patches: the overlap rule, and patches centred on a small defect."""
    h, w = mask.shape
    covered = F.avg_pool2d(mask[None, None], patch_size, stride=1)[0, 0]
    overlap = (covered >= overlap_min).numpy()
    centred = np.zeros_like(overlap)
    labels, _ = ndimage.label(mask.numpy() >= 0.5)
    for rows, cols in ndimage.find_objects(labels):
        if rows.stop - rows.start > patch_size or cols.stop - cols.start > patch_size:
            continue
        cy, cx = (rows.start + rows.stop - 1) // 2, (cols.start + cols.stop - 1) // 2
        y = min(max(cy - patch_size // 2, 0), h - patch_size)
        x = min(max(cx - patch_size // 2, 0), w - patch_size)
        centred[y, x] = not overlap[y, x]
    return overlap, centred


def build_banks(samples, processor, n_scales, patch_size, overlap_min):
    """Abnormal patches of all (source id, (C, H, W) image in [-1, 1], (H, W) bool mask) at every scale."""
    parts = [{"patches": [], "source": []} for _ in range(n_scales)]
    for source, image, mask in samples:
        images = processor.build_gaussian_pyramid(image, num_scales=n_scales)
        masks = mask_pyramid(processor, mask, n_scales)
        for s in range(n_scales):
            overlap, centred = abnormal_positions(masks[s], patch_size, overlap_min)
            ys, xs = np.nonzero(overlap | centred)
            windows = images[s].unfold(1, patch_size, 1).unfold(2, patch_size, 1)
            patches = windows[:, torch.from_numpy(ys), torch.from_numpy(xs)].permute(1, 0, 2, 3)
            parts[s]["patches"].append(patches.flatten(1).clone())
            parts[s]["source"].append(np.full(len(ys), source))
    return [{"patches": torch.cat(p["patches"]), "source": np.concatenate(p["source"])} for p in parts]


# ---------------- shapes, contrast, halo ----------------

def gray(path):
    return np.asarray(Image.open(path).convert("L"))


def contrast(img, mask):
    """Mean inside the mask minus mean of the ring 3..10 px outside it (gray levels)."""
    ring = ndimage.binary_dilation(mask, iterations=10) & ~ndimage.binary_dilation(mask, iterations=3)
    return float(img[mask].mean() - img[ring].mean())


def crop(mask):
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def largest(mask):
    lab, n = ndimage.label(mask)
    return crop(lab == int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))) + 1)


def half_width(part):
    return float(ndimage.distance_transform_edt(np.pad(part, 1)).max())


def new_shape(mask, rng, max_side):
    """Every part of the mask in its real arrangement, scaled and elastically deformed; scratches only scaled."""
    m = np.pad(crop(mask), 8)
    m = ndimage.zoom(m.astype(float), rng.uniform(SCALE_MIN, SCALE_MAX), order=1) > 0.5
    short = min(largest(m).shape)
    lab, n = ndimage.label(m)
    straight = np.zeros_like(m)
    for i in range(1, n + 1):
        part = lab == i
        if part.sum() >= 30 and half_width(part) <= THIN and max(crop(part).shape) >= THIN_LEN:
            straight |= part
    sig = max(2.0, short / 4)
    field = [ndimage.gaussian_filter(rng.uniform(-1, 1, m.shape), sig) for _ in range(2)]
    amp = ELASTIC * short / max(np.abs(field[0]).max(), np.abs(field[1]).max(), 1e-9)
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]].astype(float)
    bent = ndimage.map_coordinates((m & ~straight).astype(float), [yy + amp * field[0], xx + amp * field[1]],
                                   order=1) > 0.5
    out = crop(bent | straight)
    if max(out.shape) > max_side:  # keep it inside the canvas
        out = crop(ndimage.zoom(out.astype(float), max_side / max(out.shape), order=1) > 0.5)
    return out


def halo(img, region):
    """Edge band blended to its blur, and a soft darker ring 2..4 px outside (real defects' edge profile)."""
    band = ndimage.binary_dilation(region, iterations=3) & ~ndimage.binary_erosion(region, iterations=2)
    img = img.copy()
    img[band] += EDGE_WEIGHT * (ndimage.gaussian_filter(img, EDGE_BLUR) - img)[band]
    d = ndimage.distance_transform_edt(~region)
    return img - SHADOW * np.exp(-((d - 3) / 1.2) ** 2) * (d > 1)


def dihedral(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4)


def with_mask(path, mask, c):
    """(3, H, W): the image in [-1, 1], its mask channel and the brightness channel (contrast c)."""
    img = normalize_to_neg_one_to_one(load_image_as_tensor(str(path)))[:1]
    return torch.cat([img, MASK_WEIGHT * (2 * torch.from_numpy(mask).float()[None] - 1),
                      torch.full_like(img, BRIGHT_WEIGHT * c / 128)])


def start_scale(shape, n_scales):
    short = min(shape.shape)
    return min(MAX_START, max(s for s in range(n_scales) if short / 2 ** s >= MIN_PX or s == 0))


def window(support, margin):
    """Bounds (y0, y1, x0, x1) of a (h, w) bool map, grown by margin and clipped to the image."""
    ys, xs = torch.nonzero(support, as_tuple=True)
    h, w = support.shape
    return (max(int(ys.min()) - margin, 0), min(int(ys.max()) + 1 + margin, h),
            max(int(xs.min()) - margin, 0), min(int(xs.max()) + 1 + margin, w))


# ---------------- sampler ----------------

@torch.inference_mode()
def fill(image, region, datasets, start, proc, sched):
    """Qiu's sampler, coarse to fine from scale start; after every step the pixels outside the region are
    reset to the image at that noise level (RePaint). Only channel 0 inside the region is generated."""
    n = len(datasets)
    images = proc.build_gaussian_pyramid(image, num_scales=n)
    regions = [r[0].clamp(0, 1) for r in proc.build_gaussian_pyramid(region.float()[None], num_scales=n)]
    xt = [sched.init_noise(im.shape, image.device, image.dtype) for im in images]
    noise = [x / sched.init_noise_sigma for x in xt]
    known = lambda s, t: sched.alphas[t] * images[s][None] + sched.sigmas[t] * noise[s]  # noqa: E731
    for s in reversed(range(n)):
        if s > start:
            xt[s] = images[s][None]
            continue
        m = torch.zeros_like(images[s])[None]
        m[:, 0] = regions[s]
        y0, y1, x0, x1 = window(regions[s] > 0, P + 8)
        den = denoiser(datasets[s], (y1 - y0 - P + 1) * (x1 - x0 - P + 1))
        for t in reversed(range(1, sched.num_steps + 1)):
            x_hat = images[s][None].clone()
            win = den(xt=xt[s][..., y0:y1, x0:x1].contiguous(), t=t, scheduler=sched)
            # with a small fold_rho the window's corner pixels can get ~0 weight (0/0); they lie outside the region
            x_hat[..., y0:y1, x0:x1] = torch.where(torch.isnan(win), x_hat[..., y0:y1, x0:x1], win)
            eps = sched.epsilon_from_x_hat(xt[s], x_hat, t)
            if s != start and t != 1:
                x_hat = proc.blend_two_scale(x_hat, xt[s + 1])
            xt[s] = sched.step(x_hat, eps, t, eta=ETA)
            xt[s] = m * xt[s] + (1 - m) * known(s, t - 1)
    return xt[0][0, 0]


# ---------------- main ----------------

def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)


def find_mask(mask_dir, stem):
    hits = [p for p in list_images(mask_dir) if p.stem == stem]
    if not hits:
        raise FileNotFoundError(f"no mask named {stem}.* in {mask_dir}")
    return hits[0]


def make(defect_dir, mask_dir, normal_dir, out, ratio, n_images=None):
    out = Path(out)
    if (out / "done.json").exists():
        print(f"already made: {out}")
        return
    torch.set_flush_denormal(True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    defects = [{"id": p.stem, "image": p} for p in list_images(defect_dir)]
    normals = list_images(normal_dir)
    if not defects or not normals:
        raise SystemExit("need at least one defect image (with mask) and one defect-free image")
    masks = {}
    for d in defects:
        masks[d["id"]] = gray(find_mask(mask_dir, d["id"])) > MASK_THRESHOLD
        if gray(d["image"]).shape != masks[d["id"]].shape:
            raise SystemExit(f"{d['id']}: image and mask sizes differ")
    defects = [d for d in defects if masks[d["id"]].any()]
    src_c = {d["id"]: contrast(gray(d["image"]).astype(np.float32), masks[d["id"]]) for d in defects}
    shapes = [d["id"] for d in defects if src_c[d["id"]] >= DARK_MAX]
    if not shapes:
        raise SystemExit("no defect is brighter than its surroundings (DARK_MAX); nothing to use as a shape")

    n_scales = MAX_START + 1
    short_side = min(min(gray(p).shape) for p in normals)
    proc, sched = pyramid_processor(num_scales(short_side)), scheduler(DEVICE)
    banks = build_banks(((i, with_mask(d["image"], masks[d["id"]], src_c[d["id"]]),
                          torch.from_numpy(masks[d["id"]])) for i, d in enumerate(defects)),
                        proc, n_scales, P, OVERLAP_MIN)
    rng = np.random.default_rng(SEED)
    bank = []
    for b in banks:
        keep = np.sort(np.concatenate([rng.permutation(np.flatnonzero(b["source"] == s))[:CAP]
                                       for s in np.unique(b["source"])]))
        q = b["patches"][torch.from_numpy(keep)].view(len(keep), 3, P, P)
        flips = torch.cat([torch.rot90(q.flip(-1) if k >= 4 else q, k % 4, (-2, -1)).reshape(len(q), -1)
                           for k in range(8)])
        rest = np.setdiff1d(np.arange(len(b["source"])), keep)  # every other patch, unflipped
        bank.append((torch.cat([flips, b["patches"][torch.from_numpy(rest)]]).to(DEVICE),
                     np.concatenate([np.tile(b["source"][keep], 8), b["source"][rest]])))
    src = {d["id"]: i for i, d in enumerate(defects)}
    n = n_images or ratio * len(defects)
    print(f"{len(defects)} defect ({len(shapes)} used as shapes) + {len(normals)} defect-free images; "
          f"bank patches per scale {[len(p) for p, _ in bank]}; making {n} images on {DEVICE}", flush=True)

    synthetic, samples, t0 = [], [], time.time()
    for i in range(n):
        d = shapes[i % len(shapes)]
        datasets = [p if OWN_IN_BANK else p[torch.from_numpy(s != src[d]).to(DEVICE)] for p, s in bank]
        for tries in range(1, MAX_TRIES + 1):
            canvas = normals[int(rng.integers(len(normals)))]
            H, W = gray(canvas).shape
            shape = dihedral(new_shape(masks[d], rng, min(H, W) - 12), int(rng.integers(8)))
            h, w = shape.shape
            top, left = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
            region = np.zeros((H, W), bool)
            region[top:top + h, left:left + w] = shape
            start = start_scale(largest(region), n_scales)
            torch.manual_seed(SEED + 7919 * i + tries)
            result = fill(with_mask(canvas, region, src_c[d]).to(DEVICE),
                          torch.from_numpy(region).to(DEVICE), datasets, start, proc, sched)
            img = halo(unnormalize_to_zero_to_one(result).cpu().numpy() * 255, region)
            img = img.round().clip(0, 255).astype(np.uint8)
            c = contrast(img.astype(np.float32), region)
            if c >= MIN_RESULT_CONTRAST:
                break
        row = {"i": i, "shape_source": d, "source_contrast": round(src_c[d], 1), "canvas": canvas.name,
               "top": top, "left": left, "area": int(region.sum()), "start_scale": start,
               "contrast": round(c, 1), "tries": tries, "kept": c >= MIN_RESULT_CONTRAST}
        samples.append(row)
        if row["kept"]:
            name = f"qiuv2_{i:04d}"
            Image.fromarray(img).save(out / "images" / f"{name}.png")
            Image.fromarray((region * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
            synthetic.append({"id": name, "image": f"images/{name}.png", "mask": f"masks/{name}.png"})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}  kept {len(synthetic)}  {time.time() - t0:.0f}s", flush=True)

    for fname, rows, fields in (("samples.csv", samples, list(samples[0])),
                                ("synthetic.csv", synthetic, ["id", "image", "mask"])):
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
    (out / "done.json").write_text(json.dumps({"made": n, "kept": len(synthetic), "fold_rho": FOLD_RHO,
                                               "bright_weight": BRIGHT_WEIGHT, "scale": [SCALE_MIN, SCALE_MAX],
                                               "elastic": ELASTIC, "own_in_bank": OWN_IN_BANK,
                                               "shadow": SHADOW, "bank_patches": [len(p) for p, _ in bank],
                                               "seconds": round(time.time() - t0)}, indent=2))
    print(f"kept {len(synthetic)} of {n} -> {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--defects", required=True, help="folder of real defect images")
    ap.add_argument("--masks", required=True, help="folder of their binary masks (same file name stem)")
    ap.add_argument("--normals", required=True, help="folder of defect-free images (canvases)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--ratio", type=int, default=RATIO, help="synthetic images per real defect image")
    ap.add_argument("--n", type=int, default=None, help="exact number of images to make (overrides --ratio)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-own", action="store_true", help="leave the shape's own image out of the bank")
    a = ap.parse_args()
    SEED, OWN_IN_BANK = a.seed, not a.no_own
    make(a.defects, a.masks, a.normals, a.out, a.ratio, a.n)
