"""Compare CTI "K" settings: the common K/V resolution before fusion.

CTI keeps its 3 branches (3x3/s1, 5x5/s2, 7x7/s4). "K" is the size every
branch is resized to (bilinear) before the 1x1 fusion, as a reduction factor
of the input:  K2 -> H/2 x W/2  (32x32 for a 64x64 LR patch),
               K4 -> H/4 x W/4  (16x16).
Q keeps all H*W tokens; attention runs against the smaller K/V.

For each config the SPFI x4 model is built with that K and:
  * params + forward GFLOPs on a [1,1,64,64] LR input (exact, torch FlopCounterMode)
  * latency of one forward pass (on the chosen device)
  * PSNR / SSIM on a fixed EMPS validation set, checked every --eval_every
    iterations of an identical training budget (same seed, same data order,
    same init RNG for all configs). The table shows the best check and the last
    one: with batch size 1 and a constant LR the last step alone is noisy.

Usage
-----
    python -m algo.compare_cti_scales --emps_dir <EMPS root> --iters 300
    python -m algo.compare_cti_scales --emps_dir <EMPS root> --iters 0   # cost only

Results are printed as a table and written to <out_dir>/results.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .model import build_spfi
from .train import (EMBED_DIM, LAMBDA_SSIM, LR_PATCH, SCALE, EMPSDataset, charbonnier,
                    psnr, ssim_loss)

# name -> K (CTI kv_downsample). Everything else in the model is identical.
CONFIGS = {
    "K2": 2,
    "K4": 4,
}


def describe(k: int) -> str:
    side = -(-LR_PATCH // k)
    return f"{side}x{side}"


def build(k: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_spfi(scale=SCALE, embed_dim=EMBED_DIM, cti_kwargs={"kv_downsample": k})


def cost(model: torch.nn.Module, device: torch.device) -> dict:
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.zeros(1, 1, LR_PATCH, LR_PATCH, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    with torch.no_grad():
        model(x)  # warm-up
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model(x)
        _sync(device)
        latency = (time.perf_counter() - t0) / 3
    model.train()
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "gflops": counter.get_total_flops() / 1e9,
        "latency_ms": latency * 1e3,
    }


@torch.no_grad()
def evaluate(model, val_set, device: torch.device) -> tuple[float, float]:
    model.eval()
    p = s = 0.0
    for lr, hr in val_set:
        sr = model(lr.unsqueeze(0).to(device)).clamp(0, 1)
        hr = hr.unsqueeze(0).to(device)
        p += psnr(sr, hr)
        s += 1.0 - ssim_loss(sr, hr).item()
    model.train()
    return p / len(val_set), s / len(val_set)


def train(model, train_ds, iters: int, seed: int, lr: float, log_every: int, name: str,
          device: torch.device, val_set=None, eval_every: int = 0,
          best_path: Path | None = None) -> tuple[float, list[dict]]:
    """Train in place. Returns (sec_per_iter, history of {"iter", "psnr", "ssim"} checks).

    If best_path is given, the weights of the check with the highest PSNR are saved there.
    """
    history: list[dict] = []
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = random.Random(seed + 1)
    model.train()
    t0 = time.perf_counter()
    for it in range(1, iters + 1):
        lr_p, hr_p = train_ds.sample(rng)
        sr = model(lr_p.unsqueeze(0).to(device))
        hr = hr_p.unsqueeze(0).to(device)
        loss = charbonnier(sr, hr) + LAMBDA_SSIM * ssim_loss(sr, hr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  [{name}] iter {it}/{iters}  loss={loss.item():.4f}  {el / it:.2f}s/it", flush=True)
        if val_set is not None and eval_every > 0 and (it % eval_every == 0 or it == iters):
            p, s = evaluate(model, val_set, device)
            if best_path is not None and all(p > h["psnr"] for h in history):
                torch.save(model.state_dict(), best_path)
            history.append({"iter": it, "psnr": p, "ssim": s})
            print(f"  [{name}] eval @ {it}: PSNR={p:.3f} dB  SSIM={s:.4f}", flush=True)
    return (time.perf_counter() - t0) / max(iters, 1), history


def load_data(emps_dir: Path, seed: int, n_val: int, all_images: bool = False):
    """Training set and the fixed validation patches, the same for every config of one seed.

    Returns (train_ds, val_set), val_set being a list of (lr [1,64,64], hr [1,256,256]) pairs.
    """
    image_dir = emps_dir / "images"
    files = sorted(p.stem for p in image_dir.glob("*.png"))
    train_csv = emps_dir / "train.csv"
    if train_csv.is_file() and not all_images:
        # Same protocol as notebooks/train_colab.ipynb: only EMPS's train split is used
        # (split 80/20 into train / validation here); test.csv images are never touched.
        train_ids = {line.split(",")[0].strip()
                     for line in train_csv.read_text().splitlines() if line.strip()}
        files = [f for f in files if f in train_ids]
        print(f"using EMPS train split only: {len(files)} images (test.csv held out)")
    else:
        print(f"WARNING: using all {len(files)} images in {image_dir} (no train.csv split)")
    random.Random(seed).shuffle(files)
    n_train = int(0.8 * len(files))
    train_ds = EMPSDataset(image_dir, files[:n_train])
    val_ds = EMPSDataset(image_dir, files[n_train:])
    vrng = random.Random(42)
    return train_ds, [val_ds.sample(vrng) for _ in range(n_val)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emps_dir", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=300, help="training iterations per config (0 = cost only)")
    ap.add_argument("--n_val", type=int, default=32, help="fixed validation patches")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=500,
                    help="score PSNR/SSIM every N iterations; the table reports the best check and the last")
    ap.add_argument("--all_images", action="store_true",
                    help="use every image in images/, including EMPS's test split (not recommended)")
    ap.add_argument("--device",type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out_dir", type=Path, default=Path("runs") / "cti_scales_compare")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    # Keep results of configs not run this time (same iters / n_val / seed), so configs can be
    # run one at a time into the same out_dir, e.g. after a disconnect.
    results = {}
    results_path = args.out_dir / "results.json"
    if results_path.is_file():
        try:
            old = json.loads(results_path.read_text())
        except ValueError:
            old = {}
        if (old.get("iters"), old.get("n_val"), old.get("seed")) == (args.iters, args.n_val, args.seed):
            results = {k: v for k, v in old.get("results", {}).items()
                       if k != "bicubic" and k not in args.configs}
            if results:
                print(f"keeping earlier results from {results_path}: {', '.join(results)}")
    train_ds = val_set = None
    if args.iters > 0:
        train_ds, val_set = load_data(args.emps_dir, args.seed, args.n_val, args.all_images)
        bic = [psnr(F.interpolate(lr.unsqueeze(0), scale_factor=SCALE, mode="bicubic",
                                  align_corners=False).clamp(0, 1), hr.unsqueeze(0))
               for lr, hr in val_set]
        results["bicubic"] = {"psnr": sum(bic) / len(bic)}
        print(f"bicubic PSNR on {args.n_val} val patches: {results['bicubic']['psnr']:.2f} dB")

    for name in args.configs:
        k = CONFIGS[name]
        model = build(k, args.seed).to(device)
        r = {"k": k, "kv_size": describe(k), **cost(model, device)}
        print(f"{name}: K/V {r['kv_size']} (input {LR_PATCH}x{LR_PATCH})  params={r['params']:,}  "
              f"GFLOPs={r['gflops']:.2f}  fwd={r['latency_ms']:.0f} ms", flush=True)
        if args.iters > 0:
            r["sec_per_iter"], r["history"] = train(model, train_ds, args.iters, args.seed, args.lr,
                                                    args.log_every, name, device,
                                                    val_set=val_set, eval_every=args.eval_every or args.iters,
                                                    best_path=args.out_dir / f"{name}_best.pt")
            best = max(r["history"], key=lambda h: h["psnr"])
            last = r["history"][-1]
            r["psnr"], r["ssim"], r["best_iter"] = best["psnr"], best["ssim"], best["iter"]
            r["psnr_last"], r["ssim_last"] = last["psnr"], last["ssim"]
            print(f"  -> {name}: best PSNR={r['psnr']:.3f} dB  SSIM={r['ssim']:.4f} (iter {best['iter']})"
                  f"  | last PSNR={r['psnr_last']:.3f}  SSIM={r['ssim_last']:.4f}", flush=True)
            torch.save(model.state_dict(), args.out_dir / f"{name.split()[0]}.pt")
        results[name] = r
        (args.out_dir / "results.json").write_text(json.dumps(
            {"iters": args.iters, "n_val": args.n_val, "seed": args.seed, "results": results}, indent=2))

    print("\n| config | K/V size | params | GFLOPs | fwd ms "
          "| best PSNR | best SSIM | @iter | last PSNR | last SSIM |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(k for k in results if k != "bicubic"):
        r = results[name]

        def f(key: str, fmt: str) -> str:
            return format(r[key], fmt) if key in r else "-"

        print(f"| {name} | {r['kv_size']} | {r['params']:,} | {r['gflops']:.2f} | {r['latency_ms']:.0f} "
              f"| {f('psnr', '.3f')} | {f('ssim', '.4f')} | {f('best_iter', 'd')} "
              f"| {f('psnr_last', '.3f')} | {f('ssim_last', '.4f')} |")
    if "bicubic" in results:
        print(f"bicubic PSNR (same val patches): {results['bicubic']['psnr']:.3f} dB")


if __name__ == "__main__":
    main()
