"""Compare CTI multi-scale configurations (number of scale levels S).

For each config the SPFI x4 model is built with the given CTI levels and:
  * params + forward GFLOPs on a [1,1,64,64] LR input (exact, torch FlopCounterMode)
  * latency of one forward pass (on the chosen device)
  * PSNR / SSIM on a fixed EMPS validation set after an identical, short
    training budget (same seed, same data order, same init RNG for all configs)

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

from .cti import ScaleConfig
from .model import build_spfi
from .train import (EMBED_DIM, LAMBDA_SSIM, LR_PATCH, SCALE, EMPSDataset, charbonnier,
                    psnr, ssim_loss)

# Level i uses kernel 2i+1 and stride 2**i: wider receptive field and coarser
# resolution at every level, extending the CTI default (3/1, 5/2, 7/4).
LEVELS = [ScaleConfig(3, 1), ScaleConfig(5, 2), ScaleConfig(7, 4), ScaleConfig(9, 8)]

CONFIGS = {
    "K2": LEVELS[:2],
    "K3 (default)": LEVELS[:3],
    "K4": LEVELS[:4],
}


def describe(scales) -> str:
    return ", ".join(f"k{s.kernel_size}/s{s.stride}" for s in scales)


def build(scales, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_spfi(scale=SCALE, embed_dim=EMBED_DIM, cti_scales=scales)


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
          device: torch.device) -> float:
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
    return (time.perf_counter() - t0) / max(iters, 1)


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
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out_dir", type=Path, default=Path("runs") / "cti_scales_compare")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    results = {}
    train_ds = val_set = None
    if args.iters > 0:
        image_dir = args.emps_dir / "images"
        files = sorted(p.stem for p in image_dir.glob("*.png"))
        random.Random(args.seed).shuffle(files)
        n_train = int(0.8 * len(files))
        train_ds = EMPSDataset(image_dir, files[:n_train])
        val_ds = EMPSDataset(image_dir, files[n_train:])
        vrng = random.Random(42)
        val_set = [val_ds.sample(vrng) for _ in range(args.n_val)]
        bic = [psnr(F.interpolate(lr.unsqueeze(0), scale_factor=SCALE, mode="bicubic",
                                  align_corners=False).clamp(0, 1), hr.unsqueeze(0))
               for lr, hr in val_set]
        results["bicubic"] = {"psnr": sum(bic) / len(bic)}
        print(f"bicubic PSNR on {args.n_val} val patches: {results['bicubic']['psnr']:.2f} dB")

    for name in args.configs:
        scales = CONFIGS[name]
        model = build(scales, args.seed).to(device)
        r = {"levels": describe(scales), **cost(model, device)}
        print(f"{name}: levels [{r['levels']}]  params={r['params']:,}  "
              f"GFLOPs={r['gflops']:.2f}  fwd={r['latency_ms']:.0f} ms", flush=True)
        if args.iters > 0:
            r["sec_per_iter"] = train(model, train_ds, args.iters, args.seed, args.lr, args.log_every, name, device)
            r["psnr"], r["ssim"] = evaluate(model, val_set, device)
            print(f"  -> {name}: PSNR={r['psnr']:.3f} dB  SSIM={r['ssim']:.4f}", flush=True)
            torch.save(model.state_dict(), args.out_dir / f"{name.split()[0]}.pt")
        results[name] = r
        (args.out_dir / "results.json").write_text(json.dumps(
            {"iters": args.iters, "n_val": args.n_val, "seed": args.seed, "results": results}, indent=2))

    print("\n| config | levels (kernel/stride) | params | GFLOPs | fwd ms | PSNR | SSIM |")
    print("|---|---|---|---|---|---|---|")
    for name in args.configs:
        r = results[name]
        ps = f"{r['psnr']:.3f}" if "psnr" in r else "-"
        ss = f"{r['ssim']:.4f}" if "ssim" in r else "-"
        print(f"| {name} | {r['levels']} | {r['params']:,} | {r['gflops']:.2f} | "
              f"{r['latency_ms']:.0f} | {ps} | {ss} |")


if __name__ == "__main__":
    main()
