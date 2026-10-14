"""
Mixing modules for the Scatter Pre-process Module (SPM).

Paper: "Lightweight super-resolution method based on scattering processing
and feature interaction", Scientific Reports (2026) 16:15018,
https://doi.org/10.1038/s41598-026-44351-5
See section "Scatter pre-process module" (Fig. 2, Fig. 3, Algorithm 1).

This file implements both mixing paths of SPM: TMM (Tensor Mixing Method) for
the low-frequency band, and EMM (Einstein Mixing Method) for the high-frequency
band. Both now receive the real DTCWT output (issue #35) through
`dtcwt.DTCWT` in spm.py: TMM gets the real lowpass XL, EMM gets the complex
finest-level highpass XH.

EMM below is Shoshana Pinski's delivered implementation, unchanged. TMM was
missing from the delivery (named in __all__ and imported by spm.py, but never
defined, so spm.py could not be imported); it was reconstructed during Layer 6
integration strictly from her design document and her own tests. See the
provenance note on the TMM class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TMM", "EMM"]


class TMM(nn.Module):
    """Tensor Mixing Method (TMM) — low-frequency mixing path of SPM.

    Equation used here:
        XPL[b, c, h, w] = w[c] * XL[b, c, h, w]

    One learnable scalar per channel, shared across all spatial positions.
    Each SPM owns its own TMM, so the total TMM parameter count of the model
    is C * (number of SPMs), e.g. C * 8.

    ------------------------------------------------------------------
    PROVENANCE — read before changing this class.

    The paper (and Shoshana Pinski's design document,
    `docs/SPFI_DESIGN_shoshana.md`, section 1) specifies a full Hadamard
    product, XPL = W_theta (*) XL, with W_theta the same shape as XL
    ([C, H, W], stored at a base size [H0, W0] and resized at forward time).
    That version was disabled during investigation and then replaced by the
    task owner's decision with this per-channel scalar version: C parameters
    instead of C * H0 * W0, and no resizing, so any input size works as is.
    This is a deliberate deviation from the paper.

    Weight init: all ones, so the module starts transparent (XPL == XL) and
    learns per-channel scaling from there (task owner's decision; NOT Xavier,
    whose variance-preservation argument is built for operations that sum
    over an axis, which an element-wise product does not do).
    ------------------------------------------------------------------

    Args:
        channels: C, number of feature channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels

        # One scalar per channel, initialised to 1.0 (see class docstring).
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, low-frequency band XL, real-valued ``[B, C, H, W]``.

        Returns: XPL, real-valued ``[B, C, H, W]`` (input shape preserved).
        """
        if x.dim() != 4:
            raise ValueError(
                f"TMM expects a 4D input [B,C,H,W], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TMM was built for channels={self.channels}, got input with C={x.shape[1]}"
            )
        return x * self.weight.view(1, -1, 1, 1)


class EMM(nn.Module):
    """
    Einstein Mixing Method (EMM) — high-frequency mixing path of SPM.

    Paper equation (Algorithm 1):
        Y[..., cb, e] = Σ_d  X[..., cb, d] · W[cb, d, e]
    i.e. channels are grouped, and a small learnable [Cb, Cd, Cd] weight
    mixes only within each group independently (never across groups,
    never across spatial positions) — far cheaper than mixing all
    channels together directly.

    ------------------------------------------------------------------
    Design decisions (see AMAT3_L6_SPFI_SPM.ipynb + SPFI_DESIGN.md for
    the full discussion, including the earlier design this superseded):

    Shape convention (CONFIRMED against the real DTCWT implementation,
    via its author, in response to 3 questions about channel
    ordering/grouping):
        Input/output are `[B, C, 6, H, W]`, complex-valued (torch complex
        dtype) — B=batch, C=original feature channels going into the
        DTCWT, H,W=spatial size of the finest DTCWT level (half the SPM
        input size), 6=directional orientations (this "6" is the paper's k;
        there is no separate real/imag axis of size 2, since real+imag
        are packed into the complex dtype instead of two stacked real
        channels).

    Grouping (SUPERSEDED an earlier design that assumed a flat,
    unstructured C and picked an arbitrary `desired_cd`, auto-adjusted
    to the nearest divisor of C):
        The DTCWT's own output is already structured as exactly the
        `(Cb, Cd)` shape EMM needs: the 6 orientations of one original
        channel are meaningfully related (they're all directional
        decompositions of the same feature/scale) and should be mixed
        together, while different original channels are NOT related and
        must never be mixed. There is no meaningful order within the 6
        orientations, and no natural grouping of any other size.
        - `group_multiplier` (default 1): how many whole original
          channels' 6-orientation sets are merged into one EMM group.
          `Cd = 6 * group_multiplier`, `Cb = C // group_multiplier`.
          Default 1 means Cb=C, Cd=6 exactly (the DTCWT's natural
          structure, untouched).
        - Growth path if trained model quality isn't sufficient:
          increase `group_multiplier` (2, 3, ...) — this gives each
          mixing unit more to see (multiple whole channels' orientation
          sets together) WITHOUT EVER splitting a single channel's 6
          orientations across a group boundary, which the DTCWT
          author's answer specifically warned against.
        - Requires `channels % group_multiplier == 0` (raises a clear
          error otherwise) — channels are grouped in contiguous blocks
          of `group_multiplier`.

    Complex-number handling (NEW decision, not previously discussed —
    surfaced once the real DTCWT's complex-dtype output was known):
        `complex_mode` constructor argument, a switch between two
        strategies:
        - `"independent"` (DEFAULT): treat the real and imaginary parts
          as two separate real tensors, run the SAME real-valued
          [Cb,Cd,Cd] weight on each independently (no cross terms), then
          recombine into a complex tensor. Matches the paper's literal
          formulation exactly — the paper's own "×2" factor is a leading
          broadcast axis using one shared real weight, not a described
          complex-valued multiplication. Cost: 2x the multiplications of
          a single real EMM application; 1x the parameters (shared W).
        - `"full_complex"`: genuine complex-valued linear mixing, with
          two independent real weight matrices (Wr, Wi) and cross terms
          (`Yr = Wr·Xr − Wi·Xi`, `Yi = Wr·Xi + Wi·Xr`). This is a
          deviation beyond what the paper describes — it can represent
          phase rotations (mixing real and imaginary together), which
          "independent" structurally cannot. No empirical evidence yet
          (from the paper or otherwise) that this helps for this
          architecture — offered as a documented, ready-to-use avenue
          for future experimentation, since the task owner is treating
          the paper as a starting point rather than an endpoint. Cost:
          4x the multiplications and 2x the parameters of a single real
          EMM application (i.e. exactly 2x the cost of "independent").

    Weight initialization:
        Xavier uniform, exactly as specified by the paper's Algorithm 1
        (not a free design choice) — applied per-group, each `[Cd,Cd]`
        slice initialized independently. In `"full_complex"` mode, Wr
        and Wi are each initialized this way, independently.
    ------------------------------------------------------------------

    Args:
        channels: C, number of original feature channels (before the
            DTCWT added the orientation axis).
        group_multiplier: how many whole channels' orientation-sets to
            merge into one mixing group (default 1 — no merging, use
            the DTCWT's natural Cb=C, Cd=6 structure directly).
        complex_mode: "independent" (default) or "full_complex" — see
            above.
    """

    ORIENTATIONS = 6  # fixed by the DTCWT itself (this is the paper's k)

    def __init__(
        self,
        channels: int,
        group_multiplier: int = 1,
        complex_mode: str = "independent",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if group_multiplier <= 0:
            raise ValueError(f"group_multiplier must be positive, got {group_multiplier}")
        if channels % group_multiplier != 0:
            raise ValueError(
                f"channels={channels} must be divisible by group_multiplier="
                f"{group_multiplier} (whole original channels are merged per group, "
                f"never split)"
            )
        if complex_mode not in ("independent", "full_complex"):
            raise ValueError(
                f'complex_mode must be "independent" or "full_complex", got {complex_mode!r}'
            )

        self.channels = channels
        self.group_multiplier = group_multiplier
        self.cb = channels // group_multiplier
        self.cd = self.ORIENTATIONS * group_multiplier
        self.complex_mode = complex_mode

        def _xavier_weight() -> nn.Parameter:
            w = torch.empty(self.cb, self.cd, self.cd)
            for g in range(self.cb):
                nn.init.xavier_uniform_(w[g])
            return nn.Parameter(w)

        self.weight_real = _xavier_weight()
        if complex_mode == "full_complex":
            self.weight_imag = _xavier_weight()

    def _mix(self, x_grouped: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Y[b,g,e,h,w] = sum_d X[b,g,d,h,w] * W[g,d,e] — a single real mix."""
        return torch.einsum("bgdhw,gde->bgehw", x_grouped, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: high-frequency feature XH, complex-valued, shape
                [B, C, 6, H, W] (finest level as returned by `dtcwt.DTCWT`;
                H, W are half the SPM input size). Any H, W is accepted.

        Returns:
            XPH, complex-valued, shape [B, C, 6, H, W] (identical to the
            input shape).
        """
        if x.dim() != 5:
            raise ValueError(
                f"EMM expects a 5D input [B,C,6,H,W], got shape {tuple(x.shape)}"
            )
        b, c, orientations, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"EMM was built for channels={self.channels}, got input with C={c}"
            )
        if orientations != self.ORIENTATIONS:
            raise ValueError(
                f"EMM expects {self.ORIENTATIONS} orientations (fixed by the DTCWT), "
                f"got {orientations}"
            )
        if not torch.is_complex(x):
            raise ValueError("EMM expects a complex-valued input tensor (DTCWT output)")

        # Flat reshape [B,C,6,H,W] -> [B,Cb,Cd,H,W]: valid directly because
        # C*6 == Cb*Cd, and grouping "group_multiplier whole channels'
        # 6-orientation sets together" is exactly what this reshape does
        # (see class docstring / SPFI_DESIGN.md for the index-level proof).
        x_grouped = x.reshape(b, self.cb, self.cd, h, w)
        xr, xi = x_grouped.real, x_grouped.imag

        if self.complex_mode == "independent":
            yr = self._mix(xr, self.weight_real)
            yi = self._mix(xi, self.weight_real)  # SAME shared weight, no cross terms
        else:  # "full_complex"
            yr = self._mix(xr, self.weight_real) - self._mix(xi, self.weight_imag)
            yi = self._mix(xi, self.weight_real) + self._mix(xr, self.weight_imag)

        y_grouped = torch.complex(yr, yi)
        return y_grouped.reshape(b, c, orientations, h, w)
