"""
Scatter Pre-process Module (SPM) — assembles TMM and EMM around the
DTCWT scattering transform.

Paper: "Lightweight super-resolution method based on scattering processing
and feature interaction", Scientific Reports (2026) 16:15018,
https://doi.org/10.1038/s41598-026-44351-5
See section "Scatter pre-process module" (Fig. 2).

See SPFI_DESIGN.md and AMAT3_L6_SPFI_SPM.ipynb for the full design
discussion, decisions, and remaining uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dtcwt import DTCWT
from .mixing import TMM, EMM

__all__ = ["SPM"]


class SPM(nn.Module):
    """
    Scatter Pre-process Module (SPM).

    Flow (Fig. 2): input X -> DTCWT scattering -> (XL, XH) -> TMM(XL) ->
    XPL, EMM(XH) -> XPH -> DTCWT inverse(XPL, XPH) -> reconstructed ->
    + alpha * X (residual) -> output.

    ------------------------------------------------------------------
    Design decisions (see SPFI_DESIGN.md / notebook for full discussion):

    DTCWT integration (Layer 6): the real DTCWT (issue #35, Miri's
    implementation) is wired in through `dtcwt.DTCWT`, an SPM-facing
    wrapper around `DTCWTTransform` that exposes the `(low, high)` /
    `inverse(low, high)` contract this module was written against.
        - Requires `pytorch_wavelets`; constructing SPM without it raises
          `dtcwt.DTCWTUnavailable`.
        - Input H and W must be even (the wrapper raises otherwise rather
          than padding silently).
        - `dtcwt_levels` (default 1) sets the number of DTCWT levels. Only
          the finest high-frequency level goes through EMM; with
          `dtcwt_levels > 1` the coarser levels are stored by the wrapper
          during the forward transform and put back unchanged in the
          inverse, so no information is dropped.
        - Band shapes actually produced (input [B,C,H,W]):
            XL: real    [B,C,H,W] at 1 level, [B,C,H/2^(L-1),W/2^(L-1)]
                at L levels. TMM resizes its weight to match, so any
                size works.
            XH: complex [B,C,6,H/2,W/2] (finest level, any L).

    The "+ alpha * X" residual connection (ASSUMPTION — the paper's
    Fig. 2 diagram shows a "+" after "Inverse Scattering", but only two
    incoming arrows are visible there, from XPL and XPH; there is no
    clearly visible third arrow from the original input X. Adding a
    residual with X anyway was a deliberate choice, not a reading of
    the paper): the task owner chose to add a residual connection with
    the original input X regardless of what the paper's figure shows,
    since it is a well-established technique (better gradient flow,
    "only learn the difference from X" is often easier than learning
    the full transform from scratch) — this is EXPLICITLY a deviation/
    extension beyond the paper, not a literal reading of Fig. 2, and is
    documented as such.
        - `alpha` is a LEARNABLE scalar parameter (not fixed), initialized
          to 0.1 (small), so the residual doesn't dominate the output at
          the start of training but the network can adjust it as needed.
          Chosen over a fixed constant because the task owner is treating
          the paper as a starting point for further research, not a
          fixed target to reproduce exactly.

    Remaining uncertainty (for the PR description): whether the paper's
    own Fig. 2 "+" really is just XPL+XPH (no residual with X) is not
    100% certain without the original code — this SPM adds a residual
    with X regardless, as a deliberate design choice independent of that
    uncertainty.
    ------------------------------------------------------------------

    Args:
        channels: C, number of channels of the input feature.
        tmm_base_height: H0 for TMM's stored weight (see TMM docstring).
        tmm_base_width: W0 for TMM's stored weight.
        emm_group_multiplier: passed through to EMM (default 1: Cb=C, Cd=6).
        emm_complex_mode: passed through to EMM ("independent" or
            "full_complex").
        alpha_init: initial value of the learnable residual scale (default 0.1).
        dtcwt_levels: number of DTCWT decomposition levels (default 1); see
            "DTCWT integration" above.
    """

    def __init__(
        self,
        channels: int,
        tmm_base_height: int = 64,
        tmm_base_width: int = 64,
        emm_group_multiplier: int = 1,
        emm_complex_mode: str = "independent",
        alpha_init: float = 0.1,
        dtcwt_levels: int = 1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.tmm = TMM(channels=channels, base_height=tmm_base_height, base_width=tmm_base_width)
        self.emm = EMM(channels=channels, group_multiplier=emm_group_multiplier, complex_mode=emm_complex_mode)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # Real DTCWT (issue #35) via the SPM-facing wrapper in dtcwt.py.
        self.dtcwt = DTCWT(levels=dtcwt_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature, real-valued, shape [B, C, H, W], H and W even.

        Returns:
            output feature, real-valued, shape [B, C, H, W] (same shape
            as input — this is the task's gate).
        """
        if x.dim() != 4:
            raise ValueError(f"SPM expects a 4D input [B,C,H,W], got shape {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"SPM was built for channels={self.channels}, got input with C={c}")

        # DTCWT forward: XL real [B,C,H_low,W_low] for TMM, XH complex
        # [B,C,6,H/2,W/2] (finest level) for EMM. Coarser levels, if any,
        # are kept inside self.dtcwt for the inverse.
        XL, XH = self.dtcwt(x)

        XPL = self.tmm(XL)
        XPH = self.emm(XH)

        # DTCWT inverse: real [B,C,H,W], same shape as x.
        reconstructed = self.dtcwt.inverse(XPL, XPH)

        return reconstructed + self.alpha * x

