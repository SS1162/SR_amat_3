"""RETIRED -- temporary identity stubs for what were once unmerged SPFI dependencies.

STATUS: no longer wired into anything. SPM (#36) and CTI (#37) landed, and
``sci.py`` now imports them directly with no fallback. These classes are kept
only as a record of the shell-first integration approach; importing them into
the model path again would silently disable SPM/CTI, so do not.

Owner branch: amat-3/riki/spfi-sci-model (issue #38)

The SCI + model shell has to run end-to-end *before* the DTCWT (#35), SPM (#36),
and CTI (#37) feature branches merge. These stubs are the explicit stand-ins the
#38 brief asks for -- "use explicit temporary identity/stub modules so the shell
can run end-to-end; remove or replace them during final integration".

They are shape-preserving and differentiable, and they do nothing else. They must
be gone before Layer 6 is called done:

    grep -rn "stubs import" code/amat-3/src   # must be empty at integration

See docs/AMAT3_SPFI_IMPLEMENTATION_DECISIONS.md section
"Riki -- SCI and SPFI model shell (issue #38)" -> "Temporary stub modules".
"""

from __future__ import annotations

import torch
from torch import nn


class IdentitySPM(nn.Module):
    """Stand-in for ``spm.SPM`` (#36).

    Real SPM decomposes the feature map with a DTCWT, mixes the low-frequency
    part (TMM) and the high-frequency part (EMM), and recombines. Until #36
    merges, this returns the input unchanged so the SCI layer around it is still
    exercised.

    Shape: ``[B, C, H, W] -> [B, C, H, W]``.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - permissive on purpose
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityCTI(nn.Module):
    """Stand-in for ``cti.CTI`` (#37).

    Real CTI builds multi-scale ``K``/``V`` with depth-wise separable
    convolutions at several kernel sizes and strides, then fuses across scales
    (paper Eq. (7)-(8)). Until #37 merges, this returns ``(x, x)`` -- i.e. ``K``
    and ``V`` are just the input feature map -- so the SCI attention has a real
    K/V path to consume.

    Shape: ``[B, C, H, W] -> (K, V)`` each ``[B, C, H, W]``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x
