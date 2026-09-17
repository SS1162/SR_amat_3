"""DTCWT transform adapter for AMAT-3 SPFI.

Conventions documented here (forward output):
- Input feature tensor: x shape [N, C, H, W], float, CPU/GPU.
- Low-frequency coeffs: low shape [N, C, H_low, W_low].
- High-frequency coeffs per level: high[level] shape [N, C, 6, H_l, W_l], complex.
- Level order: high[0] is the finest level, high[J-1] is the coarsest.
- Orientation index order: exactly the backend's DTCWT orientation order (6 bands).
- Complex representation:
  - Public API: torch complex tensors (dtype complex64/complex128).
  - Backend API: real/imag pair in trailing dimension (..., 2).
- Boundary/padding mode: `symmetric` (default), configurable via `mode`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch import nn

PAPER_DEVIATIONS = (
    "Uses pytorch_wavelets DTCWT filters (biort='near_sym_b', qshift='qshift_b') unless overridden.",
    "Uses symmetric padding boundary mode.",
    "Stores high-frequency coefficients as torch complex tensors in API; backend uses real/imag pair.",
    "Implements DTCWT stage only (not SPM/CTI/full SPFI assembly).",
)


@dataclass(frozen=True)
class DTCWTCoefficients:
    low: Tensor
    high: Tuple[Tensor, ...]  # each: [N, C, 6, H_l, W_l], complex
    mode: str
    biort: str
    qshift: str

    @property
    def levels(self) -> int:
        return len(self.high)


class DTCWTTransform(nn.Module):
    def __init__(
        self,
        levels: int = 2,
        *,
        mode: str = "symmetric",
        biort: str = "near_sym_b",
        qshift: str = "qshift_b",
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = levels
        self.mode = mode
        self.biort = biort
        self.qshift = qshift

        try:
            from pytorch_wavelets import DTCWTForward, DTCWTInverse
        except ImportError as exc:
            raise ImportError(
                "pytorch_wavelets is required for DTCWTTransform. "
                "Install with: pip install pytorch_wavelets"
            ) from exc

        self._forward = DTCWTForward(
            J=levels,
            biort=biort,
            qshift=qshift,
            mode=mode,
        )
        self._inverse = DTCWTInverse(
            biort=biort,
            qshift=qshift,
            mode=mode,
        )

    def forward(self, x: Tensor) -> DTCWTCoefficients:
        if x.ndim != 4:
            raise ValueError(f"expected x with shape [N, C, H, W], got {tuple(x.shape)}")

        low, high_ri = self._forward(x)
        high_complex = tuple(self._ri_to_complex(h) for h in high_ri)

        return DTCWTCoefficients(
            low=low,
            high=high_complex,
            mode=self.mode,
            biort=self.biort,
            qshift=self.qshift,
        )

    def inverse(self, coeffs: DTCWTCoefficients) -> Tensor:
        if len(coeffs.high) != self.levels:
            raise ValueError(
                f"expected {self.levels} levels, got {len(coeffs.high)}"
            )

        high_ri = [self._complex_to_ri(h) for h in coeffs.high]
        x_hat = self._inverse((coeffs.low, high_ri))
        return x_hat

    @staticmethod
    def _ri_to_complex(x_ri: Tensor) -> Tensor:
        if x_ri.shape[-1] != 2:
            raise ValueError(f"expected real/imag trailing dim=2, got shape {tuple(x_ri.shape)}")
        return torch.complex(x_ri[..., 0], x_ri[..., 1])

    @staticmethod
    def _complex_to_ri(x_complex: Tensor) -> Tensor:
        if not torch.is_complex(x_complex):
            raise ValueError("expected complex tensor for high-frequency coefficients")
        return torch.stack((x_complex.real, x_complex.imag), dim=-1)


def dtcwt(
    x: Tensor,
    *,
    levels: int = 2,
    mode: str = "symmetric",
    biort: str = "near_sym_b",
    qshift: str = "qshift_b",
) -> DTCWTCoefficients:
    return DTCWTTransform(levels=levels, mode=mode, biort=biort, qshift=qshift)(x)


def inverse_dtcwt(coeffs: DTCWTCoefficients) -> Tensor:
    transform = DTCWTTransform(
        levels=coeffs.levels,
        mode=coeffs.mode,
        biort=coeffs.biort,
        qshift=coeffs.qshift,
    )
    return transform.inverse(coeffs)


def reconstruction_error_stats(x: Tensor, x_hat: Tensor) -> Dict[str, float]:
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(x_hat.shape)}")
    diff = x_hat - x
    return {
        "max_abs": float(diff.abs().max().item()),
        "mae": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def coefficient_energy_maps(coeffs: DTCWTCoefficients) -> Dict[str, Tensor]:
    """Small reusable debug helper for notebook visualization."""
    maps: Dict[str, Tensor] = {
        "low_mean_abs": coeffs.low.abs().mean(dim=1)
    }
    for level_idx, h in enumerate(coeffs.high, start=1):
        amp = h.abs()
        for ori in range(amp.shape[2]):
            maps[f"high_l{level_idx}_ori{ori}_mean_abs"] = amp[:, :, ori].mean(dim=1)
    return maps


# =====================================================================
# Layer 6 integration seam
# =====================================================================
# Everything above is Miri Liebeskind's delivered DTCWT (issue #35),
# unchanged. It is the authoritative implementation.
#
# SPM (#36) was written against a simpler contract than the coefficients
# dataclass above: a `(low, high)` tuple and an `inverse(low, high)` call,
# with `high` being one complex band. The wrapper below adapts Miri's
# transform to that contract rather than editing either side, so neither
# student's module had to be rewritten to fit the other.
#
# Multi-level handling: SPM's EMM mixes ONE high-frequency band. With
# `levels > 1` the finest level is handed to SPM and the coarser levels are
# carried through the inverse untouched, so no information is silently
# dropped. That is a real integration decision, recorded here and in
# docs/AMAT3_SPFI_IMPLEMENTATION_DECISIONS.md (D-I2).

__all__ = [
    "DTCWTCoefficients",
    "DTCWTTransform",
    "DTCWT",
    "DTCWT_AVAILABLE",
    "DTCWTUnavailable",
    "dtcwt",
    "inverse_dtcwt",
    "reconstruction_error_stats",
    "coefficient_energy_maps",
    "PAPER_DEVIATIONS",
]

try:  # advertised so callers can degrade politely instead of crashing on import
    import pytorch_wavelets as _pw  # noqa: F401

    DTCWT_AVAILABLE = True
except ImportError:  # pragma: no cover
    DTCWT_AVAILABLE = False


class DTCWTUnavailable(RuntimeError):
    """Raised when the DTCWT backend is not installed."""


class DTCWT(nn.Module):
    """SPM-facing view of :class:`DTCWTTransform` (Miri's #35 implementation).

    ``forward(x) -> (low, high_finest)`` and ``inverse(low, high_finest) -> x``,
    which is the contract ``spm.SPM`` consumes:

    * ``low``  -- real ``[B, C, H_low, W_low]``, the input for ``TMM``
    * ``high`` -- complex ``[B, C, 6, H_l, W_l]``, the input for ``EMM``

    Coarser levels from the last forward pass are retained and re-inserted on
    the inverse. This makes the wrapper stateful between ``forward`` and
    ``inverse``, which is safe in SPM (they are adjacent in one forward pass)
    but is the reason this is a wrapper rather than a general-purpose API --
    use :class:`DTCWTTransform` directly for anything else.
    """

    def __init__(
        self,
        levels: int = 1,
        biort: str = "near_sym_b",
        qshift: str = "qshift_b",
        mode: str = "symmetric",
    ) -> None:
        super().__init__()
        if not DTCWT_AVAILABLE:
            raise DTCWTUnavailable(
                "pytorch_wavelets is required for the DTCWT path "
                "(pip install pytorch_wavelets)."
            )
        self.transform = DTCWTTransform(
            levels=levels, mode=mode, biort=biort, qshift=qshift
        )
        self.levels = levels
        self._coarse: Tuple[Tensor, ...] = ()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"DTCWT expects [B,C,H,W], got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            raise ValueError(
                f"DTCWT needs even spatial dimensions, got H={h}, W={w}. "
                "Pad or crop upstream rather than letting the transform guess."
            )
        coeffs = self.transform(x)
        self._coarse = coeffs.high[1:]
        self._meta = (coeffs.mode, coeffs.biort, coeffs.qshift)
        return coeffs.low, coeffs.high[0]

    def inverse(self, low: Tensor, high: Tensor) -> Tensor:
        if not torch.is_complex(high):
            raise ValueError("DTCWT.inverse expects a complex highpass tensor")
        mode, biort, qshift = getattr(self, "_meta", ("symmetric", "near_sym_b", "qshift_b"))
        coeffs = DTCWTCoefficients(
            low=low,
            high=(high, *self._coarse),
            mode=mode,
            biort=biort,
            qshift=qshift,
        )
        return self.transform.inverse(coeffs)
