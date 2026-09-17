"""Top-level SPFI model shell.

Owner branch: amat-3/riki/spfi-sci-model (issue #38)

Paper: "Lightweight super-resolution method based on scattering processing and
feature interaction" (SPFI), Scientific Reports s41598-026-44351-5.

Pipeline (paper's four stages; see docs/AMAT3_SPFI_IMPLEMENTATION_DECISIONS.md,
"Riki -- SCI and SPFI model shell (issue #38)")::

    x [B,1,H,W]
      -> ShallowFeatureExtraction  (3x3 conv, 1 -> C)        # "FEN": see below
      -> shallow  ---------------------------------------+
      -> SCIStage(x=shallow, shallow)  = n_blocks x SCIBlock |
      -> body 3x3 conv                                       |
      -> + shallow  <---------------------------------------+ (global skip [ours])
      -> Reconstruction head: 3x3 conv -> PixelShuffle stages -> 3x3 conv (C -> 1)
    y [B,1,sH,sW]

Common SR interface: matches the L5 learned form -- ``forward(self, x, scale=None)``
with ``x = [B, 1, H, W]`` float, nominally in ``[0, 1]``; returns
``[B, 1, sH, sW]`` (not clamped). The sub-pixel head is scale-specific, so one
instance serves one scale; ``forward`` raises if asked for a different one.

FEN: the paper defines no block called "FEN"; its only shallow stage is "a 3x3
convolutional layer". ``ShallowFeatureExtraction`` is that conv. Its optional
``fen_depth > 0`` path is an explicit, off-by-default PLACEHOLDER for a deeper
feature-extraction network -- see the decisions doc, ambiguity (3).
"""

from __future__ import annotations

import torch
from torch import nn

from .sci import DEFAULT_MLP_RATIO, DEFAULT_NUM_HEADS, SCIStage

__all__ = ["ShallowFeatureExtraction", "ReconstructionHead", "SPFI", "build_spfi"]

# Scales the shell supports. Paper reports x2/x3/x4; x8 is [ours] because the
# L1-L5 pipeline uses it (decisions doc, ambiguity (5)).
SUPPORTED_SCALES = (2, 3, 4, 8)


class ShallowFeatureExtraction(nn.Module):
    """Paper's shallow stage: a single 3x3 conv ``in_channels -> dim``.

    ``fen_depth`` (default 0) is a PLACEHOLDER (FEN): when > 0 it appends that
    many ``Conv2d(dim, dim, 3) + GELU`` layers inside a residual, as a hook for a
    deeper feature-extraction network the team may decide SPFI needs. Off by
    default; the gate runs ``fen_depth == 0``.
    """

    def __init__(self, in_channels: int, dim: int, fen_depth: int = 0) -> None:
        super().__init__()
        if fen_depth < 0:
            raise ValueError(f"fen_depth must be >= 0, got {fen_depth}")
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.fen_depth = fen_depth
        if fen_depth > 0:
            layers: list[nn.Module] = []
            for _ in range(fen_depth):  # PLACEHOLDER (FEN)
                layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.GELU()]
            self.fen = nn.Sequential(*layers)
        else:
            self.fen = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if self.fen is not None:  # PLACEHOLDER (FEN)
            x = x + self.fen(x)
        return x


class ReconstructionHead(nn.Module):
    """3x3 conv -> sub-pixel (PixelShuffle) upsampling -> 3x3 conv to image.

    PixelShuffle arrangement (EDSR / SwinIR-light convention, [ours] -- the paper
    gives no layer counts):

    * ``scale in {2, 3}``: one ``Conv2d(dim, dim * scale**2, 3)`` + ``PixelShuffle(scale)``
    * ``scale == 4``:       two ``x2`` stages
    * ``scale == 8``:       three ``x2`` stages
    """

    def __init__(self, dim: int, scale: int, out_channels: int) -> None:
        super().__init__()
        if scale not in SUPPORTED_SCALES:
            raise ValueError(f"scale must be one of {SUPPORTED_SCALES}, got {scale}")
        self.scale = scale
        self.pre = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        ups: list[nn.Module] = []
        if scale in (2, 3):
            ups += [nn.Conv2d(dim, dim * scale * scale, kernel_size=3, padding=1),
                    nn.PixelShuffle(scale)]
        else:  # 4 or 8 -> repeated x2 stages
            n_stages = {4: 2, 8: 3}[scale]
            for _ in range(n_stages):
                ups += [nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1),
                        nn.PixelShuffle(2)]
        self.up = nn.Sequential(*ups)
        self.out = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.up(self.pre(x)))


class SPFI(nn.Module):
    """SPFI super-resolution model shell (issue #38).

    Parameters
    ----------
    scale : int
        Upscaling factor this instance is built for. One of ``SUPPORTED_SCALES``.
    in_channels : int
        Image channels. Grayscale HRTEM/SEM -> 1.
    embed_dim : int
        Feature width ``C``. Paper: 48.
    n_blocks : int
        Number of cascaded ``SCIBlock``s in the body. Not stated in the paper;
        default is [ours].
    depth : int
        SCI Transformer layers per block.
    num_heads, mlp_ratio :
        Paper: 4 and 1.
    fen_depth : int
        PLACEHOLDER (FEN) depth; 0 = paper's single 3x3 conv.

    ``forward(x, scale=None)`` -- see module docstring. ``[B, in_channels, H, W]``
    -> ``[B, in_channels, scale*H, scale*W]``.
    """

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        embed_dim: int = 48,
        n_blocks: int = 4,
        depth: int = 2,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        fen_depth: int = 0,
    ) -> None:
        super().__init__()
        if scale not in SUPPORTED_SCALES:
            raise ValueError(f"scale must be one of {SUPPORTED_SCALES}, got {scale}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.scale = scale
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.shallow = ShallowFeatureExtraction(in_channels, embed_dim, fen_depth=fen_depth)
        self.body = SCIStage(
            embed_dim,
            n_blocks=n_blocks,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
        )
        self.body_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.head = ReconstructionHead(embed_dim, scale, in_channels)

    def forward(self, x: torch.Tensor, scale: int | None = None) -> torch.Tensor:
        if scale is not None and scale != self.scale:
            raise ValueError(
                f"This SPFI instance is built for x{self.scale}; got scale={scale}. "
                "Build one SPFI per scale (see AMAT3_SPFI_IMPLEMENTATION_DECISIONS.md, "
                "ambiguity (5))."
            )
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected input [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )

        shallow = self.shallow(x)                 # [B, C, H, W]
        feat = self.body(shallow, shallow)        # [B, C, H, W]
        feat = self.body_conv(feat) + shallow     # global skip
        return self.head(feat)                    # [B, in_channels, sH, sW]


def build_spfi(scale: int = 4, **overrides) -> SPFI:
    """Factory mirroring the L5 pattern of building one model per scale.

    ``overrides`` are forwarded to :class:`SPFI` (e.g. ``embed_dim=48``,
    ``n_blocks=6``).
    """
    return SPFI(scale=scale, **overrides)
