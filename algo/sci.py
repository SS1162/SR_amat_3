"""SCI -- Scatter Cross Integration -- block for the SPFI model shell.

Owner branch: amat-3/riki/spfi-sci-model (issue #38)

Paper: "Lightweight super-resolution method based on scattering processing and
feature interaction" (SPFI), Scientific Reports s41598-026-44351-5.
Relevant equation: Eq. (1), standard attention  softmax(Q Kᵀ / √d) V.

What this module owns (per docs/AMAT3_SPFI_IMPLEMENTATION_DECISIONS.md):

* the SCI Transformer layer -- a pre-norm Transformer layer whose attention
  consumes a **separate Q path** (computed here) and a **K/V path produced by
  CTI** (issue #37);
* SCI block stacking (``SCIBlock`` -> ``SCIStage``);
* the shallow-feature residual + convolution the paper places around the
  attention output.

Naming: the article body uses "SCI" (Scatter Cross Integration) consistently; no
"SCL" spelling was found. See the decisions doc, ambiguity (2).

Deviations from the paper, all recorded in the decisions doc:
  * global attention instead of the paper's 16x16 Window Self-Attention
    (TODO(final-integration): swap ``SCIAttention`` for a windowed variant);
  * pre-norm + FFN made explicit (prose omits them, the config table implies the
    FFN via "MLP expansion ratio = 1");
  * per-block ``+ shallow`` then 3x3 conv, wrapped in an input residual for
    cascade stability.

``SPM`` (#36) and ``CTI`` (#37) are now real modules and are imported directly;
there is no identity-stub fallback (see the import block below for why).
"""

from __future__ import annotations

import torch
from torch import nn

# Layer 6 integration is complete: SPM (#36, Shoshana) and CTI (#37, Tehila) are
# real modules now, so these are hard imports. The previous try/except fallback to
# ``.stubs`` was removed deliberately -- with it in place, an ImportError anywhere
# in the SPM or CTI path would silently downgrade the model to identity modules
# and training would still "work", producing results labelled SPFI that were not
# SPFI. A missing dependency must fail loudly instead.
from .cti import CTI
from .spm import SPM

__all__ = [
    "LayerNorm2d",
    "MLP",
    "SCIAttention",
    "SCITransformerLayer",
    "SCIBlock",
    "SCIStage",
]

# Paper hyper-parameters (methods section, hyper-parameter table).
DEFAULT_NUM_HEADS = 4
DEFAULT_MLP_RATIO = 1


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an ``[B, C, H, W]`` tensor.

    Same computation as ``nn.LayerNorm`` applied to the channel axis (the
    ConvNeXt "channels_first" form), so we do not have to permute to
    ``[B, H, W, C]`` and back inside every block.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class MLP(nn.Module):
    """Position-wise feed-forward network (FFN), implemented with 1x1 convs.

    ``expansion`` is the paper's "MLP expansion ratio" (default 1).
    """

    def __init__(self, dim: int, expansion: float = DEFAULT_MLP_RATIO, drop: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(round(dim * expansion)))
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SCIAttention(nn.Module):
    """Multi-head scaled dot-product attention -- Eq. (1).

    Separate Q path and external K/V path:

    * ``q_src``  -- feature map from the SCI Transformer layer itself
      (``[B, C, Hq, Wq]``); projected here into the query space.
    * ``k``, ``v`` -- feature maps produced by CTI (``[B, C, Hk, Wk]``); may be a
      different spatial size than ``q_src`` (CTI is multi-scale), but must match
      each other. The output keeps ``q_src``'s spatial size.

    Global attention over all tokens. The paper uses 16x16 Window Self-Attention;
    swapping to windows is a self-contained change here and is tracked as
    ``TODO(final-integration)`` in the module docstring.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = DEFAULT_NUM_HEADS,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, num_heads, N, head_dim]   with N = H * W
        b, _, h, w = x.shape
        x = x.reshape(b, self.num_heads, self.head_dim, h * w)
        return x.permute(0, 1, 3, 2)

    def forward(self, q_src: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if k.shape[-2:] != v.shape[-2:]:
            raise ValueError(
                f"K and V must share spatial size, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        b, c, hq, wq = q_src.shape

        q = self._to_tokens(self.to_q(q_src))  # [B, heads, Nq, head_dim]
        k = self._to_tokens(self.to_k(k))      # [B, heads, Nkv, head_dim]
        v = self._to_tokens(self.to_v(v))      # [B, heads, Nkv, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, Nq, Nkv]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                  # [B, heads, Nq, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(b, c, hq, wq)
        out = self.proj(out)
        return self.proj_drop(out)


class SCITransformerLayer(nn.Module):
    """One SCI Transformer layer.

    Order (decisions doc, "SCI Transformer layer -- reading of an underspecified
    block"): the paper's stated  SPM -> CTI -> attention -> (+shallow) -> conv
    wrapped in a standard pre-norm Transformer layer::

        xs   = SPM(LayerNorm(x))          # frequency-domain scattering pre-process
        k, v = CTI(xs)                    # K/V path (multi-scale, from CTI)
        x    = x + Attention(q_src=xs, k, v)
        x    = x + MLP(LayerNorm(x))      # FFN, expansion ratio 1

    ``spm`` / ``cti`` can be injected (tests, final integration wiring); by
    default they are constructed from the resolved ``SPM`` / ``CTI`` symbols,
    which are identity stubs until #36 / #37 merge.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        spm: nn.Module | None = None,
        cti: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.spm = spm if spm is not None else SPM(dim)
        self.cti = cti if cti is not None else CTI(dim)
        self.attn = SCIAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, expansion=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.spm(self.norm1(x))
        k, v = self.cti(xs)
        x = x + self.attn(xs, k, v)
        x = x + self.mlp(self.norm2(x))
        return x


class SCIBlock(nn.Module):
    """A stack of ``depth`` SCI Transformer layers + the shallow-feature residual.

    ``forward(x, shallow)`` -> ``[B, C, H, W]``::

        h = layers(x)                    # depth x SCITransformerLayer
        h = conv3x3(h + shallow)         # paper: combine w/ shallow, then a conv
        return x + h                     # [ours] input residual, cascade stability
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.layers = nn.ModuleList(
            SCITransformerLayer(dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        h = self.conv(h + shallow)
        return x + h


class SCIStage(nn.Module):
    """The deep feature extraction body: ``n_blocks`` cascaded ``SCIBlock``s.

    Every block sees the same original ``shallow`` features (paper: "the original
    shallow features"). ``forward(x, shallow)`` -> ``[B, C, H, W]``.
    """

    def __init__(
        self,
        dim: int,
        n_blocks: int = 4,
        depth: int = 2,
        num_heads: int = DEFAULT_NUM_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.blocks = nn.ModuleList(
            SCIBlock(dim, depth, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(n_blocks)
        )

    def forward(self, x: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, shallow)
        return x
