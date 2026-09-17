"""Cross-token Integration (CTI) block.

Reference
---------
Zheng, X., Chen, Z. & Huang, D. "Lightweight super-resolution method based on
scattering processing and feature interaction." Scientific Reports 16, 15018
(2026). https://doi.org/10.1038/s41598-026-44351-5

Section: "Cross-token integration block" (Fig. 4). Equations reproduced from
the paper text (own notation kept close to the source):

    K_i, V_i = DSConv_i(X),  i = 1..S                                  (1)
    K~_i     = sum_j alpha_ij * Resize(K_j)                            (2)

"Larger scale K and V are subjected to downsampling compression on one hand,
and interact across scales with smaller scale K and V via 3x3 DSConv on the
other hand. Finally, a 1x1 convolution is used to integrate the multi-scale
features and adjust the output dimension."

Role in SPFI
------------
CTI sits inside each SCI Transformer layer, after the SPM (frequency
scattering) module and before self-attention:

    X -> SPM -> CTI -> {K, V}   (this module)
    X -> [separate SCI path]    -> Q            (NOT implemented here)

    Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V

Q is produced by a separate SCI Transformer path per the task description
and is intentionally out of scope for this module. CTI's contract is:
given the (post-SPM) feature map, return finite, shape-stable, and
gradient-stable K and V tensors ready to be consumed by that attention
computation.

ASSUMPTIONS (recorded because Figure 1 and the paper text are ambiguous)
--------------------------------------------------------------------
A1. Tensor layout: the paper writes features as X in R^{H x W x C}
    (channel-last mathematical notation, consistent with how SwinIR-style
    SR transformers describe tokens). Because CTI's core operations are
    depth-wise separable convolutions, this implementation uses PyTorch's
    native channel-first layout (B, C, H, W) instead. This is a pure
    layout choice with no semantic effect; conversion at the SCI
    Transformer boundary (token sequence <-> spatial map) is the caller's
    responsibility and is out of scope for this file.

A2. Number and configuration of scales S: the paper says "different kernel
    sizes and strides" but does not give exact numbers. We default to
    S = 3 scales with (kernel=3, stride=1), (kernel=5, stride=2),
    (kernel=7, stride=4) -- monotonically larger receptive field and
    coarser resolution as the branch index increases, matching the paper's
    language of "larger scale" branches being more downsampled. Scale
    count/kernels/strides are exposed as constructor arguments so they can
    be tuned without changing this documented default.

A3. Cross-scale interaction operator (Eq. 2, alpha_ij): the paper states
    weights "are learned through a lightweight attention mechanism" but
    does not specify its exact form. We implement it as: global-average-pool
    each branch to a descriptor vector, concatenate descriptors, pass
    through a small 2-layer MLP producing S x S logits per batch element,
    and softmax over the source-branch axis j (so weights for each target
    branch i sum to 1 across sources j). This is applied identically and
    independently to the K-branches and the V-branches (two separate small
    MLPs, matching the two independent multi-scale families for K and V).

A4. "Larger scale K/V are downsampled ... AND interact via 3x3 DSConv with
    smaller scale K/V": read literally this describes two separate
    mechanisms (compression, and a specific 3x3-kernel cross-scale path)
    rather than one. We treat Eq. (2)'s Resize+weighted-sum as the
    computational realization of both: "Resize" implements the
    downsampling/upsampling compression to a common resolution, and the
    weighted combination is the interaction. We additionally pass every
    resized/fused branch through one shared 3x3 depth-wise separable
    convolution (see `_cross_scale_refine`) before the final 1x1 fusion, to
    stay faithful to the explicit "via 3x3 DSConv" wording. If the
    intended architecture in fact wants two structurally distinct paths
    (compression-only vs. interaction-only) merged additively rather than
    unified through Eq. (2), that would be a deviation from this
    implementation -- flagged here for reviewer attention.

A5. Common resolution for Resize(): Eq. (2) does not state which
    resolution K~_i lives at after resizing. Because CTI's output K, V
    must eventually be attended against Q (produced by the separate SCI
    Transformer path over the *original* H x W token grid), we resize
    every branch back to the input spatial resolution (H, W) before the
    final fusion, rather than to each branch's own native (possibly
    strided) resolution. This guarantees CTI's output token count always
    matches Q's, which is required for Attention(Q, K, V) to be well
    defined. This is treated as a hard requirement, not just a preference.

A6. CTI vs. CTT naming: the paper consistently names this block
    "Cross-token Integration (CTI)" in the abstract, contributions,
    section heading and Fig. 4 caption. We standardize on "CTI"
    throughout code, tests and docs. If any team draft/notes refer to
    "CTT" ("Cross-Token Transform" or similar), treat it as the same
    block under a different working name -- we found no basis in the
    published paper text for a distinct "CTT" entity.

A7. Output channel dimension: the paper says the final 1x1 conv is used to
    "adjust the output dimension" but does not give the target width. We
    expose `out_dim` as a constructor argument (default: same as input
    `dim`) so the SCI Transformer layer can request whatever K/V width its
    attention heads expect.

Shapes are documented at every boundary via inline comments and are also
exercised by tests/spfi/test_cti.py across multiple spatial sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ScaleConfig:
    """One multi-scale DSConv branch configuration.

    kernel_size / stride follow the paper's "different kernel sizes and
    strides" (see assumption A2). ``padding`` defaults to ``kernel_size //
    2`` (SAME-style padding) so branches with the same stride only differ
    in receptive field, not in an incidental resolution shift.
    """

    kernel_size: int
    stride: int
    padding: int | None = None

    def resolved_padding(self) -> int:
        return self.padding if self.padding is not None else self.kernel_size // 2


DEFAULT_SCALES: Tuple[ScaleConfig, ...] = (
    ScaleConfig(kernel_size=3, stride=1),
    ScaleConfig(kernel_size=5, stride=2),
    ScaleConfig(kernel_size=7, stride=4),
)


class DepthwiseSeparableConv(nn.Module):
    """Depth-wise separable convolution: depthwise spatial conv + pointwise 1x1 mix.

    Shapes
    ------
    input:  (B, C_in, H, W)
    output: (B, C_out, H_out, W_out)
        H_out = floor((H + 2*padding - kernel_size) / stride) + 1
        W_out = floor((W + 2*padding - kernel_size) / stride) + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_in, H, W)
        x = self.depthwise(x)  # (B, C_in, H_out, W_out)
        x = self.pointwise(x)  # (B, C_out, H_out, W_out)
        return x


class _MultiScaleBranch(nn.Module):
    """Produces one (K_i, V_i) pair at scale i via two independent DSConvs.

    Shapes
    ------
    input x:  (B, C, H, W)
    output K_i, V_i: (B, C, H_i, W_i), where (H_i, W_i) depends on this
    branch's stride (see DepthwiseSeparableConv docstring).
    """

    def __init__(self, dim: int, scale: ScaleConfig) -> None:
        super().__init__()
        padding = scale.resolved_padding()
        self.k_conv = DepthwiseSeparableConv(dim, dim, scale.kernel_size, scale.stride, padding)
        self.v_conv = DepthwiseSeparableConv(dim, dim, scale.kernel_size, scale.stride, padding)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.k_conv(x), self.v_conv(x)


class _CrossScaleAttentionWeights(nn.Module):
    """Lightweight attention mechanism producing alpha_ij (assumption A3).

    Shapes
    ------
    branch descriptors: S tensors of shape (B, C, H_i, W_i)
    pooled descriptors:  (B, S, C)  (global-average-pooled per branch)
    output alpha:        (B, S, S), softmax-normalized over the last axis
                          (source branch j) for each target branch i.
    """

    def __init__(self, dim: int, num_scales: int, hidden_ratio: float = 0.25) -> None:
        super().__init__()
        self.num_scales = num_scales
        hidden = max(1, int(dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim * num_scales, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_scales * num_scales),
        )

    def forward(self, branches: Sequence[Tensor]) -> Tensor:
        # Each branch: (B, C, H_i, W_i) -> pooled: (B, C)
        pooled = [F.adaptive_avg_pool2d(b, 1).flatten(1) for b in branches]  # list of (B, C)
        descriptor = torch.cat(pooled, dim=1)  # (B, C * S)
        logits = self.mlp(descriptor)  # (B, S * S)
        batch = descriptor.shape[0]
        logits = logits.view(batch, self.num_scales, self.num_scales)  # (B, S, S) [i, j]
        alpha = torch.softmax(logits, dim=-1)  # normalize over source branch j
        return alpha  # (B, S, S)


class CrossTokenIntegration(nn.Module):
    """Cross-token Integration (CTI) block.

    Produces the K/V path consumed by SCI's attention. Q is intentionally
    NOT produced here -- it comes from a separate SCI Transformer path.

    Parameters
    ----------
    dim:
        Input channel dimension C (post-SPM feature width).
    out_dim:
        Output K/V channel dimension. Defaults to ``dim`` (see A7).
    scales:
        Sequence of ScaleConfig describing each multi-scale DSConv branch.
        Defaults to DEFAULT_SCALES (see A2).

    Forward
    -------
    input x: (B, C, H, W)          -- post-SPM feature map
    output (K, V): each (B, out_dim, H, W)
        K and V always match the *input* spatial resolution (H, W), per
        assumption A5, so they can be attended against Q's H*W tokens.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        scales: Sequence[ScaleConfig] = DEFAULT_SCALES,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if len(scales) < 1:
            raise ValueError("CTI requires at least one scale")

        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.scales: Tuple[ScaleConfig, ...] = tuple(scales)
        num_scales = len(self.scales)

        self.branches = nn.ModuleList(_MultiScaleBranch(dim, s) for s in self.scales)

        # Eq. (2): alpha_ij learned via a lightweight attention mechanism.
        # Separate weight predictors for the K family and the V family
        # (assumption A3), since K and V are semantically distinct even
        # though they share branch resolutions.
        self.k_weights = _CrossScaleAttentionWeights(dim, num_scales)
        self.v_weights = _CrossScaleAttentionWeights(dim, num_scales)

        # Shared 3x3 DSConv cross-scale refinement applied after resizing
        # to the common (input) resolution, per assumption A4.
        self.k_refine = DepthwiseSeparableConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.v_refine = DepthwiseSeparableConv(dim, dim, kernel_size=3, stride=1, padding=1)

        # Final 1x1 fusion + dimensional adjustment (paper: "a 1x1
        # convolution is used to integrate the multi-scale features and
        # adjust the output dimension").
        self.k_fuse = nn.Conv2d(dim * num_scales, self.out_dim, kernel_size=1, bias=True)
        self.v_fuse = nn.Conv2d(dim * num_scales, self.out_dim, kernel_size=1, bias=True)

    def _resize_to(self, feat: Tensor, size: Tuple[int, int]) -> Tensor:
        # feat: (B, C, H_i, W_i) -> (B, C, H, W)
        if feat.shape[-2:] == size:
            return feat
        return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)

    def _cross_scale_fuse(
        self,
        branch_feats: List[Tensor],
        alpha: Tensor,
        target_size: Tuple[int, int],
        refine: nn.Module,
        fuse: nn.Module,
    ) -> Tensor:
        """Implements Eq. (2) + A4 refinement + final 1x1 fusion for one of K/V.

        Shapes
        ------
        branch_feats: S tensors, each (B, C, H_i, W_i)
        alpha:        (B, S, S)
        target_size:  (H, W) -- the input resolution (A5)
        returns:      (B, out_dim, H, W)
        """
        num_scales = len(branch_feats)
        # Resize every branch to the common target resolution once, reused
        # for all i (Resize(K_j) in Eq. 2 does not depend on i).
        resized = [self._resize_to(f, target_size) for f in branch_feats]  # list of (B, C, H, W)
        resized_stack = torch.stack(resized, dim=1)  # (B, S, C, H, W)

        fused_per_target: List[Tensor] = []
        for i in range(num_scales):
            # alpha[:, i, j] weights source branch j's contribution to target i
            weights = alpha[:, i, :].view(-1, num_scales, 1, 1, 1)  # (B, S, 1, 1, 1)
            weighted_sum = (resized_stack * weights).sum(dim=1)  # (B, C, H, W)
            weighted_sum = refine(weighted_sum)  # 3x3 DSConv cross-scale refinement (A4)
            fused_per_target.append(weighted_sum)  # (B, C, H, W)

        concatenated = torch.cat(fused_per_target, dim=1)  # (B, C * S, H, W)
        out = fuse(concatenated)  # (B, out_dim, H, W)
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the CTI K/V path.

        Parameters
        ----------
        x: (B, C, H, W) with C == self.dim.

        Returns
        -------
        K, V: each (B, self.out_dim, H, W), finite, matching input (H, W).
        """
        if x.dim() != 4:
            raise ValueError(f"CTI expects a 4D (B, C, H, W) tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.dim:
            raise ValueError(f"CTI configured for dim={self.dim}, got input channel dim {x.shape[1]}")

        batch, _, height, width = x.shape  # (B, C, H, W)
        target_size = (height, width)

        # Step 1 (Eq. 1): multi-scale K_i, V_i generation.
        # Each K_i, V_i: (B, C, H_i, W_i); H_i/W_i shrink with branch stride.
        k_branches: List[Tensor] = []
        v_branches: List[Tensor] = []
        for branch in self.branches:
            k_i, v_i = branch(x)
            k_branches.append(k_i)
            v_branches.append(v_i)

        # Step 2: lightweight attention weights alpha_ij, separately for K and V.
        # alpha_k, alpha_v: (B, S, S)
        alpha_k = self.k_weights(k_branches)
        alpha_v = self.v_weights(v_branches)

        # Step 3 (Eq. 2 + A4 + final 1x1 fusion): cross-scale interaction,
        # resize-to-common-resolution, 3x3 DSConv refinement, and
        # dimensional adjustment.
        # K, V: (B, out_dim, H, W)
        K = self._cross_scale_fuse(k_branches, alpha_k, target_size, self.k_refine, self.k_fuse)
        V = self._cross_scale_fuse(v_branches, alpha_v, target_size, self.v_refine, self.v_fuse)

        assert K.shape == (batch, self.out_dim, height, width)
        assert V.shape == (batch, self.out_dim, height, width)
        return K, V


# Backward/forward-compatible alias. See assumption A6 on CTI vs. CTT naming:
# this codebase standardizes on "CTI" per the published paper text.
CTI = CrossTokenIntegration


# --- Layer 6 integration -------------------------------------------------
# ``sci.py`` resolves the K/V path via ``from .cti import CTI`` and constructs
# it as ``CTI(dim)``. Tehila's class is named ``CrossTokenIntegration``; this
# alias is the integration seam, not a second implementation.
CTI = CrossTokenIntegration

__all__ = ["CrossTokenIntegration", "CTI", "DepthwiseSeparableConv", "ScaleConfig"]
