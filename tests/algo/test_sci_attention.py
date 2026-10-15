"""SCIAttention uses PyTorch's fused attention; it must equal the explicit formula
softmax(Q Kᵀ / √d) V, forward and backward, for every head size."""

from __future__ import annotations

import pytest
import torch

from algo.sci import SCIAttention


def explicit_attention(attn: SCIAttention, q_src, k, v):
    b, c, hq, wq = q_src.shape
    q = attn._to_tokens(attn.to_q(q_src))
    k = attn._to_tokens(attn.to_k(k))
    v = attn._to_tokens(attn.to_v(v))
    weights = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(dim=-1)
    out = (weights @ v).permute(0, 1, 3, 2).reshape(b, c, hq, wq)
    return attn.proj(out)


@pytest.mark.parametrize("dim", [8, 16, 48, 64])   # head_dim 2, 4, 12, 16
def test_matches_explicit_attention(dim: int) -> None:
    torch.manual_seed(0)
    attn = SCIAttention(dim, num_heads=4).double()
    q_src = torch.randn(2, dim, 6, 10, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(2, dim, 3, 5, dtype=torch.float64, requires_grad=True)   # K/V may be smaller

    fused = attn(q_src, kv, kv)
    reference = explicit_attention(attn, q_src, kv, kv)
    assert fused.shape == q_src.shape
    torch.testing.assert_close(fused, reference)

    inputs = [q_src, kv, *attn.parameters()]
    g_fused = torch.autograd.grad(fused.square().sum(), inputs)
    g_ref = torch.autograd.grad(reference.square().sum(), inputs)
    for a, b in zip(g_fused, g_ref):
        torch.testing.assert_close(a, b)
