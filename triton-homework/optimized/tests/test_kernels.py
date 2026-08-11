import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from triton_kernels import (  # noqa: E402
    add_rms_norm,
    apply_rope,
    causal_gqa_attention,
    rms_norm,
    swiglu,
)


def reference_attention(query, key, value):
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    seq_len = query.shape[2]
    mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)
    )
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    scores = torch.where(mask, scores, float("-inf"))
    return torch.softmax(scores, dim=-1) @ value


def test_cpu_fallbacks_match_pytorch():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 64)
    update = torch.randn_like(x)
    weight = torch.randn(64)
    eps = 1e-5

    expected_norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
    torch.testing.assert_close(rms_norm(x, weight, eps), expected_norm)

    summed, normalized = add_rms_norm(x, update, weight, eps)
    expected_sum = x + update
    expected_add_norm = (
        expected_sum
        * torch.rsqrt(expected_sum.square().mean(-1, keepdim=True) + eps)
        * weight
    )
    torch.testing.assert_close(summed, expected_sum)
    torch.testing.assert_close(normalized, expected_add_norm)
    torch.testing.assert_close(swiglu(x, update), torch.nn.functional.silu(x) * update)

    rope_input = torch.randn(2, 5, 4, 16)
    sin = torch.randn(5, 8)
    cos = torch.randn(5, 8)
    half = rope_input.shape[-1] // 2
    expected_rope = torch.cat(
        (
            rope_input[..., :half] * cos[None, :, None, :]
            - rope_input[..., half:] * sin[None, :, None, :],
            rope_input[..., :half] * sin[None, :, None, :]
            + rope_input[..., half:] * cos[None, :, None, :],
        ),
        dim=-1,
    )
    torch.testing.assert_close(apply_rope(rope_input, sin, cos), expected_rope)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("seq_len", [1, 17, 63, 129])
def test_cuda_kernels_match_pytorch(seq_len):
    torch.manual_seed(11)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn(2, seq_len, 128, device=device, dtype=dtype)
    update = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=dtype)
    eps = 1e-5

    expected_norm = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    ) * weight.float()
    torch.testing.assert_close(
        rms_norm(x, weight, eps).float(), expected_norm, atol=2e-2, rtol=2e-2
    )

    summed, normalized = add_rms_norm(x, update, weight, eps)
    expected_sum = x + update
    expected_add_norm = expected_sum.float() * torch.rsqrt(
        expected_sum.float().square().mean(-1, keepdim=True) + eps
    ) * weight.float()
    torch.testing.assert_close(summed, expected_sum)
    torch.testing.assert_close(
        normalized.float(), expected_add_norm, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        swiglu(x, update).float(),
        (torch.nn.functional.silu(x) * update).float(),
        atol=2e-2,
        rtol=2e-2,
    )

    query = torch.randn(2, 8, seq_len, 64, device=device, dtype=dtype)
    key = torch.randn(2, 2, seq_len, 64, device=device, dtype=dtype)
    value = torch.randn_like(key)
    expected_attention = reference_attention(query, key, value)
    actual_attention = causal_gqa_attention(query, key, value)
    torch.testing.assert_close(
        actual_attention.float(),
        expected_attention.float(),
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_rope_matches_pytorch():
    torch.manual_seed(13)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x = torch.randn(2, 31, 8, 64, device="cuda", dtype=dtype)
    sin = torch.randn(31, 32, device="cuda", dtype=dtype)
    cos = torch.randn(31, 32, device="cuda", dtype=dtype)
    half = x.shape[-1] // 2
    expected = torch.cat(
        (
            x[..., :half] * cos[None, :, None, :]
            - x[..., half:] * sin[None, :, None, :],
            x[..., :half] * sin[None, :, None, :]
            + x[..., half:] * cos[None, :, None, :],
        ),
        dim=-1,
    )
    # Triton 会把乘加留在寄存器里，和 PyTorch 分步写回 BF16 相差一格很常见。
    torch.testing.assert_close(
        apply_rope(x, sin, cos), expected, atol=2e-2, rtol=2e-2
    )
