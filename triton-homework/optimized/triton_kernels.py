"""Llama 推理用到的 Triton 融合算子。

CUDA/Triton 不可用时会走 PyTorch 参考实现，方便在 CPU 上检查模型结构；
真正计分时应当使用 CUDA 路径。
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols

        # 这里先转成 fp32 累加，半精度直接求平方和容易有点飘。
        x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x_fp32 * x_fp32, axis=0) / n_cols + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        tl.store(output_ptr + row * row_stride + cols, x_fp32 * inv_rms * weight, mask=mask)


    @triton.jit
    def _add_rms_norm_kernel(
        residual_ptr,
        update_ptr,
        weight_ptr,
        sum_ptr,
        norm_ptr,
        row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols
        offsets = row * row_stride + cols

        # 残差相加和下一次归一化放一趟里做，少来回读一次显存。
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        update = tl.load(update_ptr + offsets, mask=mask, other=0.0)
        summed = residual + update
        summed_fp32 = summed.to(tl.float32)
        inv_rms = tl.rsqrt(
            tl.sum(summed_fp32 * summed_fp32, axis=0) / n_cols + eps
        )
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        tl.store(sum_ptr + offsets, summed, mask=mask)
        tl.store(norm_ptr + offsets, summed_fp32 * inv_rms * weight, mask=mask)


    @triton.jit
    def _swiglu_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        n_elements,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        # SiLU 和逐元素乘法顺手融合掉，两个小 kernel 就合成一个啦。
        activated = gate * tl.sigmoid(gate)
        tl.store(output_ptr + offsets, activated * up, mask=mask)


    @triton.jit
    def _rope_kernel(
        input_ptr,
        sin_ptr,
        cos_ptr,
        output_ptr,
        input_stride_b,
        input_stride_s,
        input_stride_h,
        input_stride_d,
        table_stride_s,
        table_stride_d,
        output_stride_b,
        output_stride_s,
        output_stride_h,
        output_stride_d,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        half_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row % num_heads
        token = (row // num_heads) % seq_len
        batch = row // (num_heads * seq_len)
        dims = tl.arange(0, block_size)
        mask = dims < half_dim

        input_base = (
            batch * input_stride_b + token * input_stride_s + head * input_stride_h
        )
        output_base = (
            batch * output_stride_b
            + token * output_stride_s
            + head * output_stride_h
        )
        table_offsets = token * table_stride_s + dims * table_stride_d
        x0 = tl.load(
            input_ptr + input_base + dims * input_stride_d, mask=mask, other=0.0
        )
        x1 = tl.load(
            input_ptr + input_base + (dims + half_dim) * input_stride_d,
            mask=mask,
            other=0.0,
        )
        sin = tl.load(sin_ptr + table_offsets, mask=mask, other=0.0)
        cos = tl.load(cos_ptr + table_offsets, mask=mask, other=0.0)

        tl.store(
            output_ptr + output_base + dims * output_stride_d,
            x0 * cos - x1 * sin,
            mask=mask,
        )
        tl.store(
            output_ptr + output_base + (dims + half_dim) * output_stride_d,
            x0 * sin + x1 * cos,
            mask=mask,
        )


    @triton.jit
    def _causal_gqa_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        query_stride_b,
        query_stride_h,
        query_stride_s,
        query_stride_d,
        key_stride_b,
        key_stride_h,
        key_stride_s,
        key_stride_d,
        value_stride_b,
        value_stride_h,
        value_stride_s,
        value_stride_d,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
        seq_len,
        softmax_scale: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_d: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_query_heads
        query_head = batch_head % num_query_heads
        kv_head = query_head // (num_query_heads // num_kv_heads)

        offs_m = query_block * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        query_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        query_offsets = (
            batch * query_stride_b
            + query_head * query_stride_h
            + offs_m[:, None] * query_stride_s
            + offs_d[None, :] * query_stride_d
        )
        query = tl.load(query_ptr + query_offsets, mask=query_mask, other=0.0)

        max_score = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, block_d), tl.float32)
        # exp2 比 exp 快些，把缩放系数提前换到底数 2 就行。
        qk_scale = softmax_scale * 1.4426950408889634
        causal_end = tl.minimum((query_block + 1) * block_m, seq_len)

        for start_n in tl.range(0, causal_end, block_n):
            current_n = start_n + offs_n
            key_offsets = (
                batch * key_stride_b
                + kv_head * key_stride_h
                + offs_d[:, None] * key_stride_d
                + current_n[None, :] * key_stride_s
            )
            key_mask = (offs_d[:, None] < head_dim) & (
                current_n[None, :] < seq_len
            )
            key = tl.load(key_ptr + key_offsets, mask=key_mask, other=0.0)

            scores = tl.dot(query, key) * qk_scale
            causal_mask = (offs_m[:, None] >= current_n[None, :]) & (
                current_n[None, :] < seq_len
            )
            scores = tl.where(causal_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_score, block_max)
            probabilities = tl.exp2(scores - new_max[:, None])
            correction = tl.exp2(max_score - new_max)
            new_denominator = denominator * correction + tl.sum(
                probabilities, axis=1
            )

            value_offsets = (
                batch * value_stride_b
                + kv_head * value_stride_h
                + current_n[:, None] * value_stride_s
                + offs_d[None, :] * value_stride_d
            )
            value_mask = (current_n[:, None] < seq_len) & (
                offs_d[None, :] < head_dim
            )
            value = tl.load(value_ptr + value_offsets, mask=value_mask, other=0.0)
            accumulator = accumulator * correction[:, None] + tl.dot(
                probabilities.to(query.dtype), value
            )
            max_score = new_max
            denominator = new_denominator

        output = accumulator / denominator[:, None]
        output_offsets = (
            batch * output_stride_b
            + query_head * output_stride_h
            + offs_m[:, None] * output_stride_s
            + offs_d[None, :] * output_stride_d
        )
        output_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        tl.store(output_ptr + output_offsets, output, mask=output_mask)
def _use_triton(*tensors):
    return TRITON_AVAILABLE and all(tensor.is_cuda for tensor in tensors)


def rms_norm(input, weight, eps):
    if not _use_triton(input, weight):
        return input * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + eps) * weight

    input = input.contiguous()
    n_cols = input.shape[-1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"RMSNorm hidden size is too large for this kernel: {n_cols}")
    output = torch.empty_like(input)
    rows = input.numel() // n_cols
    _rms_norm_kernel[(rows,)](
        input,
        weight,
        output,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        block_size=block_size,
        num_warps=8 if block_size >= 4096 else 4,
    )
    return output
def add_rms_norm(residual, update, weight, eps):
    """返回残差和，以及同一份残差经过 RMSNorm 后的结果。"""
    if not _use_triton(residual, update, weight):
        summed = residual + update
        normalized = (
            summed
            * torch.rsqrt(summed.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight
        )
        return summed, normalized

    residual = residual.contiguous()
    update = update.contiguous()
    n_cols = residual.shape[-1]
    block_size = triton.next_power_of_2(n_cols)
    summed = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    rows = residual.numel() // n_cols
    _add_rms_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        summed,
        normalized,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        block_size=block_size,
        num_warps=8 if block_size >= 4096 else 4,
    )
    return summed, normalized


def swiglu(gate, up):
    if not _use_triton(gate, up):
        return torch.nn.functional.silu(gate) * up

    gate = gate.contiguous()
    up = up.contiguous()
    output = torch.empty_like(gate)
    # 小序列用 512 保住并行度，张量大起来后 1024 能少发不少 program。
    block_size = 1024 if gate.numel() >= 1_500_000 else 512
    grid = (triton.cdiv(gate.numel(), block_size),)
    _swiglu_kernel[grid](gate, up, output, gate.numel(), block_size=block_size)
    return output


def apply_rope(input, sin_table, cos_table):
    if not _use_triton(input, sin_table, cos_table):
        sin = sin_table[None, :, None, :]
        cos = cos_table[None, :, None, :]
        half = input.shape[-1] // 2
        input_0, input_1 = input[..., :half], input[..., half:]
        return torch.cat(
            (input_0 * cos - input_1 * sin, input_0 * sin + input_1 * cos),
            dim=-1,
        )

    if input.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    output = torch.empty_like(input)
    batch, seq_len, num_heads, head_dim = input.shape
    half_dim = head_dim // 2
    block_size = triton.next_power_of_2(half_dim)
    _rope_kernel[(batch * seq_len * num_heads,)](
        input,
        sin_table,
        cos_table,
        output,
        *input.stride(),
        *sin_table.stride(),
        *output.stride(),
        seq_len=seq_len,
        num_heads=num_heads,
        half_dim=half_dim,
        block_size=block_size,
    )
    return output


def causal_gqa_attention(query, key, value):
    if not _use_triton(query, key, value):
        num_query_heads = query.shape[1]
        key = key.repeat_interleave(num_query_heads // key.shape[1], dim=1)
        value = value.repeat_interleave(num_query_heads // value.shape[1], dim=1)
        scale = 1 / math.sqrt(query.shape[-1])
        seq_len = query.shape[2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device))
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        scores = torch.where(mask, scores, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), value)

    batch, num_query_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if key.shape != value.shape or key.shape[2:] != (seq_len, head_dim):
        raise ValueError("Query, key and value shapes do not describe self-attention")

    output = torch.empty_like(query)
    block_m = 16
    # 4090D 上 64 列 tile 在常见 64~128 token 区间更省事，短序列也没吃亏。
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(seq_len, block_m), batch * num_query_heads)
    _causal_gqa_attention_kernel[grid](
        query,
        key,
        value,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        seq_len,
        softmax_scale=1 / math.sqrt(head_dim),
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_d=block_d,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output
