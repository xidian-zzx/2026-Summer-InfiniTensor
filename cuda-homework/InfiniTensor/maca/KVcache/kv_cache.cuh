#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 使用 MXMACA 官方 cu-bridge 保留 CUDA 风格接口。调用方仍传设备指针，
// 编译和运行时由桥接层转到 C500 的 MACA Runtime。

// 连续式 KV Cache 的布局：
//
//   cache_k/cache_v: [batch, max_seq_len, kv_heads, head_dim]
//   new_k/new_v:     [batch, new_tokens, kv_heads, head_dim]
//   positions:       [batch, new_tokens]
//
// positions[b, t] 表示 new_k/new_v 的第 [b, t] 个 token 要写到缓存的
// 哪个序列位置。positions 必须位于 [0, max_seq_len)，同一个 batch 内本次
// 更新的位置也应互不重复。
cudaError_t launch_kv_cache_update(
    const float* new_k, const float* new_v, const int* positions,
    float* cache_k, float* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim,
    cudaStream_t stream = nullptr);

cudaError_t launch_kv_cache_update(
    const half* new_k, const half* new_v, const int* positions,
    half* cache_k, half* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim,
    cudaStream_t stream = nullptr);

// 按 positions 指定的顺序把缓存读出来：
//
//   positions: [batch, requested_tokens]
//   out_k/out_v: [batch, requested_tokens, kv_heads, head_dim]
//
// positions 可以乱序或重复，因此这个接口也能用来做简单的 token 重排。
cudaError_t launch_kv_cache_gather(
    const float* cache_k, const float* cache_v, const int* positions,
    float* out_k, float* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim,
    cudaStream_t stream = nullptr);

cudaError_t launch_kv_cache_gather(
    const half* cache_k, const half* cache_v, const int* positions,
    half* out_k, half* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim,
    cudaStream_t stream = nullptr);
