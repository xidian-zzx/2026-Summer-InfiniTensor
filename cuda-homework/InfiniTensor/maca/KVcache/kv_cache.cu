#include "kv_cache.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

#ifndef KV_DECODE_WARPS
#define KV_DECODE_WARPS 8
#endif

#ifndef KV_BULK_THREADS
#define KV_BULK_THREADS 512
#endif

// C500 的物理 wavefront 是 64 线程，cu-bridge 对 CUDA 源码提供 32 线程
// 逻辑 warp。decode 按逻辑 warp 分工，shuffle 和 lane 编号都由桥接层处理。
constexpr int kWarpSize = 32;
constexpr int kDecodeWarps = KV_DECODE_WARPS;
constexpr int kBulkThreads = KV_BULK_THREADS;
constexpr int kMaxBlocks = 4096;
static_assert(kDecodeWarps > 0 && kDecodeWarps <= 32,
              "KV_DECODE_WARPS must be in [1, 32]");
static_assert(kBulkThreads > 0 && kBulkThreads <= 1024
                  && kBulkThreads % kWarpSize == 0,
              "KV_BULK_THREADS must be a warp-aligned block size");

// KV Cache 的 update 和 gather 本质上都是“搬数据”，没有复杂计算。
// 优化重点是让每个线程一次搬一小包连续元素，并让相邻线程访问相邻地址。
template<typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T elem[N];
};

// 一个 warp 专门负责一个 [batch, token, kv_head]。这样做特别适合 LLM
// decode：每步通常只有一个新 token，但每个 token 仍有多个 KV head，
// 可以把这些 head 同时铺到 GPU 上，而不会只启动寥寥几个 block。
//
// K 和 V 在同一个 kernel 中搬运。它们的位置映射完全相同，因此 position
// 每个 warp 只读一次，地址除法也只做一次，不再让每个 Pack 重复计算。
template<typename T, int PackSize>
__global__ void kv_cache_update_kernel(
    const T* __restrict__ new_k, const T* __restrict__ new_v,
    const int* __restrict__ positions, T* __restrict__ cache_k,
    T* __restrict__ cache_v, int max_seq_len, int kv_heads, int head_dim,
    int packs_per_head) {
  using PackT = Pack<T, PackSize>;
  const PackT* packed_new_k = reinterpret_cast<const PackT*>(new_k);
  const PackT* packed_new_v = reinterpret_cast<const PackT*>(new_v);
  PackT* packed_cache_k = reinterpret_cast<PackT*>(cache_k);
  PackT* packed_cache_v = reinterpret_cast<PackT*>(cache_v);

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;

  // grid.x 直接就是 batch，grid.y 表示这一批 head 的第几组。这样 kernel
  // 里不用再拿线性编号反复做除法和取模，decode 的小工作量会更划算。
  const int batch = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y) * warps_per_block + warp_in_block;
  __shared__ int position;
  if (threadIdx.x == 0) { position = positions[batch]; }
  __syncthreads();
  if (head >= kv_heads
      || static_cast<unsigned int>(position)
             >= static_cast<unsigned int>(max_seq_len)) {
    return;
  }

  // 同一个 block 的几个 warp 属于同一个 batch，所以 position 也只读一次。
  const int64_t source_element =
      (static_cast<int64_t>(batch) * kv_heads + head) * head_dim;
  const int64_t destination_element =
      ((static_cast<int64_t>(batch) * max_seq_len + position) * kv_heads + head)
      * head_dim;
  const int64_t source_pack = source_element / PackSize;
  const int64_t destination_pack = destination_element / PackSize;
  for (int pack = lane; pack < packs_per_head; pack += kWarpSize) {
    packed_cache_k[destination_pack + pack] = packed_new_k[source_pack + pack];
    packed_cache_v[destination_pack + pack] = packed_new_v[source_pack + pack];
  }
}

// gather 与 update 的地址映射方向相反：从大缓存的指定 position 读取，
// 再连续写入紧凑的 [B, Trequested, H, D] 输出。
template<typename T, int PackSize>
__global__ void kv_cache_gather_kernel(
    const T* __restrict__ cache_k, const T* __restrict__ cache_v,
    const int* __restrict__ positions, T* __restrict__ out_k,
    T* __restrict__ out_v, int max_seq_len, int kv_heads, int head_dim,
    int packs_per_head) {
  using PackT = Pack<T, PackSize>;
  const PackT* packed_cache_k = reinterpret_cast<const PackT*>(cache_k);
  const PackT* packed_cache_v = reinterpret_cast<const PackT*>(cache_v);
  PackT* packed_out_k = reinterpret_cast<PackT*>(out_k);
  PackT* packed_out_v = reinterpret_cast<PackT*>(out_v);

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int batch = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y) * warps_per_block + warp_in_block;
  __shared__ int position;
  if (threadIdx.x == 0) { position = positions[batch]; }
  __syncthreads();
  if (head >= kv_heads
      || static_cast<unsigned int>(position)
             >= static_cast<unsigned int>(max_seq_len)) {
    return;
  }

  const int64_t source_element =
      ((static_cast<int64_t>(batch) * max_seq_len + position) * kv_heads + head)
      * head_dim;
  const int64_t destination_element =
      (static_cast<int64_t>(batch) * kv_heads + head) * head_dim;
  const int64_t source_pack = source_element / PackSize;
  const int64_t destination_pack = destination_element / PackSize;
  for (int pack = lane; pack < packs_per_head; pack += kWarpSize) {
    packed_out_k[destination_pack + pack] = packed_cache_k[source_pack + pack];
    packed_out_v[destination_pack + pack] = packed_cache_v[source_pack + pack];
  }
}

// 多 token 的 prefill 场景已经有足够大的数据量，直接把所有 Pack 平铺给
// 256 个线程更容易跑满带宽。它保留为 bulk 路径，与上面的 decode 路径互补。
template<typename T, int PackSize>
__global__ void kv_cache_update_bulk_kernel(
    const T* __restrict__ new_k, const T* __restrict__ new_v,
    const int* __restrict__ positions, T* __restrict__ cache_k,
    T* __restrict__ cache_v, int new_tokens, int max_seq_len,
    int elements_per_token, int packs_per_token, int64_t total_packs) {
  using PackT = Pack<T, PackSize>;
  const PackT* packed_new_k = reinterpret_cast<const PackT*>(new_k);
  const PackT* packed_new_v = reinterpret_cast<const PackT*>(new_v);
  PackT* packed_cache_k = reinterpret_cast<PackT*>(cache_k);
  PackT* packed_cache_v = reinterpret_cast<PackT*>(cache_v);

  for (int64_t pack_id = static_cast<int64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       pack_id < total_packs;
       pack_id += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t linear_token = pack_id / packs_per_token;
    const int pack_in_token = static_cast<int>(pack_id % packs_per_token);
    const int batch = static_cast<int>(linear_token / new_tokens);
    const int token = static_cast<int>(linear_token % new_tokens);
    const int position = positions[batch * new_tokens + token];
    if (static_cast<unsigned int>(position)
        >= static_cast<unsigned int>(max_seq_len)) {
      continue;
    }
    const int64_t destination_element =
        (static_cast<int64_t>(batch) * max_seq_len + position)
        * elements_per_token;
    const int64_t destination_pack =
        destination_element / PackSize + pack_in_token;
    packed_cache_k[destination_pack] = packed_new_k[pack_id];
    packed_cache_v[destination_pack] = packed_new_v[pack_id];
  }
}

template<typename T, int PackSize>
__global__ void kv_cache_gather_bulk_kernel(
    const T* __restrict__ cache_k, const T* __restrict__ cache_v,
    const int* __restrict__ positions, T* __restrict__ out_k,
    T* __restrict__ out_v, int requested_tokens, int max_seq_len,
    int elements_per_token, int packs_per_token, int64_t total_packs) {
  using PackT = Pack<T, PackSize>;
  const PackT* packed_cache_k = reinterpret_cast<const PackT*>(cache_k);
  const PackT* packed_cache_v = reinterpret_cast<const PackT*>(cache_v);
  PackT* packed_out_k = reinterpret_cast<PackT*>(out_k);
  PackT* packed_out_v = reinterpret_cast<PackT*>(out_v);

  for (int64_t pack_id = static_cast<int64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       pack_id < total_packs;
       pack_id += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t linear_token = pack_id / packs_per_token;
    const int pack_in_token = static_cast<int>(pack_id % packs_per_token);
    const int batch = static_cast<int>(linear_token / requested_tokens);
    const int token = static_cast<int>(linear_token % requested_tokens);
    const int position = positions[batch * requested_tokens + token];
    if (static_cast<unsigned int>(position)
        >= static_cast<unsigned int>(max_seq_len)) {
      continue;
    }
    const int64_t source_element =
        (static_cast<int64_t>(batch) * max_seq_len + position)
        * elements_per_token;
    const int64_t source_pack = source_element / PackSize + pack_in_token;
    packed_out_k[pack_id] = packed_cache_k[source_pack];
    packed_out_v[pack_id] = packed_cache_v[source_pack];
  }
}

// C500 实测 512 线程在短 prefill 和大工作集上都更快。这个值仍然可以在
// 编译时覆盖，后面换卡重扫时，不需要为了调一个数字去改 kernel 正文。
inline int choose_bulk_threads() {
  return kBulkThreads;
}

inline int choose_bulk_blocks(int64_t total_packs, int threads) {
  const int64_t needed = (total_packs + threads - 1) / threads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, kMaxBlocks)));
}

template<typename T, int PackSize>
cudaError_t launch_update_pack(
    const T* new_k, const T* new_v, const int* positions,
    T* cache_k, T* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  if (new_tokens > 1) {
    const int elements_per_token = kv_heads * head_dim;
    const int packs_per_token = elements_per_token / PackSize;
    const int64_t total_packs =
        static_cast<int64_t>(batch_size) * new_tokens * packs_per_token;
    const int threads = choose_bulk_threads();
    kv_cache_update_bulk_kernel<T, PackSize>
        <<<choose_bulk_blocks(total_packs, threads), threads, 0, stream>>>(
            new_k, new_v, positions, cache_k, cache_v, new_tokens, max_seq_len,
            elements_per_token, packs_per_token, total_packs);
    return cudaPeekAtLastError();
  }

  const int packs_per_head = head_dim / PackSize;
  // 每个 warp 仍然独立处理一个 head，不过把 8 个 warp 塞进同一个 block。
  // 常见的 8 个 KV head 正好一次覆盖，能少付一些小 block 的调度成本。
  const int warps_per_block = kDecodeWarps;
  const int threads = warps_per_block * kWarpSize;
  const dim3 blocks(batch_size,
                    (kv_heads + warps_per_block - 1) / warps_per_block);
  kv_cache_update_kernel<T, PackSize>
      <<<blocks, threads, 0, stream>>>(new_k, new_v, positions, cache_k, cache_v,
                                      max_seq_len, kv_heads, head_dim,
                                      packs_per_head);
  return cudaPeekAtLastError();
}

template<typename T, int PackSize>
cudaError_t launch_gather_pack(
    const T* cache_k, const T* cache_v, const int* positions,
    T* out_k, T* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  if (requested_tokens > 1) {
    const int elements_per_token = kv_heads * head_dim;
    const int packs_per_token = elements_per_token / PackSize;
    const int64_t total_packs =
        static_cast<int64_t>(batch_size) * requested_tokens * packs_per_token;
    const int threads = choose_bulk_threads();
    kv_cache_gather_bulk_kernel<T, PackSize>
        <<<choose_bulk_blocks(total_packs, threads), threads, 0, stream>>>(
            cache_k, cache_v, positions, out_k, out_v, requested_tokens,
            max_seq_len, elements_per_token, packs_per_token, total_packs);
    return cudaPeekAtLastError();
  }

  const int packs_per_head = head_dim / PackSize;
  const int warps_per_block = kDecodeWarps;
  const int threads = warps_per_block * kWarpSize;
  const dim3 blocks(batch_size,
                    (kv_heads + warps_per_block - 1) / warps_per_block);
  kv_cache_gather_kernel<T, PackSize>
      <<<blocks, threads, 0, stream>>>(cache_k, cache_v, positions, out_k, out_v,
                                      max_seq_len, kv_heads, head_dim,
                                      packs_per_head);
  return cudaPeekAtLastError();
}

template<typename T>
bool invalid_common_arguments(const T* first_input, const T* second_input,
                              const int* positions, T* first_output,
                              T* second_output, int batch_size, int tokens,
                              int max_seq_len, int kv_heads, int head_dim) {
  // 这里只检查 host 端马上能看出来的问题。positions 中每个具体数值位于
  // GPU 显存，需要由 kernel 做最后一道越界保护。
  return first_input == nullptr || second_input == nullptr || positions == nullptr
         || first_output == nullptr || second_output == nullptr
         || batch_size <= 0 || tokens <= 0 || max_seq_len <= 0
         || kv_heads <= 0 || head_dim <= 0;
}

template<typename T>
cudaError_t update_impl(
    const T* new_k, const T* new_v, const int* positions,
    T* cache_k, T* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  if (invalid_common_arguments(new_k, new_v, positions, cache_k, cache_v,
                               batch_size, new_tokens, max_seq_len,
                               kv_heads, head_dim)) {
    return cudaErrorInvalidValue;
  }

  // 解码路径里，一个 warp 负责一个 head，Pack 不能跨过 head_dim 边界；
  // 批量路径按整个 token 摊平，所以只要 H * D 能整除 Pack 就行。
  const int vector_width_basis =
      new_tokens == 1 ? head_dim : kv_heads * head_dim;
  if constexpr (std::is_same<T, half>::value) {
    if (vector_width_basis % 8 == 0) {
      return launch_update_pack<T, 8>(new_k, new_v, positions, cache_k, cache_v,
                                      batch_size, new_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  } else {
    if (vector_width_basis % 4 == 0) {
      return launch_update_pack<T, 4>(new_k, new_v, positions, cache_k, cache_v,
                                      batch_size, new_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  }
  return launch_update_pack<T, 1>(new_k, new_v, positions, cache_k, cache_v,
                                  batch_size, new_tokens, max_seq_len,
                                  kv_heads, head_dim, stream);
}

template<typename T>
cudaError_t gather_impl(
    const T* cache_k, const T* cache_v, const int* positions,
    T* out_k, T* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  if (invalid_common_arguments(cache_k, cache_v, positions, out_k, out_v,
                               batch_size, requested_tokens, max_seq_len,
                               kv_heads, head_dim)) {
    return cudaErrorInvalidValue;
  }

  const int vector_width_basis =
      requested_tokens == 1 ? head_dim : kv_heads * head_dim;
  if constexpr (std::is_same<T, half>::value) {
    if (vector_width_basis % 8 == 0) {
      return launch_gather_pack<T, 8>(cache_k, cache_v, positions, out_k, out_v,
                                      batch_size, requested_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  } else {
    if (vector_width_basis % 4 == 0) {
      return launch_gather_pack<T, 4>(cache_k, cache_v, positions, out_k, out_v,
                                      batch_size, requested_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  }
  return launch_gather_pack<T, 1>(cache_k, cache_v, positions, out_k, out_v,
                                  batch_size, requested_tokens, max_seq_len,
                                  kv_heads, head_dim, stream);
}

}  // namespace

cudaError_t launch_kv_cache_update(
    const float* new_k, const float* new_v, const int* positions,
    float* cache_k, float* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  return update_impl(new_k, new_v, positions, cache_k, cache_v, batch_size,
                     new_tokens, max_seq_len, kv_heads, head_dim, stream);
}

cudaError_t launch_kv_cache_update(
    const half* new_k, const half* new_v, const int* positions,
    half* cache_k, half* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  return update_impl(new_k, new_v, positions, cache_k, cache_v, batch_size,
                     new_tokens, max_seq_len, kv_heads, head_dim, stream);
}

cudaError_t launch_kv_cache_gather(
    const float* cache_k, const float* cache_v, const int* positions,
    float* out_k, float* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  return gather_impl(cache_k, cache_v, positions, out_k, out_v, batch_size,
                     requested_tokens, max_seq_len, kv_heads, head_dim, stream);
}

cudaError_t launch_kv_cache_gather(
    const half* cache_k, const half* cache_v, const int* positions,
    half* out_k, half* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  return gather_impl(cache_k, cache_v, positions, out_k, out_v, batch_size,
                     requested_tokens, max_seq_len, kv_heads, head_dim, stream);
}
