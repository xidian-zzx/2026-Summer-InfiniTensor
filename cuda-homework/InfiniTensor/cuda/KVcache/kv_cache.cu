#include "kv_cache.cuh"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

// KV Cache 的 update 和 gather 本质上都是“搬数据”，没有复杂计算。
// 优化重点是让每个线程一次搬一小包连续元素，并让相邻线程访问相邻地址。
template<typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T elem[N];
};

// 把 new_k 和 new_v 放在同一个 kernel 中处理。两个张量使用完全相同的
// 位置映射，融合后只需要读取一次 positions，也少一次 kernel 启动开销。
template<typename T, int PackSize>
__global__ void kv_cache_update_kernel(
    const T* __restrict__ new_k, const T* __restrict__ new_v,
    const int* __restrict__ positions, T* __restrict__ cache_k,
    T* __restrict__ cache_v, int new_tokens, int max_seq_len,
    int elements_per_token, int packs_per_token, int64_t total_packs) {
  using PackT = Pack<T, PackSize>;
  const PackT* packed_new_k = reinterpret_cast<const PackT*>(new_k);
  const PackT* packed_new_v = reinterpret_cast<const PackT*>(new_v);
  PackT* packed_cache_k = reinterpret_cast<PackT*>(cache_k);
  PackT* packed_cache_v = reinterpret_cast<PackT*>(cache_v);

  // grid-stride loop 让 kernel 能处理任意数量的 token。即使为了控制 block
  // 数量把 grid 截到 4096，每个线程也会继续处理后面的 Pack。
  for (int64_t pack_id = static_cast<int64_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       pack_id < total_packs;
       pack_id += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // pack_id 是整个 [B, Tnew, H, D] 的线性 Pack 编号。先除以
    // packs_per_token 找到它属于第几个 token，再把 token_id 拆成 batch
    // 和 batch 内的 token。可以把这几行理解成“把一维下标还原成多维坐标”。
    const int64_t token_id = pack_id / packs_per_token;
    const int pack_in_token = static_cast<int>(pack_id % packs_per_token);
    const int batch = static_cast<int>(token_id / new_tokens);
    const int token = static_cast<int>(token_id % new_tokens);
    const int position = positions[batch * new_tokens + token];

    // positions 位于显存，host 启动函数无法逐个检查。这里保留边界保护，
    // 防止错误位置造成越界写；正常调用应提前保证每个位置都合法。
    if (static_cast<unsigned int>(position)
        >= static_cast<unsigned int>(max_seq_len)) {
      continue;
    }

    // 源张量是 [B, Tnew, H, D]，pack_id 本身就是源 Pack 的线性下标。
    // 目标张量是 [B, Smax, H, D]，只需把 token 维换成 position。
    const int64_t destination_element =
        (static_cast<int64_t>(batch) * max_seq_len + position)
        * elements_per_token;
    const int64_t destination_pack =
        destination_element / PackSize + pack_in_token;
    packed_cache_k[destination_pack] = packed_new_k[pack_id];
    packed_cache_v[destination_pack] = packed_new_v[pack_id];
  }
}

// gather 与 update 的地址映射方向相反：从大缓存的指定 position 读取，
// 再连续写入紧凑的 [B, Trequested, H, D] 输出。
template<typename T, int PackSize>
__global__ void kv_cache_gather_kernel(
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
    // 输出是连续存放的，所以 pack_id 直接表示输出位置；下面只需要算出
    // 对应的 batch 和 position，找到大缓存中的源地址。
    const int64_t token_id = pack_id / packs_per_token;
    const int pack_in_token = static_cast<int>(pack_id % packs_per_token);
    const int batch = static_cast<int>(token_id / requested_tokens);
    const int token = static_cast<int>(token_id % requested_tokens);
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

inline int choose_blocks(int64_t work_items) {
  // 小任务按实际工作量启动 block；大任务最多启动 4096 个 block，剩余工作
  // 交给 kernel 内的 grid-stride loop。block 太多只会增加调度开销。
  const int64_t needed = (work_items + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed, kMaxBlocks)));
}

template<typename T, int PackSize>
cudaError_t launch_update_pack(
    const T* new_k, const T* new_v, const int* positions,
    T* cache_k, T* cache_v, int batch_size, int new_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  const int elements_per_token = kv_heads * head_dim;
  // 一个 token 的 K 或 V 有 Hkv * D 个元素。除以 PackSize 后，就得到
  // 搬完一个 token 需要多少次 Pack 访问。
  const int packs_per_token = elements_per_token / PackSize;
  const int64_t total_packs =
      static_cast<int64_t>(batch_size) * new_tokens * packs_per_token;
  kv_cache_update_kernel<T, PackSize>
      <<<choose_blocks(total_packs), kThreads, 0, stream>>>(
          new_k, new_v, positions, cache_k, cache_v, new_tokens, max_seq_len,
          elements_per_token, packs_per_token, total_packs);
  return cudaPeekAtLastError();
}

template<typename T, int PackSize>
cudaError_t launch_gather_pack(
    const T* cache_k, const T* cache_v, const int* positions,
    T* out_k, T* out_v, int batch_size, int requested_tokens,
    int max_seq_len, int kv_heads, int head_dim, cudaStream_t stream) {
  const int elements_per_token = kv_heads * head_dim;
  const int packs_per_token = elements_per_token / PackSize;
  const int64_t total_packs =
      static_cast<int64_t>(batch_size) * requested_tokens * packs_per_token;
  kv_cache_gather_kernel<T, PackSize>
      <<<choose_blocks(total_packs), kThreads, 0, stream>>>(
          cache_k, cache_v, positions, out_k, out_v, requested_tokens,
          max_seq_len, elements_per_token, packs_per_token, total_packs);
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

  // Pack 的边界不能跨 token，否则目标 position 改变后地址就不再连续。
  // 因此按照 kv_heads * head_dim 能否整除来选择向量宽度。
  const int elements_per_token = kv_heads * head_dim;
  if constexpr (std::is_same<T, half>::value) {
    if (elements_per_token % 8 == 0) {
      return launch_update_pack<T, 8>(new_k, new_v, positions, cache_k, cache_v,
                                      batch_size, new_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  } else {
    if (elements_per_token % 4 == 0) {
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

  const int elements_per_token = kv_heads * head_dim;
  if constexpr (std::is_same<T, half>::value) {
    if (elements_per_token % 8 == 0) {
      return launch_gather_pack<T, 8>(cache_k, cache_v, positions, out_k, out_v,
                                      batch_size, requested_tokens, max_seq_len,
                                      kv_heads, head_dim, stream);
    }
  } else {
    if (elements_per_token % 4 == 0) {
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
