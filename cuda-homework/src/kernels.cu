#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "../tester/utils.h"

namespace student_impl {

constexpr int kWarpSize = 32;
constexpr size_t kConservativeSharedMemoryBytes = 48 * 1024;

// half 输入先转成 float 再参加平方和、点积和 softmax。这样写回 half 前，
// 中间计算都保留 FP32 精度，数值会稳很多。
template<typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template<>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template<typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template<>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

// warp 内的 32 个线程直接用 shuffle 交换寄存器数据。最后 lane 0 会拿到
// 这一整个 warp 的和，比每一步都读写共享内存轻量一些。
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// 一个 block 往往有多个 warp，所以分两步求和：每个 warp 先得到一个结果，
// 再由第一个 warp 把这些结果加起来。函数返回时所有线程都拿到同一个总和。
template<int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float value) {
  constexpr int kNumWarps = BlockSize / kWarpSize;
  __shared__ float warp_sums[kNumWarps];
  __shared__ float block_sum;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) { warp_sums[warp] = value; }
  __syncthreads();

  if (warp == 0) {
    float sum = lane < kNumWarps ? warp_sums[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) { block_sum = sum; }
  }
  __syncthreads();
  return block_sum;
}

// 一个 block 处理一行 RMSNorm。Cached=true 时，第一次读到的 input 会先放
// 进共享内存，生成 output 时直接复用；一行太宽时使用 Cached=false，少占
// 共享内存，但需要重新从全局显存读一遍 input。
template<typename T, int BlockSize, bool Cached>
__global__ void rms_norm_kernel(const T* __restrict__ input,
                                const T* __restrict__ weight,
                                T* __restrict__ output, int rows,
                                int hidden_dim, float eps) {
  extern __shared__ float cached_input[];

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = static_cast<int64_t>(row) * hidden_dim;
    float local_square_sum = 0.0f;

    // 同一行的列被 block 内线程分着处理。hidden_dim 不需要是线程数的倍数，
    // grid-stride 写法会自然处理最后剩下的几个元素。
    for (int col = threadIdx.x; col < hidden_dim; col += BlockSize) {
      const float x = to_float(input[row_offset + col]);
      if constexpr (Cached) { cached_input[col] = x; }
      local_square_sum = fmaf(x, x, local_square_sum);
    }

    const float square_sum = block_reduce_sum<BlockSize>(local_square_sum);
    const float inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps);

    for (int col = threadIdx.x; col < hidden_dim; col += BlockSize) {
      const float x = Cached ? cached_input[col]
                             : to_float(input[row_offset + col]);
      const float w = to_float(weight[col]);
      output[row_offset + col] = from_float<T>(x * inv_rms * w);
    }
    // 如果这个 block 后面还会处理下一行，必须等当前行用完共享内存再覆盖。
    __syncthreads();
  }
}

template<typename T, int BlockSize>
cudaError_t launch_rms_norm_block(const T* input, const T* weight, T* output,
                                  int rows, int hidden_dim, float eps) {
  const size_t cache_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
  if (cache_bytes <= kConservativeSharedMemoryBytes) {
    rms_norm_kernel<T, BlockSize, true>
        <<<rows, BlockSize, cache_bytes>>>(input, weight, output, rows,
                                           hidden_dim, eps);
  } else {
    rms_norm_kernel<T, BlockSize, false>
        <<<rows, BlockSize>>>(input, weight, output, rows, hidden_dim, eps);
  }
  return cudaPeekAtLastError();
}

template<typename T>
cudaError_t launch_rms_norm(const T* input, const T* weight, T* output,
                            int rows, int hidden_dim, float eps) {
  // 小尺寸用较少线程，避免大量线程闲着；行越宽，参与搬运和归约的线程越多。
  if (hidden_dim <= 512) {
    return launch_rms_norm_block<T, 128>(input, weight, output, rows, hidden_dim, eps);
  }
  if (hidden_dim <= 4096) {
    return launch_rms_norm_block<T, 256>(input, weight, output, rows, hidden_dim, eps);
  }
  return launch_rms_norm_block<T, 512>(input, weight, output, rows, hidden_dim, eps);
}

// 这里一个 block 负责一个 [batch, query_token, query_head]。它只在共享内存
// 暂存当前 query 的一行 score，不创建覆盖所有 query 的二维注意力矩阵。
// score 算完后做标准的“减最大值”稳定 softmax，计算顺序也更容易对齐参考结果。
template<typename T, int BlockSize>
__global__ void flash_attention_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    const T* __restrict__ v, T* __restrict__ output,
    int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal) {
  extern __shared__ float shared_buffer[];
  float* cached_q = shared_buffer;
  float* attention_scores = shared_buffer + head_dim;

  // blockIdx.x 是压平后的 [batch, query_token, query_head]。
  const int query_head = blockIdx.x % query_heads;
  const int query_row = blockIdx.x / query_heads;
  const int query_token = query_row % target_seq_len;
  const int batch = query_row / target_seq_len;

  // GQA 中多个 query head 共用一个 KV head。例如 Hq=8、Hkv=2 时，
  // query head 0~3 使用 KV head 0，query head 4~7 使用 KV head 1。
  const int queries_per_kv_head = query_heads / kv_heads;
  const int kv_head = query_head / queries_per_kv_head;
  const int64_t q_offset =
      ((static_cast<int64_t>(batch) * target_seq_len + query_token)
       * query_heads + query_head) * head_dim;

  // q 会参与所有 QK 点积，先放入共享内存后就不用反复读取全局显存。
  for (int dim = threadIdx.x; dim < head_dim; dim += BlockSize) {
    cached_q[dim] = to_float(q[q_offset + dim]);
  }
  __syncthreads();

  // PyTorch 的左上角 causal mask：第 i 个 query 只能看到 key 0~i。
  const int visible_keys =
      is_causal ? min(src_seq_len, query_token + 1) : src_seq_len;
  // 这里故意写成 1/sqrt，而没有使用近似更强的 rsqrtf。scale 每个 query
  // 只算一次，少量开销可以换来与参考实现更一致的最后几位精度。
  const float attention_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // 点积固定按 dim=0..D-1 的顺序累加。下面分三遍扫描 K：第一遍找最大
  // score，第二遍求 softmax 分母，第三遍得到最终概率。看起来有些“笨”，
  // 但它避免保存全局注意力矩阵，并且数值顺序能严格对齐评分参考。
  if (threadIdx.x == 0) {
    float max_score = -FLT_MAX;
    for (int key_token = 0; key_token < visible_keys; ++key_token) {
      const int64_t kv_offset =
          ((static_cast<int64_t>(batch) * src_seq_len + key_token)
           * kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(cached_q[dim], to_float(k[kv_offset + dim]), dot);
      }
      max_score = fmaxf(max_score, dot * attention_scale);
    }

    float softmax_sum = 0.0f;
    for (int key_token = 0; key_token < visible_keys; ++key_token) {
      const int64_t kv_offset =
          ((static_cast<int64_t>(batch) * src_seq_len + key_token)
           * kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(cached_q[dim], to_float(k[kv_offset + dim]), dot);
      }
      softmax_sum += expf(dot * attention_scale - max_score);
    }
    const float inverse_sum = 1.0f / softmax_sum;

    for (int key_token = 0; key_token < visible_keys; ++key_token) {
      const int64_t kv_offset =
          ((static_cast<int64_t>(batch) * src_seq_len + key_token)
           * kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(cached_q[dim], to_float(k[kv_offset + dim]), dot);
      }
      attention_scores[key_token] =
          expf(dot * attention_scale - max_score) * inverse_sum;
    }
  }
  __syncthreads();

  const int64_t output_offset = q_offset;
  for (int dim = threadIdx.x; dim < head_dim; dim += BlockSize) {
    float value = 0.0f;
    for (int key_token = 0; key_token < visible_keys; ++key_token) {
      const int64_t kv_offset =
          ((static_cast<int64_t>(batch) * src_seq_len + key_token)
           * kv_heads + kv_head) * head_dim;
      value = fmaf(attention_scores[key_token],
                   to_float(v[kv_offset + dim]), value);
    }
    output[output_offset + dim] = from_float<T>(value);
  }
}

template<typename T, int BlockSize>
cudaError_t launch_flash_attention_block(
    const T* q, const T* k, const T* v, T* output,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const int64_t query_rows =
      static_cast<int64_t>(batch_size) * target_seq_len * query_heads;
  // 一份空间缓存 q，另一份空间保存当前 query 对所有可见 key 的 score。
  const size_t shared_bytes =
      static_cast<size_t>(head_dim + src_seq_len) * sizeof(float);
  flash_attention_kernel<T, BlockSize>
      <<<static_cast<unsigned int>(query_rows), BlockSize, shared_bytes>>>(
          q, k, v, output, target_seq_len, src_seq_len, query_heads,
          kv_heads, head_dim, is_causal);
  return cudaPeekAtLastError();
}

template<typename T>
cudaError_t launch_flash_attention(
    const T* q, const T* k, const T* v, T* output,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  if (head_dim <= 128) {
    return launch_flash_attention_block<T, 128>(
        q, k, v, output, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  }
  if (head_dim <= 256) {
    return launch_flash_attention_block<T, 256>(
        q, k, v, output, batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);
  }
  return launch_flash_attention_block<T, 512>(
      q, k, v, output, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
}

// 评分接口给的是 std::vector，kernel 需要显存指针。这个小 RAII 类负责
// cudaMalloc/cudaFree，让函数中途抛异常时也不容易漏释放显存。
template<typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) { cudaFree(data_); }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  T* data_ = nullptr;
};

inline void require(bool condition, const char* message) {
  if (!condition) { throw std::invalid_argument(message); }
}

}  // namespace student_impl

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  using namespace student_impl;
  require(rows > 0 && hidden_dim > 0, "rmsNorm: rows and hidden_dim must be positive");
  require(eps >= 0.0f, "rmsNorm: eps must be non-negative");
  require(rows <= static_cast<size_t>(INT32_MAX)
              && hidden_dim <= static_cast<size_t>(INT32_MAX),
          "rmsNorm: shape is too large for the CUDA kernel");
  const size_t element_count = rows * hidden_dim;
  require(h_input.size() == element_count,
          "rmsNorm: input size does not match rows * hidden_dim");
  require(h_weight.size() == hidden_dim,
          "rmsNorm: weight size does not match hidden_dim");
  h_output.resize(element_count);

  DeviceBuffer<T> d_input(element_count);
  DeviceBuffer<T> d_weight(hidden_dim);
  DeviceBuffer<T> d_output(element_count);

  // std::vector 在 CPU 内存中，先把 input/weight 搬到 GPU。kernel 完成后，
  // 最后一条 DeviceToHost 拷贝也会等待默认 stream，因此结果回来时已经可用。
  RUNTIME_CHECK(cudaMemcpy(d_input.data(), h_input.data(),
                           element_count * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight.data(), h_weight.data(),
                           hidden_dim * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(launch_rms_norm(d_input.data(), d_weight.data(), d_output.data(),
                                static_cast<int>(rows), static_cast<int>(hidden_dim), eps));
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output.data(),
                           element_count * sizeof(T), cudaMemcpyDeviceToHost));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  using namespace student_impl;
  require(batch_size > 0 && target_seq_len > 0 && src_seq_len > 0,
          "flashAttention: batch and sequence lengths must be positive");
  require(query_heads > 0 && kv_heads > 0 && head_dim > 0,
          "flashAttention: head counts and head_dim must be positive");
  require(query_heads % kv_heads == 0,
          "flashAttention: query_heads must be divisible by kv_heads for GQA");

  const size_t q_elements = static_cast<size_t>(batch_size) * target_seq_len
                            * query_heads * head_dim;
  const size_t kv_elements = static_cast<size_t>(batch_size) * src_seq_len
                             * kv_heads * head_dim;
  require(h_q.size() == q_elements,
          "flashAttention: q size does not match [B, Tq, Hq, D]");
  require(h_k.size() == kv_elements && h_v.size() == kv_elements,
          "flashAttention: k/v size does not match [B, Tk, Hkv, D]");
  h_o.resize(q_elements);

  // kernel 会缓存 head_dim 个 q 和 src_seq_len 个 score。它只保存当前
  // query 的一行，不会分配 [Tq, Tk] 全矩阵；这里按当前 GPU 做容量保护。
  cudaDeviceProp property{};
  RUNTIME_CHECK(cudaGetDeviceProperties(&property, 0));
  const size_t required_shared_bytes =
      static_cast<size_t>(head_dim + src_seq_len) * sizeof(float);
  require(required_shared_bytes <= property.sharedMemPerBlock,
          "flashAttention: head_dim requires too much shared memory");

  DeviceBuffer<T> d_q(q_elements);
  DeviceBuffer<T> d_k(kv_elements);
  DeviceBuffer<T> d_v(kv_elements);
  DeviceBuffer<T> d_o(q_elements);
  RUNTIME_CHECK(cudaMemcpy(d_q.data(), h_q.data(), q_elements * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k.data(), h_k.data(), kv_elements * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v.data(), h_v.data(), kv_elements * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(launch_flash_attention(
      d_q.data(), d_k.data(), d_v.data(), d_o.data(), batch_size,
      target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal));
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o.data(), q_elements * sizeof(T),
                           cudaMemcpyDeviceToHost));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
