#include "rms_norm.cuh"

#include <type_traits>

namespace {

#ifndef RMS_BLOCK_THREADS
#define RMS_BLOCK_THREADS 256
#endif

#ifndef RMS_USE_SHARED_CACHE
#define RMS_USE_SHARED_CACHE 1
#endif

#ifndef RMS_CACHE_HALF
#define RMS_CACHE_HALF 1
#endif

// 整个文件有三条执行路径：
// 1. 常见的小尺寸用一个 warp 处理一行，并把 input 暂存在寄存器里；
// 2. 其他尺寸优先用一个 block 处理一行，并用共享内存缓存 input；
// 3. 一行大到共享内存放不下时，再用读取两遍 input 的回退版本。
// 这种分法兼顾了常见模型尺寸的性能和任意 hidden_dim 的正确性。
// BI-V150 一个 warp 有 64 个 lane。这里直接按硬件宽度做归约；如果仍按
// NVIDIA 的 32 lane 分工，后半个 warp 会读错 Pack，平方和也会少算一半。
constexpr int kWarpSize = 64;
constexpr int kWarpsPerBlock = 4;
constexpr int kWarpKernelThreads = kWarpSize * kWarpsPerBlock;
constexpr int kConservativeSharedMemoryBytes = 48 * 1024;
constexpr int kMainBlockThreads = RMS_BLOCK_THREADS;
static_assert(kMainBlockThreads == 128 || kMainBlockThreads == 256
                  || kMainBlockThreads == 512,
              "RMS_BLOCK_THREADS must be 128, 256 or 512");

// 把连续 N 个元素打成一个小包。例如 Pack<float, 4> 是 16 字节，GPU
// 可以用更少的访存指令一次搬运四个 float。alignas 用来保证地址对齐。
template<typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T elem[N];
};

// input 可以是 float 或 half，平方和始终用 float 累加。half 直接累加
// 容易损失精度，所以读取后先转 float，最后写回时再转回 half。
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

// warp 内归约。32 个线程通过 shuffle 直接交换寄存器中的值，不需要
// 每一步都经过共享内存。循环结束后，lane 0 拿到整个 warp 的总和。
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down(value, offset, kWarpSize);
  }
  return value;
}

// 常见 Transformer 宽度走这条路径。
//
// 一个 block 有 4 个 warp，因此能同时处理 4 行。以 hidden_dim=1024、
// float4 为例，每个线程读取 8 个 float4，32 * 8 * 4 正好覆盖一整行。
// x 暂存在 cached_x 寄存器数组中，归约完成后可以直接拿它生成 output，
// 省掉第二次从全局显存读取 input。
template<typename T, int PackSize, int PacksPerThread>
__global__ void rms_norm_warp_kernel(const T* __restrict__ input,
                                     const T* __restrict__ weight,
                                     T* __restrict__ output, int rows,
                                     int hidden_dim, float eps) {
  using PackT = Pack<T, PackSize>;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + warp_in_block;
  // 最后一个 block 可能凑不满 4 行，多出来的 warp 直接退出。
  if (row >= rows) { return; }

  // 进入这条 kernel 前已经保证 hidden_dim 与 PackSize 匹配，所以每一行
  // 的起始地址仍满足 PackT 的对齐要求。
  const PackT* row_input = reinterpret_cast<const PackT*>(input + row * hidden_dim);
  PackT* row_output = reinterpret_cast<PackT*>(output + row * hidden_dim);
  const PackT* packed_weight = reinterpret_cast<const PackT*>(weight);
  float cached_x[PacksPerThread][PackSize];
  float square_sum = 0.0f;

  // 相邻 lane 访问相邻的 Pack，32 个线程的访问可以合并成连续显存请求。
#pragma unroll
  for (int pack_i = 0; pack_i < PacksPerThread; ++pack_i) {
    const int pack_index = lane + pack_i * kWarpSize;
    const PackT x_pack = row_input[pack_index];
#pragma unroll
    for (int item = 0; item < PackSize; ++item) {
      const float x = to_float(x_pack.elem[item]);
      cached_x[pack_i][item] = x;
      // fmaf(x, x, sum) 就是 sum += x*x，通常会生成一条融合乘加指令。
      square_sum = fmaf(x, x, square_sum);
    }
  }

  // lane 0 得到完整平方和，再算 inv_rms 并广播给同一 warp 的所有线程。
  square_sum = warp_reduce_sum(square_sum);
  const float inv_rms = __shfl(
      rsqrtf(square_sum / static_cast<float>(hidden_dim) + eps), 0, kWarpSize);

  // 第二阶段：直接使用寄存器里的 x，只需再读取 weight 并写出 output。
#pragma unroll
  for (int pack_i = 0; pack_i < PacksPerThread; ++pack_i) {
    const int pack_index = lane + pack_i * kWarpSize;
    const PackT w_pack = packed_weight[pack_index];
    PackT y_pack;
#pragma unroll
    for (int item = 0; item < PackSize; ++item) {
      y_pack.elem[item] = from_float<T>(
          cached_x[pack_i][item] * inv_rms * to_float(w_pack.elem[item]));
    }
    row_output[pack_index] = y_pack;
  }
}

// block 路径的归约分两层完成：
//
// 每个线程的局部和 -> warp 内求和 -> 每个 warp 留下一个结果
// -> 第一个 warp 再求一次和 -> 得到整行的 inv_rms。
//
// 返回前有 __syncthreads()，所以所有线程拿到的是同一个有效结果。
template<int BlockSize>
__device__ __forceinline__ float finish_block_reduction(float thread_sum,
                                                        float eps,
                                                        int hidden_dim) {
  constexpr int kNumWarps = BlockSize / kWarpSize;
  __shared__ float warp_sums[kNumWarps];
  __shared__ float shared_inv_rms;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  thread_sum = warp_reduce_sum(thread_sum);
  // 每个 warp 只让 lane 0 写结果，避免多个线程写同一个位置。
  if (lane == 0) { warp_sums[warp] = thread_sum; }
  __syncthreads();

  // 例如 256 线程只有 8 个 warp，所以 lane 0~7 读取有效值，其他 lane
  // 用 0 补齐。这样可以继续复用同一个 warp_reduce_sum。
  if (warp == 0) {
    float block_sum = lane < kNumWarps ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      shared_inv_rms = rsqrtf(block_sum / static_cast<float>(hidden_dim) + eps);
    }
  }
  __syncthreads();
  return shared_inv_rms;
}

// 共享内存缓存路径：适合 hidden_dim 较大，但整行转成 float 后仍然能放进
// 48 KiB 共享内存的情况。一个 block 一次处理一行，因此不同行不会互相覆盖。
template<typename T, int PackSize, int BlockSize>
__global__ void rms_norm_block_cached_kernel(const T* __restrict__ input,
                                             const T* __restrict__ weight,
                                             T* __restrict__ output, int rows,
                                             int hidden_dim, float eps) {
  using PackT = Pack<T, PackSize>;
  extern __shared__ float cached_x[];
  const int num_packs = hidden_dim / PackSize;

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const PackT* row_input = reinterpret_cast<const PackT*>(input + row * hidden_dim);
    float thread_sum = 0.0f;

    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += BlockSize) {
      const PackT x_pack = row_input[pack_i];
      const int col = pack_i * PackSize;
#pragma unroll
      for (int item = 0; item < PackSize; ++item) {
        const float x = to_float(x_pack.elem[item]);
        cached_x[col + item] = x;
        thread_sum = fmaf(x, x, thread_sum);
      }
    }

    // 这个函数内部也完成了同步。返回时 cached_x 已经写好，所有线程也都
    // 拿到了相同的 inv_rms。
    const float inv_rms = finish_block_reduction<BlockSize>(thread_sum, eps, hidden_dim);
    const PackT* packed_weight = reinterpret_cast<const PackT*>(weight);
    PackT* row_output = reinterpret_cast<PackT*>(output + row * hidden_dim);

    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += BlockSize) {
      const PackT w_pack = packed_weight[pack_i];
      const int col = pack_i * PackSize;
      PackT y_pack;
#pragma unroll
      for (int item = 0; item < PackSize; ++item) {
        y_pack.elem[item] = from_float<T>(
            cached_x[col + item] * inv_rms * to_float(w_pack.elem[item]));
      }
      row_output[pack_i] = y_pack;
    }
    // 当前 block 可能接着处理下一行。先确保大家已经用完 cached_x，防止
    // 下一行提前覆盖共享内存。
    __syncthreads();
  }
}

// 超宽行回退路径：hidden_dim 太大时，缓存整行会耗尽共享内存，甚至让
// kernel 无法启动。因此先读 input 算平方和，得到 inv_rms 后再读一次
// input 生成 output。代价是多读一遍，但可以稳定支持任意大的行。
template<typename T, int PackSize, int BlockSize>
__global__ void rms_norm_block_uncached_kernel(const T* __restrict__ input,
                                               const T* __restrict__ weight,
                                               T* __restrict__ output, int rows,
                                               int hidden_dim, float eps) {
  using PackT = Pack<T, PackSize>;
  const int num_packs = hidden_dim / PackSize;

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const PackT* row_input = reinterpret_cast<const PackT*>(input + row * hidden_dim);
    float thread_sum = 0.0f;
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += BlockSize) {
      const PackT x_pack = row_input[pack_i];
#pragma unroll
      for (int item = 0; item < PackSize; ++item) {
        const float x = to_float(x_pack.elem[item]);
        thread_sum = fmaf(x, x, thread_sum);
      }
    }

    const float inv_rms = finish_block_reduction<BlockSize>(thread_sum, eps, hidden_dim);
    const PackT* packed_weight = reinterpret_cast<const PackT*>(weight);
    PackT* row_output = reinterpret_cast<PackT*>(output + row * hidden_dim);
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += BlockSize) {
      const PackT x_pack = row_input[pack_i];
      const PackT w_pack = packed_weight[pack_i];
      PackT y_pack;
#pragma unroll
      for (int item = 0; item < PackSize; ++item) {
        y_pack.elem[item] = from_float<T>(to_float(x_pack.elem[item]) * inv_rms
                                         * to_float(w_pack.elem[item]));
      }
      row_output[pack_i] = y_pack;
    }
    __syncthreads();
  }
}

template<typename T, int PackSize, int BlockSize>
cudaError_t launch_block_path(const T* input, const T* weight, T* output,
                              int rows, int hidden_dim, float eps,
                              cudaStream_t stream) {
  const size_t cache_bytes = static_cast<size_t>(hidden_dim) * sizeof(float);
  // half 输入也会转成 float 后再缓存，所以这里始终使用 sizeof(float)。
  constexpr bool cache_this_type =
      std::is_same<T, float>::value || RMS_CACHE_HALF;
  if (RMS_USE_SHARED_CACHE && cache_this_type
      && cache_bytes <= kConservativeSharedMemoryBytes) {
    rms_norm_block_cached_kernel<T, PackSize, BlockSize>
        <<<rows, BlockSize, cache_bytes, stream>>>(input, weight, output, rows,
                                                   hidden_dim, eps);
  } else {
    rms_norm_block_uncached_kernel<T, PackSize, BlockSize>
        <<<rows, BlockSize, 0, stream>>>(input, weight, output, rows,
                                        hidden_dim, eps);
  }
  return cudaPeekAtLastError();
}

template<typename T, int PackSize>
cudaError_t dispatch_block_size(const T* input, const T* weight, T* output,
                                int rows, int hidden_dim, float eps,
                                cudaStream_t stream) {
  // 数据少时用小 block，避免很多线程闲着；数据越多就增加线程数。
  // BlockSize 是模板参数，编译器还能把归约循环更充分地展开。
  if (hidden_dim <= 512) {
    return launch_block_path<T, PackSize, 128>(input, weight, output, rows,
                                               hidden_dim, eps, stream);
  }
  if (hidden_dim <= 4096) {
    return launch_block_path<T, PackSize, kMainBlockThreads>(
        input, weight, output, rows, hidden_dim, eps, stream);
  }
  return launch_block_path<T, PackSize, 512>(input, weight, output, rows,
                                             hidden_dim, eps, stream);
}

template<typename T, int PackSize, int PacksPerThread>
cudaError_t launch_warp_path(const T* input, const T* weight, T* output,
                             int rows, int hidden_dim, float eps,
                             cudaStream_t stream) {
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  rms_norm_warp_kernel<T, PackSize, PacksPerThread>
      <<<blocks, kWarpKernelThreads, 0, stream>>>(input, weight, output, rows,
                                                 hidden_dim, eps);
  return cudaPeekAtLastError();
}

template<typename T>
cudaError_t launch_impl(const T* input, const T* weight, T* output, int rows,
                        int hidden_dim, float eps, cudaStream_t stream) {
  // 先挡住空指针、非法形状和负数 eps，避免把明显错误带进 kernel。
  // CUDA 风格接口返回错误码，具体怎么打印或退出交给调用方决定。
  if (input == nullptr || weight == nullptr || output == nullptr
      || rows <= 0 || hidden_dim <= 0 || eps < 0.0f) {
    return cudaErrorInvalidValue;
  }

  // 常见尺寸优先走寄存器缓存的 warp kernel。
  //
  // BI-V150 的 warp 有 64 个 lane。下面按 64 重新配 Pack：能用 16 字节就
  // 用 16 字节，384/640/896 这些尺寸则降到更小的 Pack，保证每个 lane
  // 拿到同样多的数据。
  if constexpr (std::is_same<T, float>::value) {
    switch (hidden_dim) {
      case 128:  return launch_warp_path<T, 2, 1>(input, weight, output, rows, hidden_dim, eps, stream);
      case 256:  return launch_warp_path<T, 4, 1>(input, weight, output, rows, hidden_dim, eps, stream);
      case 384:  return launch_warp_path<T, 2, 3>(input, weight, output, rows, hidden_dim, eps, stream);
      case 512:  return launch_warp_path<T, 4, 2>(input, weight, output, rows, hidden_dim, eps, stream);
      case 640:  return launch_warp_path<T, 2, 5>(input, weight, output, rows, hidden_dim, eps, stream);
      case 768:  return launch_warp_path<T, 4, 3>(input, weight, output, rows, hidden_dim, eps, stream);
      case 896:  return launch_warp_path<T, 2, 7>(input, weight, output, rows, hidden_dim, eps, stream);
      case 1024: return launch_warp_path<T, 4, 4>(input, weight, output, rows, hidden_dim, eps, stream);
      default: break;
    }
    // 尺寸不在上面的表里，只要能被 4 整除，block 路径仍使用 float4。
    if (hidden_dim % 4 == 0) {
      return dispatch_block_size<T, 4>(input, weight, output, rows, hidden_dim, eps, stream);
    }
  } else {
    switch (hidden_dim) {
      case 128:  return launch_warp_path<T, 2, 1>(input, weight, output, rows, hidden_dim, eps, stream);
      case 256:  return launch_warp_path<T, 4, 1>(input, weight, output, rows, hidden_dim, eps, stream);
      case 384:  return launch_warp_path<T, 2, 3>(input, weight, output, rows, hidden_dim, eps, stream);
      case 512:  return launch_warp_path<T, 8, 1>(input, weight, output, rows, hidden_dim, eps, stream);
      case 640:  return launch_warp_path<T, 2, 5>(input, weight, output, rows, hidden_dim, eps, stream);
      case 768:  return launch_warp_path<T, 4, 3>(input, weight, output, rows, hidden_dim, eps, stream);
      case 896:  return launch_warp_path<T, 2, 7>(input, weight, output, rows, hidden_dim, eps, stream);
      case 1024: return launch_warp_path<T, 8, 2>(input, weight, output, rows, hidden_dim, eps, stream);
      default: break;
    }
    // half 优先一次处理 8 个元素，不能整除时再降到 4 个元素。
    if (hidden_dim % 8 == 0) {
      return dispatch_block_size<T, 8>(input, weight, output, rows, hidden_dim, eps, stream);
    }
    if (hidden_dim % 4 == 0) {
      return dispatch_block_size<T, 4>(input, weight, output, rows, hidden_dim, eps, stream);
    }
  }
  // 最后的标量路径不要求任何整除关系，负责兜底 127、4103 这类尺寸。
  return dispatch_block_size<T, 1>(input, weight, output, rows, hidden_dim, eps, stream);
}

}  // namespace

// 对外只暴露两个直观的重载，模板细节和各类 kernel 都留在当前源文件内部。
cudaError_t launch_rms_norm(const float* input, const float* weight, float* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream) {
  return launch_impl(input, weight, output, rows, hidden_dim, eps, stream);
}

cudaError_t launch_rms_norm(const half* input, const half* weight, half* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream) {
  return launch_impl(input, weight, output, rows, hidden_dim, eps, stream);
}
