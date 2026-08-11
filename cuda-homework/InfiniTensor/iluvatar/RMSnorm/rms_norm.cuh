#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CoreX 直接提供 CUDA 风格 Runtime，因此接口仍使用 cudaError_t/cudaStream_t。
// kernel 内部会按 BI-V150 的 64-lane warp 单独调度，不能照搬 32-lane 参数。

// 对外接口使用 GPU 指针：input、weight、output 都必须已经在显存里。
// input/output 可以看成 [rows, hidden_dim]，weight 是 [hidden_dim]。
// 每一行会单独计算平方均值，所有行共用同一份 weight。
cudaError_t launch_rms_norm(const float* input, const float* weight, float* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream = nullptr);

cudaError_t launch_rms_norm(const half* input, const half* weight, half* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream = nullptr);
