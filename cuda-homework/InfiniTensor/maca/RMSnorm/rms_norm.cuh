#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 这两个 CUDA 头由 MXMACA 的 cu-bridge 提供，底层调用最终会落到 MACA
// Runtime。接口保留 CUDA 风格，后面接入已有推理代码时不用再包一层转换。

// 对外接口使用 GPU 指针：input、weight、output 都必须已经在显存里。
// input/output 可以看成 [rows, hidden_dim]，weight 是 [hidden_dim]。
// 每一行会单独计算平方均值，所有行共用同一份 weight。
cudaError_t launch_rms_norm(const float* input, const float* weight, float* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream = nullptr);

cudaError_t launch_rms_norm(const half* input, const half* weight, half* output,
                            int rows, int hidden_dim, float eps,
                            cudaStream_t stream = nullptr);
