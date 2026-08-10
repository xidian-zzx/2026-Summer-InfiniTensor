// Q8_0 GEMV：A 是按 32 个元素一组量化的权重，B 是一行激活。
// 这里把 q 和 scale 分开存，GPU 读起来更连续，也不会被结构体对齐坑到。

inline float load_f16(__global const ushort *data, int index) {
    // vload_half 会把内存里的 IEEE fp16 直接转成 float，设备不用支持 fp16 算术也能跑。
    return vload_half(index, (__global const half *)data);
}

__kernel void gemv_q8_fp16_scalar(
    __global const char *weight_q,
    __global const ushort *weight_scale,
    __global const ushort *activation,
    __global const char *activation_q,
    __global const ushort *activation_scale,
    __global float *output,
    const int rows,
    const int cols,
    const float alpha,
    __local float *scratch) {

    const int row = (int)get_global_id(0);
    if (row >= rows) return;

    const int blocks = cols / 32;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const float scale = load_f16(weight_scale, row * blocks + block);
        const int offset = block * 32;
        float block_sum = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            const int col = offset + lane;
            block_sum += (float)weight_q[row * cols + col] * load_f16(activation, col);
        }
        sum += block_sum * scale;
    }
    output[row] = alpha * sum;
}

__kernel void gemv_q8_fp16_local(
    __global const char *weight_q,
    __global const ushort *weight_scale,
    __global const ushort *activation,
    __global const char *activation_q,
    __global const ushort *activation_scale,
    __global float *output,
    const int rows,
    const int cols,
    const float alpha,
    __local float *scratch) {

    const int row = (int)get_group_id(0);
    const int tid = (int)get_local_id(0);
    const int local_size = (int)get_local_size(0);
    const int blocks = cols / 32;

    // 一个线程负责若干完整量化块，scale 只读一次，接着顺手算完块内 32 个数。
    float partial = 0.0f;
    for (int block = tid; block < blocks; block += local_size) {
        const float scale = load_f16(weight_scale, row * blocks + block);
        const int offset = block * 32;
        float block_sum = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            const int col = offset + lane;
            block_sum += (float)weight_q[row * cols + col] * load_f16(activation, col);
        }
        partial += block_sum * scale;
    }

    // 先把每个线程的小计放进共享的 local memory，再做树形归约。
    scratch[tid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) output[row] = alpha * scratch[0];
}

__kernel void gemv_q8_q8_local(
    __global const char *weight_q,
    __global const ushort *weight_scale,
    __global const ushort *activation,
    __global const char *activation_q,
    __global const ushort *activation_scale,
    __global float *output,
    const int rows,
    const int cols,
    const float alpha,
    __local float *scratch) {

    const int row = (int)get_group_id(0);
    const int tid = (int)get_local_id(0);
    const int local_size = (int)get_local_size(0);
    const int blocks = cols / 32;

    float partial = 0.0f;
    for (int block = tid; block < blocks; block += local_size) {
        int int_sum = 0;
        const int offset = block * 32;
        for (int lane = 0; lane < 32; ++lane) {
            const int col = offset + lane;
            int_sum += (int)weight_q[row * cols + col] * (int)activation_q[col];
        }
        const float scale_a = load_f16(weight_scale, row * blocks + block);
        const float scale_b = load_f16(activation_scale, block);
        partial += (float)int_sum * scale_a * scale_b;
    }

    scratch[tid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) output[row] = alpha * scratch[0];
}

__kernel void gemv_q8_q8_dot4_local(
    __global const char *weight_q,
    __global const ushort *weight_scale,
    __global const ushort *activation,
    __global const char *activation_q,
    __global const ushort *activation_scale,
    __global float *output,
    const int rows,
    const int cols,
    const float alpha,
    __local float *scratch) {

    const int row = (int)get_group_id(0);
    const int tid = (int)get_local_id(0);
    const int local_size = (int)get_local_size(0);
    const int blocks = cols / 32;

    float partial = 0.0f;
    for (int block = tid; block < blocks; block += local_size) {
        float int_sum = 0.0f;
        const int offset = block * 32;
        // 四个 int8 一次读进来，再交给 dot；比一项一项读更容易生成向量指令。
        for (int lane = 0; lane < 32; lane += 4) {
            const char4 a4 = vload4(0, weight_q + row * cols + offset + lane);
            const char4 b4 = vload4(0, activation_q + offset + lane);
            int_sum += dot(convert_float4(a4), convert_float4(b4));
        }
        const float scale_a = load_f16(weight_scale, row * blocks + block);
        const float scale_b = load_f16(activation_scale, block);
        partial += int_sum * scale_a * scale_b;
    }

    scratch[tid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) output[row] = alpha * scratch[0];
}
