# Standalone CUDA RMSNorm

This project implements

```text
output[row, col] = input[row, col]
                 * rsqrt(mean(input[row, :]^2) + eps)
                 * weight[col]
```

for `float` and `half`. Accumulation and `rsqrt` use FP32.

## Optimization structure

- Common widths from 128 through 1024 use one warp per row. Input is loaded in
  aligned packs, cached in registers, reduced with warp shuffles, then combined
  with the weight without reading input again.
- Other widths use one block per row, vectorized packs when alignment permits,
  and a warp-shuffle plus shared-memory block reduction.
- Rows up to 12,288 columns cache converted input in 48 KiB shared memory. Wider
  rows use a two-read fallback to preserve occupancy and support arbitrary sizes.

The design follows the broad dispatch ideas used by OneFlow's RMSNorm CUDA code,
while the kernels and interface here are independently written for this exercise.

## Build and run in WSL2

From this directory:

```bash
make
make test
make bench
./rmsnorm bench 4096 4096 500 half
```

The default binary contains `sm_86` code for RTX 3080 and `sm_89` code for RTX
4090. CUDA Toolkit 11.x does not recognize `sm_89`; build only for the 3080 with:

```bash
make clean
make ARCH_FLAGS='-gencode arch=compute_86,code=sm_86'
```

The public API in `rms_norm.cuh` accepts device pointers and an optional CUDA
stream. The InfiniTensor assignment's `std::vector` wrapper should allocate and
copy device buffers around this kernel launcher.
