# OpenCL 作业：Q8_0 量化 GEMV

这份作业计算 `C = alpha * A * B^T`，固定 `N=1`，默认尺寸沿用题目给的 `M=122753、K=2304`。权重 `A` 每 32 个数共用一个 fp16 scale，量化值保存成 int8；程序会同时生成 CPU 参考结果、检查误差，再用 OpenCL event 统计纯 kernel 时间。

## 做了哪些优化

1. 内存布局：把 Q8_0 的 int8 权重和 fp16 scale 拆成两个连续 buffer。GPU 读权重时不会夹着 scale 跳来跳去，结构体对齐也不再影响布局。
2. 并行计算：一个 work-group 负责输出的一行，每个 work-item 处理若干完整的 32 元素块，最后用 local memory 做树形归约。基线版保留“一行一个 work-item”，方便直接对比。
3. 低位计算：把激活也按 32 个元素量化成 Q8_0，块内先累加 int8×int8，再乘两边 scale。另附 `dot(float4, float4)` 版，方便观察不同设备对向量指令的优化效果。

权重矩阵有 122753 行，直接塞进常见的 2D image 会超过设备高度限制；这里用线性 buffer 保持行内合并访问。`vload_half` 只把内存中的 fp16 转成 float，所以没有 `cl_khr_fp16` 算术扩展的设备也能编译。

## 目录

```text
opencl-homework/
├── kernels/gemv_q8.cl  # 四个 OpenCL kernel，中文注释也在这里
├── src/main.cpp         # 数据生成、Q8_0 量化、CPU 参考和性能测试
├── CMakeLists.txt
└── Makefile
```

## WSL2 编译运行

Ubuntu 22.04 可以直接装这些包：

```bash
sudo apt update
sudo apt install -y g++ make cmake ocl-icd-opencl-dev clinfo
```

还需要一个 OpenCL ICD。机器有可用的 GPU ICD时直接使用 GPU；只想先检查作业逻辑，可以安装 PoCL：

```bash
sudo apt install -y pocl-opencl-icd
clinfo --list
```

进入本目录后编译和测试：

```bash
make
make quick                                      # M=4096，先快速检查
./build/gemv_opencl --warmup 10 --iterations 100  # 题目原始尺寸
```

CMake 的用法如下：

```bash
cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build -j
./cmake-build/gemv_opencl --quick
```

程序还支持 `--rows`、`--cols`、`--local`、`--warmup`、`--iterations` 和 `--kernel`。其中 `cols` 必须是 32 的倍数，`local` 必须是 2 的幂；想看全部参数可以运行 `./build/gemv_opencl --help`。

## 实测结果

GPU 环境是 Ubuntu 24.04、NVIDIA OpenCL、RTX 4090 D 24GB，驱动 `570.124.06`。原始尺寸 `M=122753、K=2304`，`local=32`，预热 10 次、统计 100 次：

| kernel | 平均时间 | 相对基线 | kernel 最大误差 | 含量化最大误差 |
|---|---:|---:|---:|---:|
| `gemv_q8_fp16_scalar` | 3.9109 ms | 1.0000× | 1.907e-6 | 3.605e-2 |
| `gemv_q8_fp16_local` | 0.7360 ms | 5.3140× | 2.861e-6 | 3.605e-2 |
| `gemv_q8_q8_local` | 0.4981 ms | 7.8522× | 3.338e-6 | 4.638e-2 |
| `gemv_q8_q8_dot4_local` | 0.4979 ms | 7.8544× | 3.338e-6 | 4.638e-2 |

`kernel 最大误差`拿 OpenCL 输出和对应的量化 CPU 参考比较，用来检查 kernel 写对没有；`含量化最大误差`拿结果和原始 fp16 权重、fp16 激活的 CPU 结果比较，也把量化损失算进去了。4090D 上 `local=32` 最快，Q8×Q8 的标量版和 dot4 版性能接近，提交里保留两版方便换设备后继续比较。

本地 WSL2 还用 PoCL/CPU 后端做过交叉验证，四个 kernel 的最大实现误差都在 `3.0e-6` 左右。CPU 与 GPU 的性能数字没有可比性，这组测试只用来确认跨平台编译和计算结果一致。
