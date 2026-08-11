# RMSNorm for MetaX C500

输入按 `[rows, hidden_dim]` 连续存放，每一行独立归一化，所有行共用一份
`weight[hidden_dim]`：

```text
mean_square = sum(x[j] * x[j]) / hidden_dim
y[j] = x[j] * rsqrt(mean_square + eps) * weight[j]
```

## 实现思路

代码按尺寸走三条路径：

1. 128～1024 这类常见宽度由一个逻辑 warp 处理一行，input 暂存在寄存器里，
   归约后无需再读一次。
2. 常见的大宽度由一个 block 处理一行。C500 实测使用 512 线程更合适；FP32
   把 input 缓存在共享内存，减少第二次全局读取。
3. FP16 转成 float 缓存会让共享数据膨胀两倍。C500 上直接读两遍 input 更快，
   因此 FP16 默认走 uncached 路径；超宽行也用这条稳妥的回退路径。

平方和始终使用 float 累加。满足对齐时，FP32 使用 float4、FP16 使用 half8
大小的 16 字节 Pack；127、4103 等不规则宽度会自动回退到标量版本。

C500 的物理 wavefront 是 64 线程，cu-bridge 提供 32 线程 CUDA 逻辑 warp。
shuffle 归约按逻辑 warp 编写，并已覆盖常见宽度和不规则宽度测试。

## 构建与运行

```bash
cd ~/InfiniTensor/maca/RMSnorm
make
make test
make bench
./rmsnorm bench 4096 4096 500 both
```

需要重新实验 block 大小时，可以直接覆盖 Makefile 参数：

```bash
make clean
make RMS_BLOCK_THREADS=256 RMS_CACHE_HALF=1
```

## 正确性与性能

测试覆盖 8 种形状、float/half 两种类型，共 16 组。FP32 最大绝对误差不超过
`4.768e-7`，FP16 不超过 `9.770e-4`，全部 PASS。

环境：C500 25% sGPU、16 GB 配额、MACA 3.3.0.15。三次运行中位数：

| 类型 | rows | hidden | 时间 | 逻辑带宽 |
|---|---:|---:|---:|---:|
| float | 4096 | 4096 | 101.894 us | 1975.9 GB/s |
| half | 4096 | 4096 | 66.794 us | 1507.1 GB/s |

相对最初的 256-thread 通用版本，FP32 从约 120.9 us 降到 101.9 us；FP16
从约 98.0 us 降到 66.8 us。这里的逻辑带宽按 input、weight、output 的算法
访问量统计，weight 会命中缓存，因此它可以高于物理显存带宽。
