# RMSNorm for Iluvatar BI-V150

输入按 `[rows, hidden_dim]` 连续存放，每一行单独归一化，所有行共用一份
`weight[hidden_dim]`：

```text
mean_square = sum(x[j] * x[j]) / hidden_dim
y[j] = x[j] * rsqrt(mean_square + eps) * weight[j]
```

## 实现思路

代码按尺寸走三条路径：

1. 128～1024 这类常见宽度由一个 warp 处理一行，input 先放在寄存器里，
   求完平方和后可以直接写结果。
2. 其他常见宽度由一个 block 处理一行。BI-V150 实测 256 线程较稳，FP32
   用共享内存暂存 input，少做一次全局读取。
3. FP16 放进 float 共享缓存会把数据量放大一倍，在这张卡上直接读两遍更快；
   超宽行放不进共享内存时也走这条回退路径。

平方和统一用 float 累加。对齐时 FP32 使用 float4、FP16 使用 half8 这样的
16 字节 Pack；127、4103 等不规则宽度自动回退为标量版本。BI-V150 一个
warp 有 64 个 lane，shuffle 归约和线程分工都按 64 编写，这里是跨平台时最
容易顺手写错的地方，源码里也留了比较直白的注释。

## 构建与运行

```bash
cd ~/InfiniTensor/iluvatar/RMSnorm
make
make test
make bench
./rmsnorm bench 4096 4096 500 both
```

需要重新试 block 大小或 FP16 缓存策略时，可覆盖 Makefile 参数：

```bash
make clean
make RMS_BLOCK_THREADS=128 RMS_CACHE_HALF=1
```

## 正确性与性能

测试覆盖 8 种形状和 float/half 两种类型，共 16 组；FP32 最大绝对误差不超过
`4.768e-7`，FP16 不超过 `9.770e-4`，全部 PASS。

环境：完整 BI-V150 32 GB、驱动 4.4.0、CoreX 4.4.0。三次运行中位数：

| 类型 | rows | hidden | 时间 | 逻辑带宽 |
|---|---:|---:|---:|---:|
| float | 4096 | 4096 | 533.597 us | 377.3 GB/s |
| half | 4096 | 4096 | 249.629 us | 403.3 GB/s |

FP16 改成不缓存 input 后，时间从约 468 us 降到约 250 us。这里的逻辑带宽按
input、weight、output 的算法访问量计算，weight 的缓存命中会让它与物理显存
带宽的统计口径不同。
