# KV Cache for Iluvatar BI-V150

这是连续式 KV Cache，实现 update 和 gather 两个操作，布局为：

```text
cache_k/cache_v: [batch, max_seq_len, kv_heads, head_dim]
new_k/new_v:     [batch, tokens, kv_heads, head_dim]
positions:       [batch, tokens]
```

`update` 把新 K/V 写到 positions 指定的位置，`gather` 按 positions 的顺序读回
紧凑张量。gather 支持乱序和重复位置；update 要求同一个 batch 内的位置互不
冲突。两项操作只搬数据，所以测试直接要求 float/half 逐 bit 一致。

## 实现思路

内核按 token 数分成两条路：

1. `T == 1` 走 decode 小路径，一个 64-lane warp 负责一个 KV head，同一 block
   共用一次 position。参数扫描后每个 block 使用 8 个 warp。
2. `T > 1` 把所有 Pack 铺平成批量任务，使用 256 线程 block；在较大的
   `B=32,T=128` 工作集上，它与 512 线程性能接近，同时 FP16 略占优势。

K/V 在同一个 kernel 里一起搬。满足对齐时每个线程搬 16 字节：FP32 一次
4 个元素，FP16 一次 8 个元素；不规则 head_dim 自动回退成标量。64-lane
映射、host 端 half 的逐 bit 比较和冷热路径原因，都在源码里写了口语化注释。

## 构建与运行

```bash
cd ~/InfiniTensor/iluvatar/KVcache
make
make test
make bench
./kvcache bench 8 1 4096 8 128 3000 both
./kvcache bench 8 16 4096 8 128 1500 both
```

参数仍可以独立重试，例如：

```bash
make clean
make KV_DECODE_WARPS=4 KV_BULK_THREADS=512
```

## 正确性与性能

测试覆盖 decode、批量、16 字节向量化和标量回退，共 5 种形状、两种类型；
10 组结果的 update/gather error 全部为 0。

环境：完整 BI-V150 32 GB、驱动 4.4.0、CoreX 4.4.0。三次运行中位数：

| 类型 | B/T | update | gather | update 逻辑带宽 | gather 逻辑带宽 |
|---|---:|---:|---:|---:|---:|
| float | 8/1 | 3.808 us | 3.719 us | 34.4 GB/s | 35.2 GB/s |
| half | 8/1 | 3.499 us | 3.496 us | 18.7 GB/s | 18.7 GB/s |
| float | 8/16 | 7.640 us | 7.692 us | 274.5 GB/s | 272.6 GB/s |
| half | 8/16 | 5.917 us | 5.948 us | 177.2 GB/s | 176.3 GB/s |
| float | 32/128 | 145.090 us | 146.969 us | 462.5 GB/s | 456.6 GB/s |
| half | 32/128 | 67.988 us | 74.310 us | 493.5 GB/s | 451.5 GB/s |

逻辑带宽按 K/V 各读一次、写一次计算。`T=1` 的数据很少，结果主要受 kernel
发射与调度开销影响；批量用例更适合观察实际搬运能力。
