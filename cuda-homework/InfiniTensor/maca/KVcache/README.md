# KV Cache for MetaX C500

这是连续式 KV Cache，实现 update 和 gather 两个操作。缓存布局保持为：

```text
cache_k/cache_v: [batch, max_seq_len, kv_heads, head_dim]
new_k/new_v:     [batch, tokens, kv_heads, head_dim]
positions:       [batch, tokens]
```

`update` 把新 K/V 写到 positions 指定的位置；`gather` 按 positions 给出的顺序
读出紧凑张量。gather 支持乱序和重复位置，update 要求同一 batch 内的位置互不
冲突。两条操作都只搬数据，因此测试要求 float/half 结果逐 bit 相同。

## 实现思路

内核按 token 数分流：

1. `T == 1` 是逐 token decode。一个逻辑 warp 处理一个 KV head，同一 block
   共享一次 position，并省掉热循环里的除法和取模。
2. `T > 1` 是批量路径。所有 Pack 摊平给线程块，C500 参数扫描后选用 512
   线程；相比 256 线程，当前 `T=16` 用例大约快 9%～11%。

K/V 在同一个 kernel 中一起搬运。满足对齐时，每个线程使用 16 字节 Pack：
FP32 一次 4 个元素，FP16 一次 8 个元素；不规则 head_dim 自动回退到标量搬运。
默认参数仍可通过 `KV_DECODE_WARPS` 和 `KV_BULK_THREADS` 重新实验。

## 构建与运行

```bash
cd ~/InfiniTensor/maca/KVcache
make
make test
make bench
./kvcache bench 8 1 4096 8 128 20000 both
./kvcache bench 8 16 4096 8 128 10000 both
```

例如重新测试 256-thread bulk：

```bash
make clean
make KV_BULK_THREADS=256
```

## 正确性与性能

测试覆盖 decode、批量、16 字节向量化和标量回退，共 5 种形状、float/half
两种类型，10 组结果的 update/gather error 均为 0。

环境：C500 25% sGPU、16 GB 配额、MACA 3.3.0.15。三次运行中位数：

| 类型 | T | update | gather | update 逻辑带宽 | gather 逻辑带宽 |
|---|---:|---:|---:|---:|---:|
| float | 1 | 3.851 us | 4.148 us | 34.0 GB/s | 31.6 GB/s |
| half | 1 | 3.829 us | 4.171 us | 17.1 GB/s | 15.7 GB/s |
| float | 16 | 7.252 us | 7.618 us | 289.2 GB/s | 275.3 GB/s |
| half | 16 | 6.141 us | 6.394 us | 170.7 GB/s | 164.0 GB/s |

逻辑带宽按 K/V 各读一次、写一次计算。`T=1` 数据量很小，结果主要反映 kernel
发射与调度开销；`T=16` 更适合观察批量搬运能力。
