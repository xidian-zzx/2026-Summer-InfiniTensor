# CUDA KV Cache

这是一个用于学习和连接 Flash Attention 的连续式 KV Cache，缓存布局与本仓库
Attention 接口一致：

```text
cache_k/cache_v: [batch, max_seq_len, kv_heads, head_dim]
```

项目提供两个操作：

1. `launch_kv_cache_update` 把 `[B, Tnew, Hkv, D]` 的新 K/V 写到
   `positions[B, Tnew]` 指定的缓存位置。
2. `launch_kv_cache_gather` 按 positions 指定的顺序读取缓存，输出紧凑的
   `[B, Trequested, Hkv, D]` 张量。

K/V 在同一个 kernel 中融合搬运；满足对齐条件时，FP32 使用 `float4` 大小的
16 字节 Pack，FP16 使用 8 个 half 的 16 字节 Pack，不规则维度自动使用标量
回退路径。update 只改指定位置，其他缓存内容保持不变。

内核会根据 token 数自动走两条路：

1. `T == 1` 是逐 token 解码。一个 warp 负责一个 KV head，8 个 warp 合成一个
   block；同一 block 只读一次 position，也省掉热循环里的除法和取模。
2. `T > 1` 是批量写入或读取。所有 16 字节 Pack 直接摊平给线程块，数据量
   足够大时更容易把显存带宽吃满。程序会根据 GPU 自动选择 block：RTX 3080
   使用 256 线程，RTX 4090D 使用 512 线程；编译时可以用
   `-DKV_BULK_THREADS=...` 强制覆盖，方便继续实验。

## 构建和运行

```bash
cd /mnt/e/infinitensor/Learning-CUDA-master/InfiniTensor/cuda/KVcache
make test
make bench
./kvcache bench 8 1 4096 8 128 2000 half
```

默认同时生成 RTX 3080（sm_86）和 RTX 4090（sm_89）代码。`positions` 位于
GPU 显存，调用方需要保证位置处于 `[0, max_seq_len)`；同一个 batch 的一次
update 中不要提交重复位置，否则多个线程会同时写同一缓存位置。

## RTX 3080 / RTX 4090D 实测记录

测试日期为 2026-08-10，CUDA 12.8，形状统一为
`B=8, Smax=4096, Hkv=8, D=128`。每项独立运行 3 次，表中记录中位数；
微秒级 kernel 会受 GPU 频率和系统调度影响，小幅波动属于正常现象。

| GPU | 类型 | T | update | gather | update 逻辑带宽 | gather 逻辑带宽 |
|---|---|---:|---:|---:|---:|---:|
| RTX 3080（SM 8.6） | float | 1 | 6.518 us | 6.385 us | 20.1 GB/s | 20.5 GB/s |
| RTX 3080（SM 8.6） | half | 1 | 6.354 us | 6.258 us | 10.3 GB/s | 10.5 GB/s |
| RTX 3080（SM 8.6） | float | 16 | 6.582 us | 6.507 us | 318.6 GB/s | 322.3 GB/s |
| RTX 3080（SM 8.6） | half | 16 | 6.566 us | 6.461 us | 159.7 GB/s | 162.3 GB/s |
| RTX 4090D（SM 8.9） | float | 1 | 2.198 us | 2.161 us | 59.6 GB/s | 60.6 GB/s |
| RTX 4090D（SM 8.9） | half | 1 | 2.188 us | 2.192 us | 29.9 GB/s | 29.9 GB/s |
| RTX 4090D（SM 8.9） | float | 16 | 2.230 us | 2.264 us | 940.3 GB/s | 926.5 GB/s |
| RTX 4090D（SM 8.9） | half | 16 | 2.222 us | 2.207 us | 471.9 GB/s | 475.2 GB/s |

复现命令：

```bash
./kvcache bench 8 1 4096 8 128 30000 both
./kvcache bench 8 16 4096 8 128 10000 both
```

这里的逻辑带宽按 K/V 各读一次、写一次计算。`T=1` 的数据量很小，主要看
kernel 发射延迟；`T=16` 更能体现批量搬运能力。4090D 上另用超过 L2 的大工作
集验证，float/half 的 update 和 gather 均约为 927--931 GB/s。

这是连续式教学实现，适合单卡和固定最大序列长度。生产推理系统常用 Paged KV
Cache，以 block table 管理非连续物理页；那属于下一层内存管理设计。
