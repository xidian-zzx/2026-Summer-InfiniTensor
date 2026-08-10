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

这是连续式教学实现，适合单卡和固定最大序列长度。生产推理系统常用 Paged KV
Cache，以 block table 管理非连续物理页；那属于下一层内存管理设计。
