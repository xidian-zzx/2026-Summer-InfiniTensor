# MetaX C500 算子练习

这里放的是沐曦 C500 版本，当前包含两个可以单独构建的小项目：

```text
maca/
├── RMSnorm/   # 按行 RMSNorm，支持 float/half
└── KVcache/   # 连续式 KV Cache update/gather，支持 float/half
```

实测环境为 MetaX C500 的 25% sGPU，显存配额 16 GB，驱动 3.8.30，
MACA 3.3.0.15。源码通过官方 cu-bridge 编译，默认工具链位置是 `/opt/maca`。
cu-bridge 让接口继续保持 CUDA 风格，kernel 最终仍在 C500 上执行。

两个目录都使用相同命令：

```bash
make
make test
make bench
```

正确性测试使用固定随机种子，方便复现。README 中的性能数据是三次独立运行
的中位数；实例是算力切片，所以这些结果用于当前环境内比较优化前后，不代表
完整 C500 整卡性能。
