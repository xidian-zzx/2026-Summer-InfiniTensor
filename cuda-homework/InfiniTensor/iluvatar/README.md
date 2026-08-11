# 天垓 BI-V150 算子练习

这里是天数智芯天垓 BI-V150 的实现，目前包含两个可以单独构建的小项目：

```text
iluvatar/
├── RMSnorm/   # 按行 RMSNorm，支持 float/half
└── KVcache/   # 连续式 KV Cache update/gather，支持 float/half
```

实测机器使用一张完整 BI-V150 32 GB，驱动 4.4.0、CoreX 4.4.0，设备报告
compat 7.1，硬件 warp 宽度为 64。CoreX 的 `nvcc` 入口只负责报告版本，本项目
直接调用 `/usr/local/corex-4.4.0/bin/clang++`，以 `ivcore11` 为目标编译 CUDA
风格源码。

两个目录都可以直接运行：

```bash
make
make test
make bench
```

运行时需要能找到 CoreX 动态库，Makefile 已替 `test` 和 `bench` 设置好
`LD_LIBRARY_PATH`。手动执行二进制时可先运行：

```bash
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64
```

正确性测试使用固定随机种子。README 中的性能是同一台服务器三次独立运行的
中位数，逻辑带宽用于比较 kernel 版本，不等同于显存控制器的物理带宽。
