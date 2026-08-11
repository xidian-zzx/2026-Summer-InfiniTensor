# Triton Llama 3.2 1B 推理优化

该项目使用 Triton 融合 RMSNorm、residual add + RMSNorm、RoPE、SwiGLU 与 causal GQA attention，保持完整序列重算、模型权重和贪心解码方式不变。

## 目录

- [`baseline/`](./baseline/)：训练营提供的 PyTorch 基线，补充了 ModelScope tokenizer 的 padding 兼容处理。
- [`optimized/`](./optimized/)：Triton 优化实现、CUDA 数值测试、环境与 RTX 3080/4090D 基准记录。
- [`LICENSE`](./LICENSE)：原作业许可证。

模型权重不进入 Git 仓库。可通过 ModelScope 下载推理所需文件：

```bash
pip install modelscope
modelscope download LLM-Research/Llama-3.2-1B \
  config.json generation_config.json model.safetensors \
  special_tokens_map.json tokenizer.json tokenizer_config.json \
  --local-dir /path/to/Llama-3.2-1B
```

进入 `optimized/` 安装依赖并验证：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python infer.py --help
```

完整优化说明、运行命令和实测数据见 [`optimized/README.md`](./optimized/README.md)。
