# InfiniTensor Triton Llama 优化

该目录给原作业中的 Llama 3 推理加入 Triton 融合算子，保持模型权重、贪心解码流程和完整序列重算方式不变。没有引入 KV cache，也没有用 PyTorch 内置高性能算子替换作业目标算子。

## 优化内容

推理热路径压缩为三类 Triton kernel：

1. 注意力：融合 causal mask、在线 softmax、GQA 头映射和 `QK^T @ V`，K/V 不再通过 `repeat_interleave` 复制，也不再落盘完整注意力矩阵。
2. 逐元素融合：RoPE、SwiGLU，以及 residual add + RMSNorm 均各自单趟完成，减少中间张量和 kernel launch。
3. 跨层调度：上一子层的残差相加与下一子层的 RMSNorm 合并；首层保持独立 RMSNorm，末层直接产出最终归一化结果。

CUDA/Triton 不可用时，`triton_kernels.py` 自动使用等价 PyTorch 实现，便于 CPU 检查；性能评测必须使用 CUDA。

## WSL2 环境

当前机器的 CUDA 编译器位于 `/usr/local/cuda-12.8`。建议单独建环境，避免改动现有项目：

```bash
cd /mnt/e/infinitensor/go-llama-go-master/InfiniTensor/triton
conda create -n infinitensor-triton python=3.11 -y
conda activate infinitensor-triton
python -m pip install -r requirements.txt
```

环境安装本身不会运行 GPU。确认依赖与静态回退路径：

```bash
python -m pytest -q
```

GPU 空闲后再执行 CUDA kernel 正确性测试；同一条命令会自动运行原先跳过的 CUDA 用例。

ModelScope 可直接下载本项目需要的单文件权重，显式列出文件能避开重复的 `consolidated.00.pth`：

```bash
modelscope download LLM-Research/Llama-3.2-1B \
  config.json generation_config.json model.safetensors \
  special_tokens_map.json tokenizer.json tokenizer_config.json \
  --local-dir /root/models/Llama-3.2-1B
```

## 推理与基准

先在仓库根目录运行原始实现，再进入本目录运行优化实现，参数保持一致。至少设置一次预热，让 Triton JIT 编译不进入计时区间：

```bash
python infer.py \
  --model /data/shared/models/Llama-3.2-1B/ \
  --prompts \
  "The emergence of deep learning domain-specific languages (DSLs) has substantially reduced the obstacles in developing high-performance, cross-platform compute kernels, but current DSLs" \
  "Driven by recent advancements in the AI industry, the AI accelerator sector has increasingly diversified, with vendors developing their own hardware architectures and programming models, such as NVIDIA" \
  --max-new-tokens 64 \
  --device cuda \
  --num-warmup-iterations 1 \
  --num-profiling-iterations 3
```

验收时检查两项：生成文本应与原实现相同或仅有可接受的浮点误差分歧；优化版 `num_tokens_per_second` 相对原版提升至少 80%。

## 本机 CUDA 验收记录

2026-08-10 在 RTX 3080 20 GB、CUDA 12.8、PyTorch 2.11、Triton 3.6 环境完成了以下检查：

1. CUDA 算子测试共 6 项，RMSNorm、residual + RMSNorm、SwiGLU、RoPE 和多种序列长度的 causal GQA attention 全部通过。
2. 使用 Llama-3.2-1B 主体尺寸的合成模型测试，固定长度前向由 41.572 ms 降至 12.766 ms，达到 3.256 倍加速；隔离进程中的 16-token 生成耗时稳定在约 0.170 秒，相对原版稳定样本达到约 4.1 倍加速。
3. 本机尚未找到 Llama 权重，仅发现结构不兼容的 Qwen2-7B；因此真实生成文本和官方参数下的最终 `num_tokens_per_second` 仍需拿到 Llama 模型路径后复测。合成随机权重模型会放大 BF16 误差，不能用于判断文本质量。

## RTX 4090D 真实模型记录

服务器环境为 RTX 4090D 24 GB、CUDA 12.8、PyTorch 2.6 NVIDIA 版、Triton 3.1，模型为 ModelScope `LLM-Research/Llama-3.2-1B`：

1. 最终 CUDA 数值测试为 `6 passed`；64-token 生成的四轮文本保持一致，两条长 prompt 均能生成连贯技术内容。
2. 原始作业入口在 64-token 预热阶段占用 23.51 GB 后 OOM；优化入口完整运行，最终实测为 `114.07 token/s`。8-token 同入口对照中，原版为 `53.95 token/s`，优化版为 `112.47 token/s`，达到 2.08 倍加速。
3. 为分离原版 autograd 显存问题，使用 `run_no_grad.py` 对双方做保守的纯算子对照：原版 `77.31 token/s`；完成 4090D tile 调优后的 Triton 版 `131.47 token/s`，达到 1.70 倍。该辅助脚本不改变提交入口的语义。

## RTX 3080 真实模型记录

本地环境为 RTX 3080 20 GB、CUDA 12.8、PyTorch 2.11、Triton 3.6，使用同一份 ModelScope Llama-3.2-1B 权重：

1. CUDA 数值测试为 `6 passed`；8-token 原入口直接对照的四轮文本逐字一致，原版 `32.52 token/s`，优化版 `120.23 token/s`，达到 3.70 倍加速。
2. 64-token 无梯度保守对照中，原版 `72.91 token/s`，优化版 `113.50 token/s`，达到 1.56 倍；第一条文本完全一致，第二条仅出现不影响语义的 BF16 舍入分歧。
3. 优化版按作业原入口完成 64-token、1 次预热、3 次计时，得到 `121.93 token/s`，四轮输出一致且内容连贯。

## 文件说明

- `llama.py`：保持原权重命名的模型实现与跨层融合调度。
- `triton_kernels.py`：RMSNorm、residual + RMSNorm、RoPE、SwiGLU、causal GQA attention。
- `infer.py`：与原作业参数兼容的推理入口。
- `run_no_grad.py`：只用于隔离 autograd 影响的保守性能对照。
- `tests/test_kernels.py`：CPU 回退与 CUDA 数值对照测试。
