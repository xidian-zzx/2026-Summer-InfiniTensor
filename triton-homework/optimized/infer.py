import argparse
import json
import time

import torch
from transformers import AutoTokenizer

import llama


def parse_args():
    parser = argparse.ArgumentParser(description="Triton-optimized Llama inference")
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-warmup-iterations", type=int, default=0)
    parser.add_argument("--num-profiling-iterations", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_profiling_iterations < 1:
        raise ValueError("num-profiling-iterations must be at least 1")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        # Llama 基础模型没单独放 PAD，用 EOS 顶一下就成，权重表也不用跟着改。
        tokenizer.pad_token = tokenizer.eos_token
    # 生成代码总是取最后一个位置，左填充才能让这里真的是 prompt 末尾。
    tokenizer.padding_side = "left"
    inputs = tokenizer(args.prompts, padding=True, return_tensors="pt").to(args.device)
    model = llama.ModelForCausalLM.from_pretrained(args.model).to(args.device)
    texts = []

    # 预热会把 Triton JIT 编译时间挡在计时区间外面，第一次慢一点很正常。
    for _ in range(args.num_warmup_iterations):
        outputs = model.generate(inputs.input_ids, max_new_tokens=args.max_new_tokens)
        texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    if args.device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = 0.0
    for _ in range(args.num_profiling_iterations):
        start_time = time.perf_counter()
        outputs = model.generate(inputs.input_ids, max_new_tokens=args.max_new_tokens)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed_time += time.perf_counter() - start_time
        texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    average_time = elapsed_time / args.num_profiling_iterations
    num_input_tokens = inputs.input_ids.size(-1)
    num_output_tokens = outputs.size(-1) - num_input_tokens
    print(
        json.dumps(
            {
                "texts": texts,
                "average_time": average_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "num_tokens_per_second": num_output_tokens / average_time,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
