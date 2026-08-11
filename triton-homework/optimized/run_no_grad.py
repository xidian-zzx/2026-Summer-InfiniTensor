"""在不改动目标 infer.py 的前提下运行无梯度基准。"""

import runpy
import sys
from pathlib import Path

import torch


if len(sys.argv) < 2:
    raise SystemExit("usage: python run_no_grad.py /path/to/infer.py [arguments]")

infer_path = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(infer_path.parent))

# 这份脚本只用来测纯算子性能，提交入口 infer.py 还是原来的执行语义。
torch.set_grad_enabled(False)
runpy.run_path(str(infer_path), run_name="__main__")
