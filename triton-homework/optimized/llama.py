import dataclasses
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from triton_kernels import (
    add_rms_norm,
    apply_rope,
    causal_gqa_attention,
    rms_norm,
    swiglu,
)


@dataclasses.dataclass
class ModelConfig:
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    vocab_size: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input):
        return rms_norm(input, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, input):
        gate = self.gate_proj(input)
        up = self.up_proj(input)
        # 两个投影还是交给 cuBLAS，激活和相乘由 Triton 一把收掉。
        return self.down_proj(swiglu(gate, up))


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )
    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    return torch.sin(positions * theta), torch.cos(positions * theta)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )
    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        # RoPE 和 GQA 都直接在原始头数上算，避免显式复制 K/V。
        query_states = apply_rope(query_states, sin_table, cos_table).permute(0, 2, 1, 3)
        key_states = apply_rope(key_states, sin_table, cos_table).permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)
        attention = causal_gqa_attention(query_states, key_states, value_states)
        attention = attention.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(attention)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        attention_update = self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )
        hidden_states, normalized = add_rms_norm(
            hidden_states,
            attention_update,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return hidden_states + self.mlp(normalized)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.torch_dtype = config.torch_dtype
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )
        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]
        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        # 把“残差相加 + 下一层 RMSNorm”跨层串起来，省掉一批小 kernel。
        normalized = self.layers[0].input_layernorm(hidden_states)
        for layer_index, layer in enumerate(self.layers):
            attention_update = layer.self_attn(normalized, sin_table, cos_table)
            hidden_states, normalized = add_rms_norm(
                hidden_states,
                attention_update,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )
            mlp_update = layer.mlp(normalized)

            if layer_index + 1 < self.num_hidden_layers:
                next_norm = self.layers[layer_index + 1].input_layernorm
            else:
                next_norm = self.norm
            hidden_states, normalized = add_rms_norm(
                hidden_states,
                mlp_update,
                next_norm.weight,
                next_norm.eps,
            )

        return normalized


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)
        with open(model_path / "config.json", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)
        if "head_dim" not in config_dict:
            config_dict["head_dim"] = (
                config_dict["hidden_size"] // config_dict["num_attention_heads"]
            )
        config = ModelConfig(
            **{
                key: value
                for key, value in config_dict.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))
        state_dict = load_file(model_path / "model.safetensors")
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        model.load_state_dict(state_dict)
        return model
