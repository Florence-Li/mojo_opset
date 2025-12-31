import argparse
import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _env_flag_true(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _resolve_local_files_only(model_id_or_path: str) -> bool:
    if os.path.isdir(os.path.expanduser(model_id_or_path)):
        return True
    return any(
        _env_flag_true(k)
        for k in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_LOCAL_FILES_ONLY",
        )
    )


def _import_torch_qwen3_dense_local():
    this_dir = Path(__file__).resolve().parent
    target = this_dir / "torch_qwen3_dense.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing sibling file: {target}")

    spec = importlib.util.spec_from_file_location("torch_qwen3_dense_local", str(target))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {target}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tqd = _import_torch_qwen3_dense_local()


def _patch_attention_forward_bug():
    # torch_qwen3_dense.Qwen3Attention.forward currently reshapes query_states
    # instead of using the computed attention output. Patch it at runtime.
    if getattr(_tqd.Qwen3Attention.forward, "__name__", "") == "_fixed_forward":
        return

    def _fixed_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        use_cache,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        context_lens = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else torch.zeros(bsz, dtype=torch.long, device=hidden_states.device)
        )

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states).transpose(1, 2)  # [B, Nq, S, D]
        key_states = self.k_norm(key_states).transpose(1, 2)  # [B, Nk, S, D]
        value_states = value_states.transpose(1, 2)  # [B, Nk, S, D]

        cos, sin = position_embeddings
        query_states, key_states = _tqd.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is None:
            raise ValueError("Paged Attention requires a PagedDummyCache instance.")

        attn_output, _ = _tqd.paged_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            past_key_values=past_key_values,
            context_lens=context_lens,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
        return self.o_proj(attn_output), None

    _fixed_forward.__name__ = "_fixed_forward"
    _tqd.Qwen3Attention.forward = _fixed_forward


_patch_attention_forward_bug()


class HFToTorchQwen3Config(_tqd.Qwen3Config):
    def __init__(self, hf_config, num_layers_override: Optional[int] = None):
        super().__init__()

        self.model_type = getattr(hf_config, "model_type", "qwen3")
        self.vocab_size = int(getattr(hf_config, "vocab_size", 151936))

        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_attention_heads = int(hf_config.num_attention_heads)
        self.num_key_value_heads = int(getattr(hf_config, "num_key_value_heads", self.num_attention_heads))
        self.head_dim = int(getattr(hf_config, "head_dim", self.hidden_size // self.num_attention_heads))

        self.attention_bias = bool(getattr(hf_config, "attention_bias", False))
        self.attention_dropout = float(getattr(hf_config, "attention_dropout", 0.0))
        self.hidden_act = getattr(hf_config, "hidden_act", "silu")
        self.rms_norm_eps = float(getattr(hf_config, "rms_norm_eps", 1e-6))

        self.max_position_embeddings = int(getattr(hf_config, "max_position_embeddings", 40960))
        self.rope_theta = float(getattr(hf_config, "rope_theta", 1000000.0))

        full_layers = int(getattr(hf_config, "num_hidden_layers", 36))
        self.num_hidden_layers = int(num_layers_override or full_layers)
        self.layer_types = ["full_attention"] * self.num_hidden_layers

        self.sliding_window = getattr(hf_config, "sliding_window", None)
        self._attn_implementation = getattr(hf_config, "_attn_implementation", "eager")

        self.rope_parameters = getattr(
            hf_config,
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
            },
        )


class TorchQwen3DecoderLayer(nn.Module):
    def __init__(self, config: HFToTorchQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _tqd.Qwen3Attention(config, layer_idx)
        self.mlp = _tqd.Qwen3MLP(config)

        self.input_layernorm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.input_layernorm = _tqd.Qwen3RMSNorm(config.rms_norm_eps, gamma=self.input_layernorm_weight)
        self.post_attention_layernorm = _tqd.Qwen3RMSNorm(config.rms_norm_eps, gamma=self.post_attention_layernorm_weight)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TorchQwen3Model(nn.Module):
    def __init__(self, config: HFToTorchQwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TorchQwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        self.norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.norm = _tqd.Qwen3RMSNorm(config.rms_norm_eps, gamma=self.norm_weight)
        self.rotary = _tqd.Qwen3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["_tqd.PagedDummyCache"] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, "_tqd.PagedDummyCache"]:
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = _tqd.PagedDummyCache(self.config, batch_size=bsz, device=str(device), block_size=16)

        past_len = int(past_key_values.get_seq_length(0).max().item())
        position_ids = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class TorchQwen3ForCausalLM(nn.Module):
    def __init__(self, config: HFToTorchQwen3Config):
        super().__init__()
        self.config = config
        self.model = TorchQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional["_tqd.PagedDummyCache"] = None,
        use_cache: bool = True,
    ):
        hidden_states, past_key_values = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = self.lm_head(hidden_states)
        return logits, past_key_values


def _copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))


def _get_hf_layers(hf_model) -> nn.ModuleList:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "layers"):
        return hf_model.transformer.layers
    raise AttributeError("Unsupported HF model structure: cannot find decoder layers")


def load_weights_from_hf(hf_model, torch_model: TorchQwen3ForCausalLM) -> None:
    hf_layers = _get_hf_layers(hf_model)

    if hasattr(hf_model.model, "embed_tokens"):
        _copy_(torch_model.model.embed_tokens.weight, hf_model.model.embed_tokens.weight)

    for i, layer in enumerate(torch_model.model.layers):
        hf_layer = hf_layers[i]

        _copy_(layer.self_attn.q_proj.weight, hf_layer.self_attn.q_proj.weight)
        _copy_(layer.self_attn.k_proj.weight, hf_layer.self_attn.k_proj.weight)
        _copy_(layer.self_attn.v_proj.weight, hf_layer.self_attn.v_proj.weight)
        _copy_(layer.self_attn.o_proj.weight, hf_layer.self_attn.o_proj.weight)

        _copy_(layer.mlp.gate_proj.weight, hf_layer.mlp.gate_proj.weight)
        _copy_(layer.mlp.up_proj.weight, hf_layer.mlp.up_proj.weight)
        _copy_(layer.mlp.down_proj.weight, hf_layer.mlp.down_proj.weight)

        _copy_(layer.input_layernorm_weight, hf_layer.input_layernorm.weight)
        _copy_(layer.post_attention_layernorm_weight, hf_layer.post_attention_layernorm.weight)

    if hasattr(hf_model.model, "norm"):
        _copy_(torch_model.model.norm_weight, hf_model.model.norm.weight)

    if hasattr(hf_model, "lm_head"):
        _copy_(torch_model.lm_head.weight, hf_model.lm_head.weight)


def build_torch_qwen3_from_hf(
    model_id_or_path: str,
    device: str,
    num_layers: Optional[int] = None,
    trust_remote_code: bool = True,
) -> TorchQwen3ForCausalLM:
    local_files_only = _resolve_local_files_only(model_id_or_path)

    hf_config = AutoConfig.from_pretrained(
        model_id_or_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).eval()

    torch_cfg = HFToTorchQwen3Config(hf_config, num_layers_override=num_layers)

    torch_model = TorchQwen3ForCausalLM(torch_cfg).to(torch.bfloat16).eval()  # keep on CPU for weight copy
    load_weights_from_hf(hf_model, torch_model)

    return torch_model.to(device)


def _try_load_tokenizer(model_id_or_path: str, local_files_only: bool, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_id_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.getenv("QWEN3_MODEL_PATH", ""))
    parser.add_argument("--device", type=str, default=os.getenv("QWEN3_DEVICE", "npu"))
    parser.add_argument("--num_layers", type=int, default=int(os.getenv("QWEN3_NUM_LAYERS", "36")))
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=0)
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError(
            "Please pass --model_path /path/to/qwen3-8b (local dir with config.json & weights), "
            "or set env QWEN3_MODEL_PATH."
        )

    local_files_only = _resolve_local_files_only(args.model_path)

    model = build_torch_qwen3_from_hf(
        args.model_path,
        device=args.device,
        num_layers=args.num_layers,
        trust_remote_code=True,
    )

    tokenizer = _try_load_tokenizer(args.model_path, local_files_only=local_files_only, trust_remote_code=True)
    if tokenizer is not None:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    else:
        input_ids = torch.randint(0, model.config.vocab_size, (1, 16), device=args.device)

    with torch.no_grad():
        logits, _ = model(input_ids)

    print("logits:", tuple(logits.shape), logits.dtype, logits.device)
    print("logits[0, -1, :5] =", logits[0, -1, :5])