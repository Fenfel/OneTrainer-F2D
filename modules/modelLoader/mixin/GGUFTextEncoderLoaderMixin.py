"""
Loads a ready-made quantized GGUF text encoder instead of the stock safetensors TE.

Currently supports only the Flux.2-dev Mistral TE (llama-architecture GGUF quantizations
of Mistral-Small-3.2-24B). The GGUF weights stay packed (quantized) in memory: quantized
linear layers are replaced with diffusers GGUFLinear modules that dequantize per-layer at
forward time, so the full BF16 copy of the TE is never materialized. Only the token
embedding is dequantized eagerly (it is an indexing lookup, not a matmul; same trade-off
ComfyUI makes).

The ggml -> transformers name mapping and the q/k head permutation are adapted from
ComfyUI-GGUF (https://github.com/city96/ComfyUI-GGUF, Apache-2.0).
"""

import os

import torch
from torch import nn

from accelerate import init_empty_weights

from diffusers.quantizers.gguf.utils import (
    GGUFLinear,
    GGUFParameter,
    SUPPORTED_GGUF_QUANT_TYPES,
    dequantize_gguf_tensor,
)

import gguf


# ggml tensor name -> transformers MistralModel name (language model only).
# Adapted from ComfyUI-GGUF LLAMA_SD_MAP (Apache-2.0).
GGML_TO_HF_NAME_MAP = {
    "token_embd": "embed_tokens",
    "output_norm": "norm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_norm": "post_attention_layernorm",
}

# tensors that exist in the GGUF but are not needed for hidden-state encoding
GGUF_IGNORED_TENSORS = ("output.weight",)  # lm_head


class GGUFTextEncoderLoaderMixin:
    def _load_gguf_text_encoder(
            self,
            gguf_path: str,
            config,  # text_config of the base model TE (MistralConfig for Mistral3 base models)
            compute_dtype: torch.dtype,
    ) -> nn.Module:
        """
        Build a text-only MistralModel from the base model config and load the quantized
        weights from the GGUF file without ever materializing the full unquantized model.
        """
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF text encoder not found: {gguf_path}")

        reader = gguf.GGUFReader(gguf_path)

        architecture = self._gguf_field(reader, "general.architecture")
        if architecture != "llama":
            raise ValueError(
                f"GGUF text encoder architecture {architecture!r} is not supported. "
                "Only 'llama' architecture GGUF files (Mistral TE for Flux.2-dev) are supported."
            )

        fields = {key: reader.fields[key].contents() for key in reader.fields if key.startswith("llama.")}
        self._validate_gguf_metadata(fields, config)

        # build the text-only model on meta device: no weights allocated yet.
        # buffers (rope inv_freq) are computed from config, keep them real.
        from transformers import MistralModel
        with init_empty_weights(include_buffers=False):
            model = MistralModel(config)

        state_dict = {}
        quantized_names = set()
        for tensor in reader.tensors:
            hf_name = self._map_ggml_name(tensor.name)
            if hf_name is None:
                continue  # lm_head and other ignored tensors

            quant_type = tensor.tensor_type
            unquantized = quant_type in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
            )
            if not unquantized and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
                raise ValueError(
                    f"GGUF tensor {tensor.name} has unsupported quantization type {quant_type}."
                )

            weights = torch.from_numpy(tensor.data.copy())
            if not unquantized:
                weights = self._permute_packed_rows(weights, tensor.name, config)
                param = GGUFParameter(weights, quant_type=quant_type)
                if hf_name == "embed_tokens.weight":
                    # embedding lookup cannot run on packed rows; dequantize eagerly
                    # (same trade-off as ComfyUI). All other tensors stay packed.
                    state_dict[hf_name] = dequantize_gguf_tensor(param).to(dtype=compute_dtype)
                else:
                    state_dict[hf_name] = param
                    quantized_names.add(hf_name)
            else:
                if quant_type == gguf.GGMLQuantizationType.F32:
                    weights = weights.view(torch.float32)
                elif quant_type == gguf.GGMLQuantizationType.F16:
                    weights = weights.view(torch.float16)
                else:
                    weights = weights.view(torch.bfloat16)
                state_dict[hf_name] = weights.to(dtype=compute_dtype)

        self._check_tensor_coverage(model, state_dict)

        # replace quantized linear layers with GGUFLinear before assigning weights
        for name in quantized_names:
            module_path = name.rsplit(".", 1)[0]
            module = model.get_submodule(module_path)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"GGUF weight {name} maps to a non-linear module {type(module).__name__}")
            parent_path, _, attr = module_path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            with init_empty_weights():
                setattr(parent, attr, GGUFLinear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                ))

        # assign parameters directly: load_state_dict would re-wrap GGUFParameter into a
        # plain nn.Parameter and drop the quant_type metadata
        for name, value in state_dict.items():
            module_path, _, attr = name.rpartition(".")
            module = model.get_submodule(module_path)
            if attr not in module._parameters:
                raise ValueError(f"{module} has no parameter {attr} for GGUF tensor {name}")
            module._parameters[attr] = value if isinstance(value, nn.Parameter) else nn.Parameter(value)

        # GGUFLinear bias was meta-initialized; Mistral linears are bias-free, but be safe
        for module in model.modules():
            if isinstance(module, GGUFLinear) and module.bias is not None:
                module.bias = nn.Parameter(torch.zeros(module.out_features, dtype=compute_dtype))

        model.requires_grad_(False)
        model.eval()
        return model

    @staticmethod
    def _gguf_field(reader, key):
        if key not in reader.fields:
            return None
        return reader.fields[key].contents()

    @staticmethod
    def _validate_gguf_metadata(fields: dict, config) -> None:
        """Fail loudly if the GGUF architecture does not match the base model TE config."""
        checks = [
            ("block_count", lambda: fields.get("llama.block_count"), lambda: config.num_hidden_layers),
            ("embedding_length", lambda: fields.get("llama.embedding_length"), lambda: config.hidden_size),
            ("feed_forward_length", lambda: fields.get("llama.feed_forward_length"), lambda: config.intermediate_size),
            ("attention.head_count", lambda: fields.get("llama.attention.head_count"), lambda: config.num_attention_heads),
            ("attention.head_count_kv", lambda: fields.get("llama.attention.head_count_kv"), lambda: config.num_key_value_heads),
            ("attention.key_length", lambda: fields.get("llama.attention.key_length"), lambda: getattr(config, "head_dim", None)),
            ("vocab_size", lambda: fields.get("llama.vocab_size"), lambda: config.vocab_size),
        ]
        mismatches = []
        for label, gguf_val, cfg_val in checks:
            g, c = gguf_val(), cfg_val()
            if g is not None and c is not None and g != c:
                mismatches.append(f"llama.{label}: GGUF={g} vs config={c}")
        if mismatches:
            raise ValueError(
                "GGUF text encoder is not compatible with the base model text encoder config:\n  "
                + "\n  ".join(mismatches)
            )

    @staticmethod
    def _map_ggml_name(name: str) -> str | None:
        """Map a ggml tensor name to a MistralModel (language model) parameter name.

        Returns None for tensors that are intentionally dropped (lm_head).
        Raises ValueError for tensors that don't fit the expected layout (e.g. vision tower).
        """
        if name in GGUF_IGNORED_TENSORS:
            return None

        if name.startswith("blk."):
            rest = name[len("blk."):]
            layer_idx, _, suffix = rest.partition(".")
            if not layer_idx.isdigit():
                raise ValueError(f"Unexpected GGUF tensor name: {name}")
            base, _, ext = suffix.rpartition(".")
            if ext != "weight" or base not in GGML_TO_HF_NAME_MAP:
                raise ValueError(f"Unexpected GGUF tensor name: {name}")
            return f"layers.{layer_idx}.{GGML_TO_HF_NAME_MAP[base]}.weight"

        if name in ("token_embd.weight", "output_norm.weight"):
            return GGML_TO_HF_NAME_MAP[name.rsplit(".", 1)[0]] + ".weight"

        raise ValueError(f"Unexpected GGUF tensor name: {name}. "
                         "Only text-model (llama architecture) GGUF files are supported; "
                         "vision tower tensors are not.")

    @staticmethod
    def _permute_packed_rows(weights: torch.Tensor, ggml_name: str, config) -> torch.Tensor:
        """Undo the q/k head permutation llama.cpp applies during conversion.

        Each output row of a quantized tensor is a self-contained set of quantization
        blocks, so the head permutation is a pure row permutation on the packed bytes
        and the tensor stays quantized. Adapted from ComfyUI-GGUF llama_permute.
        """
        if not ggml_name.endswith(("attn_q.weight", "attn_k.weight")):
            return weights

        n_head = config.num_attention_heads if ggml_name.endswith("attn_q.weight") else config.num_key_value_heads
        return weights.reshape(
            n_head, weights.shape[0] // n_head // 2, 2, *weights.shape[1:],
        ).swapaxes(1, 2).reshape(weights.shape)

    @staticmethod
    def _check_tensor_coverage(model: nn.Module, state_dict: dict) -> None:
        expected = set(model.state_dict().keys())
        got = set(state_dict.keys())
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing or extra:
            raise ValueError(
                "GGUF text encoder tensors do not match the base model architecture: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
