"""Lightweight regression tests for the save/backup path: no unnecessary full copies.

The safetensors hash must stream over the tensors without materializing a full byte copy of
every tensor (the old .tobytes() did), and _convert_state_dict_dtype with dtype=None must not
copy tensors that are already on the CPU (torch .to() returns self when nothing changes).
Tiny synthetic tensors only - no model weights, no GPU.
"""

import hashlib

import pytest

from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin

import torch


class _Saver(DtypeModelSaverMixin):
    def __init__(self):
        super().__init__()


def _reference_hash(state_dict) -> str:
    """The pre-optimization implementation: one full .tobytes() copy per tensor."""
    sha = hashlib.sha256()
    for _key, tensor in sorted(state_dict.items()):
        # exactly the old production implementation, byte for byte
        sha.update(tensor.contiguous().cpu().flatten().view(torch.uint8).numpy().tobytes())
    return f"0x{sha.hexdigest()}"


class TestStreamingSafetensorsHash:
    def test_hash_matches_tobytes_reference(self):
        gen = torch.Generator().manual_seed(42)
        state_dict = {
            "b.weight": torch.randn(64, 32, generator=gen),
            "a.weight": torch.randn(8, 128, generator=gen),
            "a.bias": torch.arange(8, dtype=torch.float32),
        }
        saver = _Saver()
        streamed = saver._DtypeModelSaverMixin__calculate_safetensors_hash(state_dict)
        assert streamed == _reference_hash(state_dict)

    def test_hash_independent_of_dict_insertion_order(self):
        gen = torch.Generator().manual_seed(7)
        a, b = torch.randn(16, generator=gen), torch.randn(16, generator=gen)
        saver = _Saver()
        h1 = saver._DtypeModelSaverMixin__calculate_safetensors_hash({"a": a, "b": b})
        h2 = saver._DtypeModelSaverMixin__calculate_safetensors_hash({"b": b, "a": a})
        assert h1 == h2

    def test_hash_handles_non_contiguous_and_multidtype(self):
        gen = torch.Generator().manual_seed(3)
        base = torch.randn(32, 16, generator=gen)
        state_dict = {
            "transpose": base.t(),  # non-contiguous
            "fp16": base[:4].to(torch.float16),
            "bf16": base[:4].to(torch.bfloat16),
        }
        saver = _Saver()
        h = saver._DtypeModelSaverMixin__calculate_safetensors_hash(state_dict)
        assert h == _reference_hash(state_dict)

    def test_none_state_dict_returns_none(self):
        assert _Saver()._DtypeModelSaverMixin__calculate_safetensors_hash(None) is None


class TestConvertStateDictDtypeNoCopies:
    def test_dtype_none_keeps_cpu_tensors_as_is(self):
        # backup path: dtype=None. Tensors already on the CPU must not be copied - torch .to()
        # returns the same tensor when device and dtype do not change.
        t = torch.randn(1000, 500)
        converted = _Saver()._convert_state_dict_dtype({"w": t}, None)
        assert converted["w"] is t

    def test_dtype_conversion_produces_new_tensors(self):
        t = torch.randn(64, 64)
        converted = _Saver()._convert_state_dict_dtype({"w": t}, torch.float16)
        assert converted["w"] is not t
        assert converted["w"].dtype == torch.float16
        # fp16 has ~3 decimal digits: compare with a relative tolerance, an absolute 1e-3 is
        # marginal for values around 4 and flakes between runs
        assert torch.allclose(converted["w"].float(), t, rtol=2e-3, atol=1e-3)

    def test_nested_dicts_are_recursed(self):
        t = torch.randn(8)
        converted = _Saver()._convert_state_dict_dtype({"outer": {"inner": t}}, torch.float16)
        assert converted["outer"]["inner"].dtype == torch.float16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
