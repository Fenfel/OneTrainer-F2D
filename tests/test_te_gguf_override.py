"""Tests for the GGUF text-encoder override (Flux.2 Mistral GGUF TE).

Covers: config round-trip, ModelNames plumbing, loader dispatch errors,
ggml->transformers name mapping, metadata validation, frozen-TE guard.
"""

import json
import os

import pytest

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames


def _flux2_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.FLUX_2
    return config


class TestConfigPlumbing:
    def test_empty_override_keeps_old_behaviour(self):
        config = _flux2_config()
        names = config.model_names()
        assert names.text_encoder_model == ""

    def test_config_roundtrip_preserves_te_override(self):
        config = _flux2_config()
        config.text_encoder.model_name = "C:/models/Mistral-Small-3.2-Q4_K_M.gguf"

        data = json.loads(json.dumps(config.to_dict()))

        config2 = TrainConfig.default_values()
        config2.from_dict(data)

        assert config2.text_encoder.model_name == "C:/models/Mistral-Small-3.2-Q4_K_M.gguf"
        assert config2.model_names().text_encoder_model == "C:/models/Mistral-Small-3.2-Q4_K_M.gguf"

    def test_model_names_accepts_text_encoder_model(self):
        names = ModelNames(text_encoder_model="some/te.gguf")
        assert names.text_encoder_model == "some/te.gguf"


class TestLoaderDispatch:
    def test_non_gguf_override_raises_directly(self):
        # an override that is not a .gguf file must fail loudly, not fall back to the full TE
        from modules.modelLoader.Flux2ModelLoader import Flux2ModelLoader

        loader = Flux2ModelLoader()
        names = ModelNames(base_model="does/not/matter", text_encoder_model="some/te.safetensors")
        with pytest.raises(ValueError, match="GGUF"):
            loader.load(
                model=None,
                model_type=ModelType.FLUX_2,
                model_names=names,
                weight_dtypes=None,
                quantization=None,
            )

    def test_missing_gguf_file_raises(self, tmp_path):
        from modules.modelLoader.Flux2ModelLoader import Flux2ModelLoader

        loader = Flux2ModelLoader()
        names = ModelNames(base_model="does/not/matter", text_encoder_model=str(tmp_path / "missing.gguf"))
        with pytest.raises(Exception, match="missing.gguf"):
            loader.load(
                model=None,
                model_type=ModelType.FLUX_2,
                model_names=names,
                weight_dtypes=None,
                quantization=None,
            )


class TestGGUFNameMapping:
    def test_maps_ggml_names_to_mistral_model(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin

        map_name = GGUFTextEncoderLoaderMixin._map_ggml_name
        assert map_name("token_embd.weight") == "embed_tokens.weight"
        assert map_name("output_norm.weight") == "norm.weight"
        assert map_name("blk.0.attn_q.weight") == "layers.0.self_attn.q_proj.weight"
        assert map_name("blk.39.attn_k.weight") == "layers.39.self_attn.k_proj.weight"
        assert map_name("blk.7.attn_v.weight") == "layers.7.self_attn.v_proj.weight"
        assert map_name("blk.7.attn_output.weight") == "layers.7.self_attn.o_proj.weight"
        assert map_name("blk.7.attn_norm.weight") == "layers.7.input_layernorm.weight"
        assert map_name("blk.7.ffn_norm.weight") == "layers.7.post_attention_layernorm.weight"
        assert map_name("blk.7.ffn_gate.weight") == "layers.7.mlp.gate_proj.weight"
        assert map_name("blk.7.ffn_up.weight") == "layers.7.mlp.up_proj.weight"
        assert map_name("blk.7.ffn_down.weight") == "layers.7.mlp.down_proj.weight"

    def test_lm_head_is_dropped(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
        assert GGUFTextEncoderLoaderMixin._map_ggml_name("output.weight") is None

    def test_unknown_tensor_raises(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
        with pytest.raises(ValueError, match="v.blk"):
            GGUFTextEncoderLoaderMixin._map_ggml_name("v.blk.0.attn_q.weight")


class TestMetadataValidation:
    @staticmethod
    def _text_config():
        from transformers import Mistral3Config
        return Mistral3Config(
            hidden_size=5120,
            intermediate_size=32768,
            num_hidden_layers=40,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            rms_norm_eps=1e-5,
            vocab_size=131072,
            rope_theta=1e8,
        )

    @staticmethod
    def _gguf_fields():
        return {
            "llama.block_count": 40,
            "llama.embedding_length": 5120,
            "llama.feed_forward_length": 32768,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128,
            "llama.attention.value_length": 128,
            "llama.attention.layer_norm_rms_epsilon": 1e-5,
            "llama.vocab_size": 131072,
            "llama.rope.freq_base": 1e8,
        }

    def test_matching_metadata_passes(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
        GGUFTextEncoderLoaderMixin._validate_gguf_metadata(self._gguf_fields(), self._text_config())

    def test_mismatched_metadata_raises(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
        fields = self._gguf_fields()
        fields["llama.block_count"] = 32
        with pytest.raises(ValueError, match="block_count"):
            GGUFTextEncoderLoaderMixin._validate_gguf_metadata(fields, self._text_config())

    def test_mismatched_vocab_raises(self):
        from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
        fields = self._gguf_fields()
        fields["llama.vocab_size"] = 128000
        with pytest.raises(ValueError, match="vocab"):
            GGUFTextEncoderLoaderMixin._validate_gguf_metadata(fields, self._text_config())


class TestCacheIsolation:
    @staticmethod
    def _text_cache_dirs(config):
        from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
        from mgds.pipelineModules.DiskCache import DiskCache

        class _Host(DataLoaderText2ImageMixin):
            # minimal concrete subclass so the mixin's cache factories can be called normally
            def _preparation_modules(self, config, model):
                return []

            def _cache_modules(self, config, model, model_setup):
                return []

            def _output_modules(self, config, model, model_setup):
                return []

            def _debug_modules(self, config, model):
                return []

        modules = _Host()._cache_modules_from_names(
            model=None,
            model_setup=None,
            image_split_names=["latent_image"],
            image_aggregate_names=["image_path"],
            text_split_names=["tokens"],
            sort_names=["prompt"],
            config=config,
            text_caching=True,
        )
        return [m.cache_dir for m in modules if isinstance(m, DiskCache)]

    def test_te_override_isolates_text_cache(self):
        config = _flux2_config()
        config.cache_dir = "C:/tmp/ot-cache"
        config.latent_caching = True
        config.text_encoder.model_name = "C:/models/te.gguf"

        image_dir, text_dir = self._text_cache_dirs(config)
        assert image_dir == os.path.join("C:/tmp/ot-cache", "image")
        assert text_dir != os.path.join("C:/tmp/ot-cache", "text")
        assert text_dir.startswith(os.path.join("C:/tmp/ot-cache", "text") + os.sep + "te-")

    def test_no_override_keeps_shared_text_cache(self):
        config = _flux2_config()
        config.cache_dir = "C:/tmp/ot-cache"
        config.latent_caching = True

        image_dir, text_dir = self._text_cache_dirs(config)
        assert text_dir == os.path.join("C:/tmp/ot-cache", "text")

    def test_different_overrides_get_different_dirs(self):
        config = _flux2_config()
        config.cache_dir = "C:/tmp/ot-cache"
        config.latent_caching = True
        config.text_encoder.model_name = "C:/models/a.gguf"
        dir_a = self._text_cache_dirs(config)[1]
        config.text_encoder.model_name = "C:/models/b.gguf"
        dir_b = self._text_cache_dirs(config)[1]
        assert dir_a != dir_b


class TestFrozenTEGuard:
    def test_train_te_with_gguf_override_raises(self):
        from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
        config = _flux2_config()
        config.text_encoder.model_name = "C:/models/te.gguf"
        config.text_encoder.train = True
        with pytest.raises(ValueError, match="frozen"):
            BaseFlux2Setup._check_gguf_te_constraints(config)

    def test_frozen_te_with_gguf_override_passes(self):
        from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
        config = _flux2_config()
        config.text_encoder.model_name = "C:/models/te.gguf"
        config.text_encoder.train = False
        BaseFlux2Setup._check_gguf_te_constraints(config)

    def test_no_override_passes_with_train(self):
        from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
        config = _flux2_config()
        config.text_encoder.train = True
        BaseFlux2Setup._check_gguf_te_constraints(config)
