"""Tests for release_text_encoder_after_cache (freezing the TE after text caching, Flux.2 LoRA).

Lightweight, isolated tests only: no model weights, no dataset, no training loop.
Covers: config round-trip, pre-load validation rules, cache-readiness query,
released-encoder placeholder, and the encode_text guard.
"""

import json
import os
import weakref

import pytest

import torch

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import TextCachingDiskCache
from modules.model.Flux2Model import Flux2Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.Flux2LoRASetup import Flux2LoRASetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.released_text_encoder import ReleasedTextEncoder


def _flux2_lora_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.FLUX_2
    config.training_method = TrainingMethod.LORA
    config.latent_caching = True
    config.text_encoder.train = False
    config.release_text_encoder_after_cache = True
    config.samples = []
    return config


def _flux2_lora_setup() -> Flux2LoRASetup:
    return Flux2LoRASetup(torch.device("cpu"), torch.device("cpu"), False)


class TestConfigPlumbing:
    def test_default_is_off(self):
        config = TrainConfig.default_values()
        assert config.release_text_encoder_after_cache is False

    def test_config_roundtrip_preserves_flag(self):
        config = _flux2_lora_config()
        data = json.loads(json.dumps(config.to_dict()))
        assert data["release_text_encoder_after_cache"] is True

        config2 = TrainConfig.default_values()
        config2.from_dict(data)
        assert config2.release_text_encoder_after_cache is True

    def test_old_config_without_flag_defaults_to_off(self):
        config = _flux2_lora_config()
        data = json.loads(json.dumps(config.to_dict()))
        del data["release_text_encoder_after_cache"]

        config2 = TrainConfig.default_values()
        config2.from_dict(data)
        assert config2.release_text_encoder_after_cache is False


class TestPreLoadValidation:
    def test_valid_combination_passes(self):
        _flux2_lora_setup().validate_pre_load(_flux2_lora_config())

    def test_flag_off_never_raises(self):
        config = _flux2_lora_config()
        config.release_text_encoder_after_cache = False
        config.training_method = TrainingMethod.FINE_TUNE
        config.latent_caching = False
        config.text_encoder.train = True
        _flux2_lora_setup().validate_pre_load(config)

    def test_rejects_non_lora(self):
        config = _flux2_lora_config()
        config.training_method = TrainingMethod.FINE_TUNE
        with pytest.raises(ValueError, match="LoRA"):
            _flux2_lora_setup().validate_pre_load(config)

    def test_rejects_without_latent_caching(self):
        config = _flux2_lora_config()
        config.latent_caching = False
        with pytest.raises(ValueError, match="latent caching"):
            _flux2_lora_setup().validate_pre_load(config)

    def test_rejects_trainable_text_encoder(self):
        config = _flux2_lora_config()
        config.text_encoder.train = True
        with pytest.raises(ValueError, match="frozen text encoder"):
            _flux2_lora_setup().validate_pre_load(config)

    def test_enabled_samples_are_allowed(self):
        # samples work with staged loading: the text encoder / VAE are reloaded on demand for
        # the sampling phases and released again afterwards
        config = _flux2_lora_config()
        # samples list with one enabled entry: a minimal object with .enabled is enough
        class _S:
            enabled = True
        config.samples = [_S()]
        _flux2_lora_setup().validate_pre_load(config)

    def test_non_flux2_setup_rejects_flag(self):
        # BaseModelSetup base implementation rejects the flag for all non-Flux.2 setups
        class _DummySetup(BaseModelSetup):
            def create_parameters(self, model, config):
                pass

            def setup_optimizations(self, model, config):
                pass

            def setup_model(self, model, config):
                pass

            def setup_train_device(self, model, config):
                pass

            def predict(self, model, batch, config, train_progress, *, deterministic=False):
                pass

            def calculate_loss(self, model, batch, data, config):
                pass

            def after_optimizer_step(self, model, config, train_progress):
                pass

        setup = _DummySetup(torch.device("cpu"), torch.device("cpu"), False)
        config = _flux2_lora_config()
        with pytest.raises(ValueError, match="Flux.2"):
            setup.validate_pre_load(config)


class TestCacheReadiness:
    def _make_cache(self, tmp_path) -> TextCachingDiskCache:
        cache = TextCachingDiskCache(
            cache_dir=str(tmp_path),
            split_names=["tokens"],
            aggregate_names=[],
            variations_in_name="concept.text_variations",
            balancing_in_name="concept.balancing",
            balancing_strategy_in_name="concept.balancing_strategy",
            variations_group_in_name=["concept.path"],
            group_enabled_in_name="concept.enabled",
        )
        return cache

    def test_not_ready_before_variations_initialized(self, tmp_path):
        assert self._make_cache(tmp_path).all_variations_cached() is False

    def test_not_ready_when_aggregate_missing(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.variations_initialized = True
        cache.group_variations = {"groupA": 2, "groupB": 1}
        os.makedirs(os.path.join(cache.cache_dir, "groupA", "variation-0"), exist_ok=True)
        open(os.path.join(cache.cache_dir, "groupA", "variation-0", "aggregate.pt"), "w").close()
        assert cache.all_variations_cached() is False

    def test_ready_when_all_aggregates_present(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.variations_initialized = True
        cache.group_variations = {"groupA": 2, "groupB": 1}
        for group, variations in cache.group_variations.items():
            for v in range(variations):
                d = os.path.join(cache.cache_dir, group, f"variation-{v}")
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, "aggregate.pt"), "w").close()
        assert cache.all_variations_cached() is True


class TestReleasedPlaceholder:
    def test_call_raises_clear_error(self):
        with pytest.raises(RuntimeError, match="release_text_encoder_after_cache"):
            ReleasedTextEncoder()(None)


class TestReleaseCollectsEncoder:
    """Regression: __release_text_encoder must not keep a local strong reference to the TE.

    A local `text_encoder` variable would outlive torch_gc() inside the frame, so the in-function
    weakref check and the memory snapshot would report the encoder as still alive.
    """

    def _make_trainer(self, encoder):
        from modules.trainer.GenericTrainer import GenericTrainer

        class _Model:
            pass

        class _Loader:
            def get_pipeline_modules(self):
                return []

        class _Conductor:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        trainer = object.__new__(GenericTrainer)
        model = _Model()
        model.text_encoder = encoder
        model.text_encoder_offload_conductor = _Conductor()
        trainer.model = model
        trainer.data_loader = _Loader()
        trainer.model_sampler = None
        return trainer, model

    def test_encoder_collected_and_no_still_referenced_warning(self, capsys):
        # The encoder is created inside this helper frame and only the model (and a weakref)
        # survive it: if the release method adds its own strong reference, the weakref stays
        # alive through the torch_gc() below and the test fails.
        class _Encoder:
            pass

        def create():
            encoder = _Encoder()
            ref = weakref.ref(encoder)
            trainer, model = self._make_trainer(encoder)
            return trainer, model, ref

        trainer, model, encoder_ref = create()

        conductor = model.text_encoder_offload_conductor
        trainer._GenericTrainer__release_text_encoder()

        assert model.text_encoder is None
        assert model.text_encoder_released is True
        # the conductor's pinned cache buffers must be unregistered before it is dropped,
        # otherwise the RAM stays pinned and the next cudaHostRegister fails with error 712
        assert conductor.released is True
        assert model.text_encoder_offload_conductor is None
        assert encoder_ref() is None, "TE must be collectable while the release frame is still running"
        out = capsys.readouterr().out
        assert "Text encoder was garbage collected." in out
        assert "still referenced" not in out

    def test_no_local_strong_ref_in_source(self):
        import inspect

        from modules.trainer.GenericTrainer import GenericTrainer

        src = inspect.getsource(GenericTrainer._GenericTrainer__release_text_encoder)
        assert "text_encoder = model.text_encoder" not in src


class TestEncodeTextGuard:
    def test_released_encoder_without_cached_output_raises(self):
        model = Flux2Model(ModelType.FLUX_2)
        model.text_encoder = None
        model.text_encoder_released = True
        with pytest.raises(RuntimeError, match="released after text caching"):
            model.encode_text(train_device=torch.device("cpu"), text="hello")

    def test_cached_output_passes_through_without_encoder(self):
        model = Flux2Model(ModelType.FLUX_2)
        model.text_encoder = None
        model.text_encoder_released = True
        cached = torch.zeros(1, 4, 8)
        out = model.encode_text(train_device=torch.device("cpu"), text_encoder_output=cached)
        assert out is cached
