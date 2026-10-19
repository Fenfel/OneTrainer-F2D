"""Tests for the staged loading order of release_text_encoder_after_cache (Flux.2 LoRA).

Event-order/contract tests with lightweight stubs only: no model weights, no dataset, no
training loop. Covers: TE load -> text cache complete -> TE release -> VAE load -> image
cache complete -> VAE release -> transformer load -> LoRA/optimizer setup; the transformer
loader must not run when a cache/release stage fails; guards for missing components; the
slim cache-warming pipelines (text stage without latent cache / image sorting / batch
assembly, image stage without text encoding); and that the training pipeline is created
fresh after the warming stages instead of rewinding a warmed one.
"""

import contextlib
import gc
import os
import weakref
from types import SimpleNamespace

import pytest

import torch

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import (
    LatentCachingDiskCache,
    TextCachingDiskCache,
)
from modules.model.Flux2Model import Flux2Model
from modules.modelSetup.Flux2LoRASetup import Flux2LoRASetup
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.released_vae import ReleasedVae


def _staged_config(tmp_path) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.FLUX_2
    config.training_method = TrainingMethod.LORA
    config.latent_caching = True
    config.text_encoder.train = False
    config.release_text_encoder_after_cache = True
    config.validation = False
    config.workspace_dir = str(tmp_path / "workspace")
    config.cache_dir = str(tmp_path / "cache")
    return config


def _make_cache(cls: type, cache_dir: str, groups: dict[str, int], cached: bool):
    """A real readiness DiskCache with pre-seeded group state and (optionally) on-disk aggregates."""
    cache = cls(
        cache_dir=cache_dir,
        split_names=["x"],
        aggregate_names=[],
        variations_in_name="concept.variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=["concept.path"],
        group_enabled_in_name="concept.enabled",
    )
    cache.variations_initialized = True
    cache.group_variations = groups
    if cached:
        for group, variations in groups.items():
            for v in range(variations):
                d = os.path.join(cache_dir, group, f"variation-{v}")
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, "aggregate.pt"), "w").close()
    return cache


class _FakeDataSet:
    def __init__(self):
        self.epoch_rounds = 0

    def start_next_epoch(self):
        self.epoch_rounds += 1


class _FakeLoader:
    def __init__(self, modules: list):
        self._modules = modules
        self.ds = _FakeDataSet()

    def get_data_set(self):
        return self.ds

    def get_pipeline_modules(self):
        return self._modules


class _StubModel:
    model_type = ModelType.FLUX_2

    def __init__(self):
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.text_encoder_offload_conductor = None
        self.parameters = type("_Params", (), {"parameters": lambda self: []})()
        self.train_progress = object()
        self.train_config = None

    def is_dev(self) -> bool:
        return True

    def save_vae_bn_stats(self):
        self.bn_stats_saved = True

    def eval(self):
        pass


class _RecordingModelLoader:
    def __init__(self, events: list):
        self.events = events

    def create_model(self, model_type, model_names):
        self.events.append("create_model")
        return _StubModel()

    def load_metadata(self, model, model_type, model_names, weight_dtypes, quantization):
        self.events.append("load_metadata")

    def load_text_encoder(self, model, model_type, model_names, weight_dtypes, quantization):
        self.events.append("load_text_encoder")

        class _Encoder:
            pass

        encoder = _Encoder()
        # weakref for the collection assertions; no strong ref survives this frame
        model._encoder_ref = weakref.ref(encoder)
        model.text_encoder = encoder

    def load_vae(self, model, model_type, model_names, weight_dtypes, quantization):
        self.events.append("load_vae")

        class _Vae:
            pass

        vae = _Vae()
        model._vae_ref = weakref.ref(vae)
        model.vae = vae

    def load_transformer(self, model, model_type, model_names, weight_dtypes, quantization):
        self.events.append("load_transformer")
        model.transformer = object()

    def load_lora(self, model, model_names):
        self.events.append("load_lora")


class _RecordingModelSetup:
    def __init__(self, events: list):
        self.events = events

    def setup_optimizations(self, model, config):
        self.events.append("setup_optimizations")

    def setup_train_device(self, model, config):
        self.events.append("setup_train_device")

    def setup_model(self, model, config):
        self.events.append("setup_model")


class _Callbacks:
    def on_update_status(self, status):
        pass


def _make_trainer(config: TrainConfig, events: list, text_cached: bool = True,
                  image_cached: bool = True, hold_encoder: bool = False):
    trainer = object.__new__(GenericTrainer)
    trainer.config = config
    trainer.callbacks = _Callbacks()
    trainer.model_loader = _RecordingModelLoader(events)
    trainer.model_setup = _RecordingModelSetup(events)
    trainer.model_sampler = None
    trainer.model = None
    trainer._holders = []

    text_cache = _make_cache(
        TextCachingDiskCache, os.path.join(config.cache_dir, "text"),
        {"groupA": 2, "groupB": 1}, text_cached,
    )
    image_cache = _make_cache(
        LatentCachingDiskCache, os.path.join(config.cache_dir, "image"),
        {"groupA": 1}, image_cached,
    )
    trainer._image_cache = image_cache

    created_loaders: list[_FakeLoader] = []

    def create_data_loader(model, model_setup, train_progress, is_validation=False, cache_warming_stage=None):
        events.append("create_data_loader:" + str(cache_warming_stage))
        loader = _FakeLoader([text_cache, image_cache])
        created_loaders.append(loader)
        return loader

    trainer.create_data_loader = create_data_loader
    trainer._created_loaders = created_loaders
    trainer.create_model_saver = lambda: events.append("create_model_saver") or object()
    trainer.create_model_sampler = lambda model: events.append("create_model_sampler") or None

    if hold_encoder:
        original_load_te = trainer.model_loader.load_text_encoder

        def load_te(model, *args, **kwargs):
            original_load_te(model, *args, **kwargs)
            trainer._holders.append(model.text_encoder)  # simulates a leaked owner

        trainer.model_loader.load_text_encoder = load_te

    return trainer


class TestStagedEventOrder:
    def test_full_stage_order(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events)

        trainer._GenericTrainer__init_staged(ModelNames())

        # the heavy components must appear in exactly this order, each cache stage between
        # its load and its release, and the transformer only after both releases. The training
        # pipeline (stage None) is created fresh at the very end, after the transformer.
        order = [e for e in events if e.startswith("create_data_loader") or e in (
            "create_model", "load_metadata", "load_text_encoder",
            "load_vae", "load_transformer", "load_lora", "setup_model", "create_model_sampler",
        )]
        assert order == [
            "create_model", "load_metadata", "load_text_encoder", "create_data_loader:text",
            "load_vae", "create_data_loader:image", "load_transformer", "load_lora", "setup_model",
            "create_data_loader:None", "create_model_sampler",
        ]

        assert trainer.model.text_encoder is None
        assert trainer.model.text_encoder_released is True
        assert trainer.model.vae is None
        assert trainer.model.vae_released is True
        assert trainer.model.bn_stats_saved is True, "BN stats must be saved before the VAE is released"
        assert trainer.model.transformer is not None
        assert trainer.parameters == []

    def test_components_collected_after_release(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events)

        trainer._GenericTrainer__init_staged(ModelNames())

        assert trainer.model._encoder_ref() is None, "TE must be collectable after release"
        assert trainer.model._vae_ref() is None, "VAE must be collectable after release"

    def test_transformer_not_loaded_when_text_cache_incomplete(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, text_cached=False)

        with pytest.raises(RuntimeError, match="text cache"):
            trainer._GenericTrainer__init_staged(ModelNames())

        assert "load_vae" not in events
        assert "load_transformer" not in events

    def test_transformer_not_loaded_when_text_encoder_leaks(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, hold_encoder=True)

        with pytest.raises(RuntimeError, match="text encoder was still referenced"):
            trainer._GenericTrainer__init_staged(ModelNames())

        assert "load_vae" not in events
        assert "load_transformer" not in events

    def test_transformer_not_loaded_when_image_cache_incomplete(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, image_cached=False)

        with pytest.raises(RuntimeError, match=r"image \(latent\) cache"):
            trainer._GenericTrainer__init_staged(ModelNames())

        assert "load_transformer" not in events
        # the VAE stage itself completed before the image cache failed
        assert "load_vae" in events
        assert getattr(trainer.model, "vae_released", False) is False

    def test_warming_loaders_dropped_and_training_pipeline_created_fresh(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events)

        trainer._GenericTrainer__init_staged(ModelNames())

        # three loaders were created: text warming, image warming, and the fresh training
        # pipeline. The training loader is a new object (not a rewound warming pipeline) and
        # has not run any caching epochs itself, so its first training epoch starts clean.
        text_warming, image_warming, training = trainer._created_loaders
        assert trainer.data_loader is training
        assert training is not text_warming and training is not image_warming
        assert training.ds.epoch_rounds == 0


class TestSlimWarmingPipelines:
    """The staged cache-warming pipelines are real mgds pipelines built by Flux2BaseDataLoader.

    Text stage: no VAE/image modules, no latent cache, no batch sorting/assembly; image stage:
    no text-encoder modules. Module indices up to the augmentation/crop modules (text stage)
    and up to the image cache (image stage) are identical to the training pipeline, so the
    augmentation seeds, cache keys, balancing and variation selection are the standard ones.
    """

    class _StubModel:
        model_type = ModelType.FLUX_2

        def __init__(self):
            from modules.util.enum.DataType import DataType
            self.train_dtype = DataType.FLOAT_16
            self.autocast_context = contextlib.nullcontext()
            # Tokenize.__init__ attaches a lock to the tokenizer object, so it must be a real
            # instance with a __dict__
            self.tokenizer = type("StubTokenizer", (), {})()
            self.text_encoder = None
            self.vae = None

        def is_dev(self) -> bool:
            return True

    def _config(self, tmp_path):
        from modules.util.config.ConceptConfig import ConceptConfig
        config = TrainConfig.default_values()
        config.model_type = ModelType.FLUX_2
        config.latent_caching = True
        config.cache_dir = str(tmp_path / "cache")
        config.concepts = [ConceptConfig.default_values()]
        return config

    def _modules(self, tmp_path, stage):
        from modules.dataLoader.Flux2BaseDataLoader import Flux2BaseDataLoader
        from modules.util.TrainProgress import TrainProgress
        loader = Flux2BaseDataLoader(
            torch.device("cpu"), torch.device("cpu"), self._config(tmp_path),
            self._StubModel(), None, TrainProgress(), False, stage,
        )
        return [type(m).__name__ for m in loader.get_pipeline_modules()]

    def test_text_stage_has_no_latent_cache_image_sorting_or_vae(self, tmp_path):
        names = self._modules(tmp_path, "text")
        assert "TextCachingDiskCache" in names
        assert "VariationSorting" in names
        assert "OutputPipelineModule" in names
        for forbidden in ("LatentCachingDiskCache", "EncodeVAE", "DecodeVAE", "RescaleImageChannels",
                          "AspectBatchSorting", "InlineAspectBatchSorting", "RandomLatentMaskRemove"):
            assert forbidden not in names, f"text warming pipeline must not contain {forbidden}"

    def test_image_stage_has_no_text_encoding_or_text_cache(self, tmp_path):
        names = self._modules(tmp_path, "image")
        assert "LatentCachingDiskCache" in names
        assert "EncodeVAE" in names
        for forbidden in ("TextCachingDiskCache", "EncodeMistralText", "EncodeQwenText", "Tokenize",
                          "AspectBatchSorting", "InlineAspectBatchSorting"):
            assert forbidden not in names, f"image warming pipeline must not contain {forbidden}"

    def test_module_indices_match_training_pipeline(self, tmp_path):
        full = self._modules(tmp_path, None)
        text = self._modules(tmp_path, "text")
        image = self._modules(tmp_path, "image")

        def common_prefix_len(a, b):
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            return n

        # text stage: identical indices for every module up to the augmentations; the slim
        # pipeline continues straight with the text encoding where the full one has the
        # image-encoding part
        k = common_prefix_len(full, text)
        assert full[k] == "RescaleImageChannels" and text[k] == "Tokenize"
        assert "RandomFlip" in full[:k] and "RandomFlip" in text[:k]

        # image stage: identical indices up to and including the image encoding; the slim
        # pipeline goes straight to the latent cache where the full one has the text encoding
        k = common_prefix_len(full, image)
        assert full[k] == "Tokenize" and image[k] == "LatentCachingDiskCache"


class TestReleasedVaePlaceholder:
    def test_encode_raises_clear_error(self):
        with pytest.raises(RuntimeError, match="release_text_encoder_after_cache"):
            ReleasedVae().encode(None)

    def test_decode_raises_clear_error(self):
        with pytest.raises(RuntimeError, match="release_text_encoder_after_cache"):
            ReleasedVae().decode(None)


class TestVaeBnStatsSurviveRelease:
    """scale_latents()/unscale_latents() must give bit-identical results after the VAE is
    released, using only the saved batch-norm stats (no real weights, no training)."""

    def _model_with_fake_vae(self):
        class _FakeVae:  # weakref-able, unlike SimpleNamespace
            pass

        model = Flux2Model(ModelType.FLUX_2)
        gen = torch.Generator().manual_seed(123)
        vae = _FakeVae()
        # the real AutoencoderKLFlux2 bn stats cover the patchified latents: 128 * 4 = 512 channels
        vae.bn = SimpleNamespace(
            running_mean=torch.randn(512, generator=gen),
            running_var=torch.rand(512, generator=gen) + 0.5,
        )
        vae.config = SimpleNamespace(batch_norm_eps=0.0001)
        model.vae = vae
        return model, vae

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_scale_unscale_bitwise_identical_after_release(self, dtype):
        model, _vae = self._model_with_fake_vae()
        gen = torch.Generator().manual_seed(7)
        # patchified latents: 128 latent channels * 4 = 512
        latents = torch.randn(2, 512, 4, 6, generator=gen).to(dtype)

        before_scale = model.scale_latents(latents)
        before_unscale = model.unscale_latents(latents)

        # what __release_vae does: save the stats while the VAE is alive, then drop the VAE
        model.save_vae_bn_stats()
        model.vae = None

        assert torch.equal(model.scale_latents(latents), before_scale)
        assert torch.equal(model.unscale_latents(latents), before_unscale)

    def test_saved_stats_are_independent_cpu_clones(self):
        model, vae = self._model_with_fake_vae()
        original_mean = vae.bn.running_mean.clone()
        original_var = vae.bn.running_var.clone()

        model.save_vae_bn_stats()

        for saved, original in ((model.vae_bn_running_mean, original_mean),
                                (model.vae_bn_running_var, original_var)):
            assert saved.device.type == "cpu"
            assert not saved.requires_grad
            assert saved.data_ptr() != original.data_ptr()
            assert saved._base is None  # a clone, not a view into the VAE tensor
            assert torch.equal(saved, original)
        assert model.vae_batch_norm_eps == vae.config.batch_norm_eps

        # mutating the VAE tensors afterwards must not affect the saved copies
        vae.bn.running_mean += 1.0
        vae.bn.running_var *= 2.0
        assert torch.equal(model.vae_bn_running_mean, original_mean)
        assert torch.equal(model.vae_bn_running_var, original_var)

    def test_saved_stats_do_not_keep_the_vae_alive(self):
        model, vae = self._model_with_fake_vae()
        vae_ref = weakref.ref(vae)

        model.save_vae_bn_stats()
        model.vae = None
        del vae
        gc.collect()

        assert vae_ref() is None, "the saved stats must not reference the VAE"
        latents = torch.randn(1, 512, 2, 2)
        assert torch.isfinite(model.scale_latents(latents)).all()
        assert torch.isfinite(model.unscale_latents(latents)).all()

    def test_missing_stats_raise_clear_error(self):
        model = Flux2Model(ModelType.FLUX_2)
        with pytest.raises(RuntimeError, match="release_text_encoder_after_cache"):
            model.scale_latents(torch.randn(1, 512, 2, 2))
        with pytest.raises(RuntimeError, match="release_text_encoder_after_cache"):
            model.unscale_latents(torch.randn(1, 512, 2, 2))


class TestMissingComponentGuards:
    """Staged stages run while transformer/vae/text_encoder are still None."""

    def _stub_model(self):
        class _M:
            text_encoder = None
            vae = None
            transformer = None
            transformer_lora = None
            train_progress = None

            def materialize_only(self, *parts):
                pass

        return _M()

    def _config(self):
        config = TrainConfig.default_values()
        config.latent_caching = True
        return config

    def test_setup_train_device_with_missing_components(self):
        setup = Flux2LoRASetup(torch.device("cpu"), torch.device("cpu"), False)
        setup.setup_train_device(self._stub_model(), self._config())

    def test_after_optimizer_step_with_missing_components(self):
        setup = Flux2LoRASetup(torch.device("cpu"), torch.device("cpu"), False)
        setup.after_optimizer_step(self._stub_model(), self._config(), object())


class TestIsDevFromConfig:
    def test_is_dev_before_transformer_loaded(self):
        model = Flux2Model(ModelType.FLUX_2)
        model.transformer_config = {"num_attention_heads": 48}
        assert model.is_dev() is True

        model.transformer_config = {"num_attention_heads": 24}
        assert model.is_dev() is False

    def test_is_dev_without_anything_raises(self):
        model = Flux2Model(ModelType.FLUX_2)
        with pytest.raises(RuntimeError, match="neither the transformer nor its config"):
            model.is_dev()
