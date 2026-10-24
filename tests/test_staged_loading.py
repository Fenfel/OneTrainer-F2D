"""Tests for the staged loading order of release_text_encoder_after_cache (Flux.2 LoRA).

Event-order/contract tests with lightweight stubs only: no model weights, no dataset, no
training loop. Covers: with a fully valid cache the text encoder / VAE are never loaded at all
(BN stats read from disk instead); with an incomplete cache the component is loaded on demand
through its reload hook when the pipeline reaches the cache stage, used, and released again;
the transformer loads only after both releases; guards for a cache that never completes or a
component that leaks; the slim cache-warming pipelines (text stage without latent cache /
image sorting / batch assembly, image stage without text encoding); and that the training
pipeline is created fresh after the warming stages instead of rewinding a warmed one.
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
from modules.util.config.SampleConfig import SampleConfig
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
    # explicit empty list: any_samples_enabled() must not fall back to the samples.json file
    # in the current working directory
    config.samples = []
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
        self.on_epoch = None

    def start_next_epoch(self):
        if self.on_epoch is not None:
            self.on_epoch()
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
        self.component_reload_hooks = {}
        self.transformer_config = {"num_attention_heads": 48, "guidance_embeds": False}
        self.sample_conditioning_cache: dict = {}
        self.autocast_context = contextlib.nullcontext()
        self.encode_text_calls = 0

    def is_dev(self) -> bool:
        return True

    def materialize_only(self, *parts):
        # mimic BaseModel._move_part: a released/never-loaded part triggers its reload hook
        for part in parts:
            if getattr(self, part, None) is None and part in self.component_reload_hooks:
                self.component_reload_hooks[part]()

    def encode_text(self, text, train_device, text_encoder_sequence_length=None):
        self.encode_text_calls += 1
        n = 2 if isinstance(text, list) else 1
        return torch.randn(n, 4, 8, generator=torch.Generator().manual_seed(1))

    def save_vae_bn_stats(self):
        self.bn_stats_saved = True

    def load_vae_bn_stats_from_disk(self, base_model_name, vae_model_name=None):
        self.bn_stats_from_disk = True

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
        self.optimized_parts = set()

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
                  image_cached: bool = True, hold_encoder: bool = False, fill_cache: bool = True):
    trainer = object.__new__(GenericTrainer)
    trainer.config = config
    trainer.callbacks = _Callbacks()
    trainer.train_device = torch.device("cpu")
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

    def run_stage(loader: _FakeLoader, part: str, cache):
        # simulates the pipeline's before_cache_fun -> materialize_only(part): the reload hook
        # loads the component on demand, then the pipeline writes the missing cache entries
        # (unless fill_cache is False, which simulates a cache that never completes)
        model = trainer.model
        if model is not None and getattr(model, part) is None:
            model.component_reload_hooks[part]()
        if fill_cache:
            for group, variations in cache.group_variations.items():
                for v in range(variations):
                    d = os.path.join(cache.cache_dir, group, f"variation-{v}")
                    os.makedirs(d, exist_ok=True)
                    open(os.path.join(d, "aggregate.pt"), "w").close()

    def create_data_loader(model, model_setup, train_progress, is_validation=False, cache_warming_stage=None):
        events.append("create_data_loader:" + str(cache_warming_stage))
        loader = _FakeLoader([text_cache, image_cache])
        if cache_warming_stage == "text" and not text_cached:
            loader.ds.on_epoch = lambda: run_stage(loader, "text_encoder", text_cache)
        elif cache_warming_stage == "image" and not image_cached:
            loader.ds.on_epoch = lambda: run_stage(loader, "vae", image_cache)
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

        # With both caches fully valid, the text encoder and VAE are never loaded at all -
        # only metadata, the two cache probes, the transformer and the LoRA/setup run. The
        # training pipeline (stage None) is created fresh at the very end, after the transformer.
        order = [e for e in events if e.startswith("create_data_loader") or e in (
            "create_model", "load_metadata", "load_text_encoder",
            "load_vae", "load_transformer", "load_lora", "setup_model", "create_model_sampler",
        )]
        assert order == [
            "create_model", "load_metadata", "create_data_loader:text",
            "create_data_loader:image", "load_transformer", "load_lora", "setup_model",
            "create_data_loader:None", "create_model_sampler",
        ]

        assert trainer.model.text_encoder is None
        assert trainer.model.text_encoder_released is True
        assert trainer.model.vae is None
        assert trainer.model.vae_released is True
        assert not hasattr(trainer.model, "bn_stats_saved"), "no VAE was loaded -> nothing to snapshot"
        assert trainer.model.bn_stats_from_disk is True, "BN stats must be read from disk without the VAE"
        assert trainer.model.transformer is not None
        assert trainer.parameters == []

    def test_incomplete_cache_reloads_on_demand_and_releases(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, text_cached=False, image_cached=False)

        trainer._GenericTrainer__init_staged(ModelNames())

        # the components are loaded only when the cache probe finds missing entries, through
        # the reload hook, and released again before the transformer loads
        order = [e for e in events if e in (
            "load_text_encoder", "load_vae", "load_transformer",
        )]
        assert order == ["load_text_encoder", "load_vae", "load_transformer"]

        assert trainer.model.text_encoder is None
        assert trainer.model.text_encoder_released is True
        assert trainer.model.vae is None
        assert trainer.model.vae_released is True
        assert trainer.model.bn_stats_saved is True, "BN stats must be saved before the VAE is released"
        assert not hasattr(trainer.model, "bn_stats_from_disk"), "the VAE was loaded -> stats came from it"

    def test_components_collected_after_release(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, text_cached=False, image_cached=False)

        trainer._GenericTrainer__init_staged(ModelNames())

        assert trainer.model._encoder_ref() is None, "TE must be collectable after release"
        assert trainer.model._vae_ref() is None, "VAE must be collectable after release"

    def test_transformer_not_loaded_when_text_cache_never_completes(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, text_cached=False, fill_cache=False)

        with pytest.raises(RuntimeError, match="text cache"):
            trainer._GenericTrainer__init_staged(ModelNames())

        # the TE was loaded on demand for the incomplete cache, but the stage never finished
        assert "load_text_encoder" in events
        assert "load_vae" not in events
        assert "load_transformer" not in events

    def test_transformer_not_loaded_when_text_encoder_leaks(self, tmp_path):
        events: list[str] = []
        # the encoder can only leak if it was loaded at all -> incomplete text cache
        trainer = _make_trainer(_staged_config(tmp_path), events, text_cached=False, hold_encoder=True)

        with pytest.raises(RuntimeError, match="text encoder was still referenced"):
            trainer._GenericTrainer__init_staged(ModelNames())

        assert "load_vae" not in events
        assert "load_transformer" not in events

    def test_transformer_not_loaded_when_image_cache_never_completes(self, tmp_path):
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events, image_cached=False, fill_cache=False)

        with pytest.raises(RuntimeError, match=r"image \(latent\) cache"):
            trainer._GenericTrainer__init_staged(ModelNames())

        assert "load_transformer" not in events
        # the VAE stage itself ran (loaded on demand) before the image cache failed
        assert "load_vae" in events
        assert getattr(trainer.model, "vae_released", False) is False

    def test_sample_conditioning_key_matches_sampler_without_guidance_key(self, tmp_path):
        # The shipped FLUX.2-dev transformer config.json has NO guidance_embeds key; the model
        # class defaults it to True, so the sampler runs batch_size 1 (no negative prompt).
        # The warmup must read the same default - defaulting to False here keyed the entry with
        # batch_size 2 and the sampler missed it, reloading the text encoder for every sample.
        events: list[str] = []
        config = _staged_config(tmp_path)
        sample = SampleConfig.default_values(ModelType.FLUX_2)
        sample.prompt = "a red fox"
        sample.negative_prompt = "blurry"
        sample.cfg_scale = 3.0
        config.samples = [sample]

        trainer = _make_trainer(config, events)
        trainer._GenericTrainer__init_staged(ModelNames())
        # the stub declares guidance_embeds False explicitly; emulate the real config: no key
        trainer.model.transformer_config = {"num_attention_heads": 48}

        key = trainer._GenericTrainer__sample_conditioning_key(sample)
        assert key == (sample.prompt, None, sample.text_encoder_1_sequence_length, 1)

    def test_sample_conditioning_warmed_with_cache(self, tmp_path):
        # Fixed sample prompts are encoded during the text-cache stage, together with the rest
        # of the caching, and persisted under cache_dir. With a fully valid text cache this is
        # the ONLY text encoder load in the whole run; after the release the sampler finds every
        # scheduled prompt in the conditioning cache and never reloads the TE.
        events: list[str] = []
        config = _staged_config(tmp_path)
        sample = SampleConfig.default_values(ModelType.FLUX_2)
        sample.prompt = "a red fox"
        sample.negative_prompt = "blurry"
        sample.cfg_scale = 3.0  # batch_size 2 -> the negative prompt is part of the key
        config.samples = [sample]

        trainer = _make_trainer(config, events)
        trainer._GenericTrainer__init_staged(ModelNames())

        # the TE was loaded exactly once - for the conditioning warmup - and released again
        assert events.count("load_text_encoder") == 1
        assert trainer.model.encode_text_calls == 1
        assert trainer.model.text_encoder is None
        assert trainer.model.text_encoder_released is True

        # the embedding lives on the model (CPU) and on disk, keyed exactly like the sampler does
        key = (
            "a red fox", "blurry", config.text_encoder_sequence_length, 2,
        )
        assert key in trainer.model.sample_conditioning_cache
        assert trainer.model.sample_conditioning_cache[key].device.type == "cpu"
        cache_file = os.path.join(config.cache_dir, "sample_conditioning.pt")
        assert os.path.isfile(cache_file)

        # a second staged init with the same config reuses the persisted conditioning: the text
        # encoder is never loaded at all
        events2: list[str] = []
        trainer2 = _make_trainer(config, events2)
        trainer2._GenericTrainer__init_staged(ModelNames())
        assert "load_text_encoder" not in events2
        assert trainer2.model.encode_text_calls == 0
        assert key in trainer2.model.sample_conditioning_cache

    def test_lora_backup_skips_full_model_evict(self, tmp_path):
        # A LoRA backup/save serializes only LoRA weights + optimizer/EMA state, never the base
        # weights, so evicting the transformer around it is pure overhead: the offload conductor
        # clones every weight (transiently +1x the model bytes) and setup_train_device moves it
        # all back. Fine-tune saves DO write the base model and must still evict.
        events: list[str] = []
        trainer = _make_trainer(_staged_config(tmp_path), events)
        trainer.model = _StubModel()
        trainer.model.evict = lambda: events.append("evict")

        assert trainer._GenericTrainer__evict_model_for_backup_or_save() is False
        assert "evict" not in events

        trainer.config.training_method = TrainingMethod.FINE_TUNE
        assert trainer._GenericTrainer__evict_model_for_backup_or_save() is True
        assert "evict" in events

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
