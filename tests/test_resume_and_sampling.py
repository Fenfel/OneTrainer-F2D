"""Lightweight regression tests for resume state restoration and the sampling path.

Resume: _load_internal_data must restore train_progress and the optimizer/EMA state dicts from a
backup directory (tiny synthetic files, no real weights).

Sampling: the Flux2Sampler must encode a fixed prompt once and reuse the cached conditioning on
the second sample (no text encoder materialization), while the VAE decode runs on every sample.
Tiny stub components only - no GPU, no real weights.
"""

import json
import os

from modules.model.Flux2Model import Flux2Model
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelSampler.Flux2Sampler import Flux2Sampler
from modules.util.enum.DataType import DataType
from modules.util.enum.ImageFormat import ImageFormat

import pytest

import torch


class _Loader(InternalModelLoaderMixin):
    def __init__(self):
        super().__init__()


class _ProgressModel:
    optimizer_state_dict = None
    ema_state_dict = None
    train_progress = None


class TestResumeStateRestoration:
    def test_backup_restores_progress_optimizer_and_ema(self, tmp_path):
        (tmp_path / "optimizer").mkdir()
        (tmp_path / "ema").mkdir()
        with open(tmp_path / "meta.json", "w") as f:
            json.dump({"train_progress": {
                "epoch": 3, "epoch_step": 17, "epoch_sample": 5, "global_step": 137,
            }}, f)
        optimizer_state = {"state": {"step": torch.tensor(137.0)}, "param_groups": [{"lr": 0.0004}]}
        torch.save(optimizer_state, tmp_path / "optimizer" / "optimizer.pt")
        ema_state = {"shadow": {"w": torch.ones(4)}}
        torch.save(ema_state, tmp_path / "ema" / "ema.pt")

        model = _ProgressModel()
        _Loader()._load_internal_data(model, str(tmp_path))

        tp = model.train_progress
        assert (tp.epoch, tp.epoch_step, tp.epoch_sample, tp.global_step) == (3, 17, 5, 137)
        assert model.optimizer_state_dict["param_groups"][0]["lr"] == 0.0004
        assert torch.equal(model.optimizer_state_dict["state"]["step"], torch.tensor(137.0))
        assert torch.equal(model.ema_state_dict["shadow"]["w"], torch.ones(4))

    def test_training_noise_is_derived_from_global_step(self):
        # The training step seeds its generator from train_progress.global_step (see
        # BaseFlux2Setup.predict), so restoring global_step on resume reproduces the noise
        # sequence exactly - no separate RNG state file is needed. Pin that contract here.
        def noise_for_step(global_step, world_size=1, rank=0):
            batch_seed = global_step * world_size + rank
            gen = torch.Generator(device="cpu").manual_seed(batch_seed)
            return torch.randn(8, generator=gen)

        assert torch.equal(noise_for_step(137), noise_for_step(137))
        assert not torch.equal(noise_for_step(137), noise_for_step(138))


# --------------------------------------------------------------------------- sampling

class _StubImageProcessor:
    def postprocess(self, image, output_type):
        assert output_type == "pil"
        # one PIL image per batch entry, without importing PIL machinery deeply
        from PIL import Image
        return [Image.new("RGB", (4, 4)) for _ in range(image.shape[0])]


class _StubPipeline:
    def __init__(self, transformer):
        self.image_processor = _StubImageProcessor()
        self.transformer = transformer
        self.text_encoder = None


class _StubTransformer:
    def __init__(self):
        self.config = type("C", (), {"guidance_embeds": False})()
        self.calls = 0

    def __call__(self, hidden_states, **_kwargs):
        self.calls += 1
        return type("R", (), {"sample": torch.zeros_like(hidden_states)})()


class _StubVae:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, latents, return_dict):
        self.decode_calls += 1
        batch, channels, h, w = latents.shape
        return (torch.zeros(batch, 3, h * 8, w * 8),)


class _StubModel:
    """Real Flux2Model geometry helpers, stub components around them."""

    model_type = None

    def __init__(self, transformer):
        self._flux = Flux2Model.__new__(Flux2Model)
        self.transformer = transformer
        self.vae = _StubVae()
        self.text_encoder = None
        self.noise_scheduler = _make_scheduler()
        self.autocast_context = __import__("contextlib").nullcontext()
        self.train_dtype = DataType.FLOAT_32
        self.materialize_calls: list[tuple] = []
        self.encode_text_calls = 0
        self.sample_conditioning_cache: dict = {}
        self._pipeline = _StubPipeline(transformer)

    # geometry: delegate to the real Flux2Model statics
    patchify_latents = staticmethod(Flux2Model.patchify_latents)
    unpack_latents = staticmethod(Flux2Model.unpack_latents)
    pack_latents = staticmethod(Flux2Model.pack_latents)
    prepare_latent_image_ids = staticmethod(Flux2Model.prepare_latent_image_ids)
    prepare_text_ids = staticmethod(Flux2Model.prepare_text_ids)
    unpatchify_latents = staticmethod(Flux2Model.unpatchify_latents)

    def unscale_latents(self, latents):
        return latents

    def create_pipeline(self):
        return self._pipeline

    def materialize_only(self, *parts):
        self.materialize_calls.append(("only",) + parts)
        if "vae" in parts and self.vae is None:
            self.vae = _StubVae()

    def materialize(self, *parts):
        # the sampler materializes the VAE without evicting the transformer (the evict would
        # clone every weight into RAM); the stub records the call shape
        self.materialize_calls.append(("all",) + parts)
        if "vae" in parts and self.vae is None:
            self.vae = _StubVae()

    def encode_text(self, text, train_device, text_encoder_sequence_length=None):
        self.encode_text_calls += 1
        n = 2 if isinstance(text, list) else 1
        return torch.randn(n, 16, 128, generator=torch.Generator().manual_seed(1))


def _make_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)


def _sample_config(prompt):
    from modules.util.config.SampleConfig import SampleConfig
    config = SampleConfig.default_values()
    config.prompt = prompt
    config.negative_prompt = ""
    config.height = 64
    config.width = 64
    config.seed = 42
    config.random_seed = False
    config.diffusion_steps = 2
    config.cfg_scale = 1.0
    return config


class TestSamplerConditioningCache:
    def test_fixed_prompt_encoded_once_vae_decodes_every_time(self, tmp_path):
        transformer = _StubTransformer()
        model = _StubModel(transformer)
        sampler = Flux2Sampler(torch.device("cpu"), torch.device("cpu"), model, model.model_type)

        config = _sample_config("a red fox")
        sampler.sample(config, str(tmp_path / "a.png"), image_format=ImageFormat.PNG)
        first_encode = model.encode_text_calls
        first_te_materializations = sum("text_encoder" in p for p in model.materialize_calls)
        assert first_encode == 1
        assert first_te_materializations == 1
        assert model.vae.decode_calls == 1

        # second sample, same prompt: conditioning comes from the cache, no text encoder work
        sampler.sample(config, str(tmp_path / "b.png"), image_format=ImageFormat.PNG)
        assert model.encode_text_calls == first_encode
        assert sum("text_encoder" in p for p in model.materialize_calls) == first_te_materializations
        assert model.vae.decode_calls == 2
        assert transformer.calls == 4  # 2 steps per sample
        # the VAE must be materialized WITHOUT evicting the transformer: an evict would clone
        # every GGUF weight into RAM (the +16 GiB sampling peak)
        assert ("all", "vae") in model.materialize_calls
        assert ("only", "vae") not in model.materialize_calls

    def test_different_prompt_reencodes(self, tmp_path):
        model = _StubModel(_StubTransformer())
        sampler = Flux2Sampler(torch.device("cpu"), torch.device("cpu"), model, model.model_type)

        sampler.sample(_sample_config("a red fox"), str(tmp_path / "a.png"), image_format=ImageFormat.PNG)
        sampler.sample(_sample_config("a blue whale"), str(tmp_path / "b.png"), image_format=ImageFormat.PNG)
        assert model.encode_text_calls == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
