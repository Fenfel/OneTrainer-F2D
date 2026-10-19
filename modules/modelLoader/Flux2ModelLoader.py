import os
import traceback

from modules.model.Flux2Model import Flux2Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2Transformer2DModel,
    GGUFQuantizationConfig,
)
from transformers import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    PixtralProcessor,
    Qwen2Tokenizer,
    Qwen3ForCausalLM,
)


class Flux2ModelLoader(
    HFModelLoaderMixin,
    GGUFTextEncoderLoaderMixin,
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __validate_text_encoder_override(text_encoder_model_name: str | None):
        if not text_encoder_model_name:
            return
        # validate the TE override up front: an invalid override must fail directly,
        # never silently fall back to the full stock TE
        if not text_encoder_model_name.lower().endswith(".gguf"):
            raise ValueError(
                "The text encoder override must be a GGUF file "
                f"(only quantized GGUF text encoders are supported): {text_encoder_model_name}"
            )
        if not os.path.isfile(text_encoder_model_name):
            raise FileNotFoundError(f"GGUF text encoder file not found: {text_encoder_model_name}")

    def __load_metadata(self, model: Flux2Model, base_model_name: str):
        # Reads only configs/tokenizer/scheduler - no weight tensors. The transformer config decides
        # dev vs. klein (is_dev) without loading the transformer, so the staged loading mode can
        # build the dataloader and pick the text encoder before any heavy component is loaded.
        transformer_config = Flux2Transformer2DModel.load_config(base_model_name, subfolder="transformer")
        model.transformer_config = transformer_config

        if transformer_config["num_attention_heads"] == 48:  # Flux2.Dev
            tokenizer = PixtralProcessor.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            ).tokenizer
        else:  # Flux2.Klein
            tokenizer = Qwen2Tokenizer.from_pretrained(
                base_model_name,
                subfolder="tokenizer",
            )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler

    def __load_transformer(
            self,
            model: Flux2Model,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            quantization: QuantizationConfig,
    ):
        if transformer_model_name:
            transformer = Flux2Transformer2DModel.from_single_file(
                transformer_model_name,
                config=base_model_name,
                subfolder="transformer",
                #avoid loading the transformer in float32:
                torch_dtype=torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                Flux2Transformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.transformer = transformer

    def __load_text_encoder(
            self,
            model: Flux2Model,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            text_encoder_model_name: str,
            quantization: QuantizationConfig,
    ):
        self.__validate_text_encoder_override(text_encoder_model_name)

        if model.is_dev():
            if text_encoder_model_name:
                # a GGUF file overrides the stock text encoder; the full BF16 TE is never loaded
                text_config = Mistral3Config.from_pretrained(
                    base_model_name,
                    subfolder="text_encoder",
                ).text_config
                compute_dtype = weight_dtypes.text_encoder.torch_dtype() or torch.bfloat16
                text_encoder = self._load_gguf_text_encoder(
                    text_encoder_model_name,
                    text_config,
                    compute_dtype,
                )
            else:
                text_encoder = self._load_transformers_sub_module(
                    Mistral3ForConditionalGeneration,
                    weight_dtypes.text_encoder,
                    weight_dtypes.fallback_train_dtype,
                    base_model_name,
                    "text_encoder",
                )
        else:  # Flux2.Klein
            if text_encoder_model_name:
                raise ValueError(
                    "Overriding the text encoder with a GGUF file is only supported for Flux.2-dev (Mistral), "
                    "not for Flux.2.Klein (Qwen3)."
                )
            text_encoder = self._load_transformers_sub_module(
                Qwen3ForCausalLM,
                weight_dtypes.text_encoder,
                weight_dtypes.fallback_train_dtype,
                base_model_name,
                "text_encoder",
            )

        model.text_encoder = text_encoder

    def __load_vae(
            self,
            model: Flux2Model,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
    ):
        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLFlux2,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        model.vae = vae

    def __load_internal(
            self,
            model: Flux2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            text_encoder_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name,
                text_encoder_model_name, vae_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: Flux2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            text_encoder_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        self.__load_metadata(model, base_model_name)
        self.__load_transformer(model, weight_dtypes, base_model_name, transformer_model_name, quantization)
        self.__load_text_encoder(model, weight_dtypes, base_model_name, text_encoder_model_name, quantization)
        self.__load_vae(model, weight_dtypes, base_model_name, vae_model_name)
        model.model_type = model_type

    def __load_safetensors(
            self,
            model: Flux2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            text_encoder_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        #no single file .safetensors for Qwen available at the time of writing this code
        raise NotImplementedError("Loading of single file Flux2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    # Staged loading API: each component can be loaded on its own, in the order the
    # release_text_encoder_after_cache mode needs (metadata -> text encoder -> VAE -> transformer).
    # load_metadata must run first: it sets transformer_config, which is what is_dev() reads before
    # the transformer itself exists.

    def load_metadata(
            self,
            model: Flux2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        self.__load_metadata(model, model_names.base_model)
        model.model_type = model_type

    def load_text_encoder(
            self,
            model: Flux2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        self.__load_text_encoder(
            model, weight_dtypes, model_names.base_model, model_names.text_encoder_model, quantization,
        )

    def load_vae(
            self,
            model: Flux2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        self.__load_vae(model, weight_dtypes, model_names.base_model, model_names.vae_model)

    def load_transformer(
            self,
            model: Flux2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        self.__load_transformer(
            model, weight_dtypes, model_names.base_model, model_names.transformer_model, quantization,
        )

    def load(
            self,
            model: Flux2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        self.__validate_text_encoder_override(model_names.text_encoder_model)

        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.text_encoder_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.text_encoder_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.text_encoder_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class Flux2LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _legacy_conversion(self, model: Flux2Model) -> list | None:
        return self._mixture_legacy_conversion(model)

    def load(
            self,
            model: Flux2Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


Flux2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.FLUX_2: "resources/sd_model_spec/flux_2.0-lora.json",
    },
    model_class=Flux2Model,
    model_loader_class=Flux2ModelLoader,
    lora_loader_class=Flux2LoRALoader,
    embedding_loader_class=None,
)

Flux2FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.FLUX_2: "resources/sd_model_spec/flux_2.0.json",
    },
    model_class=Flux2Model,
    model_loader_class=Flux2ModelLoader,
    embedding_loader_class=None,
)
