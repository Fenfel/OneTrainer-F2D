from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util import factory
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


def make_lora_model_loader(
    model_spec_map: dict[ModelType, str],
    model_class: type[BaseModel],
    model_loader_class: type,
    embedding_loader_class: type | None,
    lora_loader_class: type,
):
    class GenericLoRAModelLoader(
        BaseModelLoader,
        ModelSpecModelLoaderMixin,
        InternalModelLoaderMixin,
    ):
        def __init__(self):
            super().__init__()

        def _default_model_spec_name(
                self,
                model_type: ModelType,
        ) -> str | None:
            return model_spec_map.get(model_type)

        def load(
                self,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ) -> model_class | None:
            base_model_loader = model_loader_class()
            lora_model_loader = lora_loader_class()
            if embedding_loader_class is not None:
                embedding_loader = embedding_loader_class()

            model = model_class(model_type=model_type)
            self._load_internal_data(model, model_names.lora)
            model.model_spec = self._load_default_model_spec(model_type)

            if model_names.base_model is not None:
                base_model_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            lora_model_loader.load(model, model_names)
            if embedding_loader_class is not None:
                embedding_loader.load(model, model_names.lora, model_names)

            return model

        # Staged loading API (release_text_encoder_after_cache): build the model and load its
        # components one at a time, in the order the caching stages need. Only model loaders that
        # implement the per-component methods (currently Flux.2) support this.

        def _staged_base_loader(self):
            base_model_loader = model_loader_class()
            if not hasattr(base_model_loader, "load_metadata"):
                raise NotImplementedError(
                    "staged loading is not supported for this model loader: it has no per-component load methods"
                )
            return base_model_loader

        def create_model(
                self,
                model_type: ModelType,
                model_names: ModelNames,
        ) -> model_class:
            # create the model and load internal data (train progress, optimizer state) and the
            # model spec, but no component weights yet
            model = model_class(model_type=model_type)
            self._load_internal_data(model, model_names.lora)
            model.model_spec = self._load_default_model_spec(model_type)
            return model

        def load_metadata(
                self,
                model: BaseModel,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ):
            self._staged_base_loader().load_metadata(model, model_type, model_names, weight_dtypes, quantization)

        def load_text_encoder(
                self,
                model: BaseModel,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ):
            self._staged_base_loader().load_text_encoder(model, model_type, model_names, weight_dtypes, quantization)

        def load_vae(
                self,
                model: BaseModel,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ):
            self._staged_base_loader().load_vae(model, model_type, model_names, weight_dtypes, quantization)

        def load_transformer(
                self,
                model: BaseModel,
                model_type: ModelType,
                model_names: ModelNames,
                weight_dtypes: ModelWeightDtypes,
                quantization: QuantizationConfig,
        ):
            self._staged_base_loader().load_transformer(model, model_type, model_names, weight_dtypes, quantization)

        def load_lora(
                self,
                model: BaseModel,
                model_names: ModelNames,
        ):
            # the LoRA state dict is loaded only after the base components exist: canonicalization
            # enumerates module names on the live transformer
            lora_loader_class().load(model, model_names)

    for model_type in model_spec_map:
        factory.register(BaseModelLoader, GenericLoRAModelLoader, model_type, TrainingMethod.LORA)
    return GenericLoRAModelLoader
