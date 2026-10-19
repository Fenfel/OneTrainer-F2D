import copy
import json
import math
import os
import shutil
import traceback
import weakref
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import LatentCachingDiskCache, TextCachingDiskCache
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, huggingface_util, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.compile_util import init_compile, reset_compile
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.profiling_util import PeakMemoryRecorder, TorchMemoryRecorder, TorchProfiler
from modules.util.released_text_encoder import ReleasedTextEncoder
from modules.util.released_vae import ReleasedVae
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeMistralText import EncodeMistralText
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove

from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        # torch._dynamo.config overrides are thread-local, so init_compile() must be called in the training thread/process.
        reset_compile()
        init_compile()

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

    def start(self):
        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.clear_cache_before_training and self.config.latent_caching:
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()
        self.model_setup.validate_pre_load(self.config)

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        huggingface_util.configure_hub(
            self.config.secrets.huggingface_token,
            offline_mode=self.config.offline_mode,
            cache_dir=self.config.huggingface_cache_dir,
            on_status=self.callbacks.on_update_status,
        )

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model_sampler = None
        self.previous_sample_time = -1
        self.sample_queue = []

        if self.config.release_text_encoder_after_cache:
            self.__init_staged(model_names)
            return

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.eval()

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)

        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __init_staged(self, model_names):
        # Staged loading for release_text_encoder_after_cache: components are loaded one at a time,
        # each followed by its cache stage, so that only one heavy component is resident at once:
        #   metadata -> text encoder -> warm text cache -> release TE
        #   -> VAE -> warm image cache -> release VAE -> transformer -> LoRA/optimizer/sampler.
        # The point is that the transformer (the largest component) is loaded only after the TE and
        # VAE weights were freed, so a big/quantized DiT fits where loading everything at once OOMs.
        model_type = self.config.model_type
        weight_dtypes = self.config.weight_dtypes()
        quantization = self.config.quantization

        self.callbacks.on_update_status("staged loading: model metadata")
        self.model = self.model_loader.create_model(model_type, model_names)
        self.model.train_config = self.config
        self.model_loader.load_metadata(self.model, model_type, model_names, weight_dtypes, quantization)
        if not self.model.is_dev():
            raise NotImplementedError(
                "Staged loading (release_text_encoder_after_cache) currently supports only Flux.2-dev. "
                "Disable it to train Flux.2.Klein with the normal loading order."
            )
        # sets train_dtype/autocast; per-part optimizations are applied once per loaded part
        self.model_setup.setup_optimizations(self.model, self.config)

        # --- stage 1: text encoder + text cache ---
        self.__stage_marker("text encoder load: before")
        self.callbacks.on_update_status("staged loading: text encoder")
        self.model_loader.load_text_encoder(self.model, model_type, model_names, weight_dtypes, quantization)
        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.__stage_marker("text encoder load: after")

        self.callbacks.on_update_status("staged loading: warming the text cache")
        self.__create_warming_loaders("text")
        self.__warm_cache_until_ready(self.__text_cache_ready, "text cache")
        self.__drop_warming_loaders()

        self.__stage_marker("text encoder release: before")
        if not self.__release_text_encoder():
            raise RuntimeError(
                "Staged loading stopped: the text encoder was still referenced after release, so its "
                "memory was not freed and loading the next component would risk the same OOM. Report "
                "this with the list of enabled options."
            )
        self.__stage_marker("text encoder release: after")

        # --- stage 2: VAE + image cache ---
        self.__stage_marker("VAE load: before")
        self.callbacks.on_update_status("staged loading: VAE")
        self.model_loader.load_vae(self.model, model_type, model_names, weight_dtypes, quantization)
        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.__stage_marker("VAE load: after")

        self.callbacks.on_update_status("staged loading: warming the image cache")
        self.__create_warming_loaders("image")
        self.__warm_cache_until_ready(self.__image_cache_ready, "image (latent) cache")
        self.__drop_warming_loaders()

        self.__stage_marker("VAE release: before")
        if not self.__release_vae():
            raise RuntimeError(
                "Staged loading stopped: the VAE was still referenced after release, so its memory "
                "was not freed and loading the transformer would risk the same OOM. Report this with "
                "the list of enabled options."
            )
        self.__stage_marker("VAE release: after")

        # --- stage 3: transformer + LoRA/optimizer ---
        self.__stage_marker("transformer load: before")
        self.callbacks.on_update_status("staged loading: transformer")
        self.model_loader.load_transformer(self.model, model_type, model_names, weight_dtypes, quantization)
        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.__stage_marker("transformer load: after")

        # LoRA state dict, optimizer and offload conductors are created only now, on real
        # transformer parameters; resume state was picked up by create_model()/load_lora().
        self.callbacks.on_update_status("running model setup")
        self.model_loader.load_lora(self.model, model_names)
        self.model_setup.setup_model(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model.eval()

        # The real training pipeline is created fresh here (not reused from the warming stages):
        # its epoch counters start at train_progress again, so the first training epoch shuffles
        # and picks variations exactly like it would without staged loading.
        self.callbacks.on_update_status("creating the data loader/caching")
        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __stage_marker(self, label: str):
        print(f"[staged loading] {label}: {self.__memory_snapshot()}")

    def __create_warming_loaders(self, stage: str):
        # Slim pipelines that only warm one cache (see DataLoaderText2ImageMixin): the text stage
        # runs without any image/VAE modules, latent cache, batch sorting or batch assembly; the
        # image stage runs without any text-encoder modules. Module order up to the warmed cache
        # matches the training pipeline, so indices, balancing and variations are the standard ones.
        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress, cache_warming_stage=stage
        )
        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress,
                is_validation=True, cache_warming_stage=stage,
            )

    def __drop_warming_loaders(self):
        # The warming pipelines hold the component they encode (their encode modules) plus loaded
        # aggregate caches; drop them before the release so the component can be collected.
        self.data_loader = None
        self.validation_data_loader = None

    def __iter_pipeline_modules(self):
        loaders = [getattr(self, "data_loader", None), getattr(self, "validation_data_loader", None)]
        for loader in loaders:
            if loader is None:
                continue
            for module in loader.get_pipeline_modules():
                yield module

    def __warm_cache_until_ready(self, ready_fun: Callable[[], bool], what: str, max_rounds: int = 64):
        # mgds computes caches lazily at start_next_epoch(); each round refreshes the variation
        # window for the current pipeline epoch, so repeat until every group/variation reports a
        # finished cache. No training steps are run for this.
        for _ in range(max_rounds):
            if ready_fun():
                return
            self.data_loader.get_data_set().start_next_epoch()
            if self.config.validation:
                self.validation_data_loader.get_data_set().start_next_epoch()
        if not ready_fun():
            raise RuntimeError(
                f"Staged loading stopped: the {what} did not become complete after {max_rounds} "
                "caching rounds. Something about the dataset/cache prevents it from finishing; "
                "disable release_text_encoder_after_cache to fall back to the normal loading order."
            )

    def __image_cache_ready(self) -> bool:
        found = False
        for module in self.__iter_pipeline_modules():
            if isinstance(module, LatentCachingDiskCache):
                found = True
                if not module.all_variations_cached():
                    return False
        return found

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text']):
                    shutil.rmtree(path)

    @staticmethod
    def __memory_snapshot() -> str:
        # Diagnostics for release_text_encoder_after_cache: Windows RAM of this process and its
        # children (not system-wide memory), plus CUDA allocator allocated/reserved per device.
        parts = []
        try:
            import psutil
            proc = psutil.Process()
            rss = proc.memory_info().rss
            try:
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
            except (OSError, psutil.Error):
                pass
            parts.append(f"process RAM incl. children: {rss / 2 ** 30:.2f} GiB")
        except ImportError:
            parts.append("process RAM: n/a (psutil not available)")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                parts.append(
                    f"CUDA {i}: allocated {torch.cuda.memory_allocated(i) / 2 ** 30:.2f} GiB, "
                    f"reserved {torch.cuda.memory_reserved(i) / 2 ** 30:.2f} GiB"
                )
        return " | ".join(parts)

    def __text_cache_ready(self) -> bool:
        # True only if every text DiskCache in the training (and validation) pipeline reports that
        # all groups and all requested text variations have a finished cache on disk. Not just the
        # variations touched by the current epoch.
        loaders = [self.data_loader]
        validation_loader = getattr(self, "validation_data_loader", None)
        if validation_loader is not None:
            loaders.append(validation_loader)

        found = False
        for loader in loaders:
            for module in loader.get_pipeline_modules():
                if isinstance(module, TextCachingDiskCache):
                    found = True
                    if not module.all_variations_cached():
                        return False
        return found

    def __release_text_encoder(self) -> bool:
        # Returns True if the encoder was actually collected, False if something still holds it.
        model = self.model
        print("All text embeddings are cached: releasing the text encoder (release_text_encoder_after_cache)...")
        print(f"  memory before release: {self.__memory_snapshot()}")

        # weakref for diagnostics only - it does not keep the encoder alive. Deliberately no local
        # strong reference: it would outlive torch_gc() in this frame and skew the memory snapshot
        # and the collection check below.
        text_encoder_ref = weakref.ref(model.text_encoder) if model.text_encoder is not None else None

        # Drop every owner of a reference to the text encoder. The conductor owns the encoder's
        # layers and its allocator caches; the mgds encode modules hold the module directly; the
        # sampler's diffusers pipeline was created at startup and captured the encoder as well.
        model.text_encoder_offload_conductor = None

        # the warming pipelines were already dropped, so usually nothing here is left; iterate
        # defensively in case another pipeline still references the encoder
        for module in self.__iter_pipeline_modules():
            if isinstance(module, (EncodeMistralText, EncodeQwenText)):
                module.text_encoder = ReleasedTextEncoder()

        sampler_pipeline = getattr(self.model_sampler, "pipeline", None) if self.model_sampler is not None else None
        if sampler_pipeline is not None:
            sampler_pipeline.text_encoder = None

        model.text_encoder = None
        model.text_encoder_released = True

        torch_gc()

        print(f"  memory after release:  {self.__memory_snapshot()}")
        if text_encoder_ref is not None and text_encoder_ref() is not None:
            print(
                "  Warning: the text encoder is still referenced after release - something else holds it "
                "and its memory was not freed. Report this with the list of enabled options."
            )
            return False
        print("  Text encoder was garbage collected.")
        return True

    def __release_vae(self) -> bool:
        # Same contract as __release_text_encoder: drop every owner of a VAE reference, then verify
        # with a weakref (never a local strong reference) that the weights were actually collected.
        model = self.model
        print("All latents are cached: releasing the VAE (release_text_encoder_after_cache)...")
        print(f"  memory before release: {self.__memory_snapshot()}")

        vae_ref = weakref.ref(model.vae) if model.vae is not None else None

        # the training step still scales cached latents after the release -> keep independent
        # copies of the batch-norm stats while the VAE is alive
        model.save_vae_bn_stats()

        for module in self.__iter_pipeline_modules():
            if isinstance(module, (EncodeVAE, DecodeVAE, RandomLatentMaskRemove)):
                module.vae = ReleasedVae()

        model.vae = None
        model.vae_released = True

        torch_gc()

        print(f"  memory after release:  {self.__memory_snapshot()}")
        if vae_ref is not None and vae_ref() is not None:
            print(
                "  Warning: the VAE is still referenced after release - something else holds it "
                "and its memory was not freed. Report this with the list of enabled options."
            )
            return False
        print("  VAE was garbage collected.")
        return True

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    tqdm.write(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        with PeakMemoryRecorder("sampling", enabled=False):
            for fun in self.sample_queue:
                fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            ema_applied: bool,
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}"
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                tqdm.write("Error during sampling, proceeding without sampling")

            torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                tqdm.write("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        if self.model.ema:
            #the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            #non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
            ema_applied = self.config.ema != EMAMode.OFF
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
                ema_applied = False,
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("Calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                            average_loss,
                                            train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                tqdm.write("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
        except Exception:
            traceback.print_exc()
            tqdm.write("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                tqdm.write("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save(self, train_progress: TrainProgress, print_msg: bool = True):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            tqdm.write("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            tqdm.write("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                tqdm.write("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)


    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        ema_loss = None
        ema_loss_steps = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            #call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                if self.config.latent_caching:
                    self.data_loader.get_data_set().start_next_epoch()
                    self.model_setup.setup_train_device(self.model, self.config)
                else:
                    self.model_setup.setup_train_device(self.model, self.config)
                    self.data_loader.get_data_set().start_next_epoch()

            if self.config.release_text_encoder_after_cache \
                    and not getattr(self.model, "text_encoder_released", False) \
                    and self.__text_cache_ready():
                self.__release_text_encoder()

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()

            if multi.is_master():
                batches = step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length,
                                 initial=train_progress.epoch_step)
            else:
                batches = self.data_loader.get_data_loader()
            for batch in batches:
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                if not self.commands.get_stop_command() and self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        self.model.evict()
                        if backup:
                            self.__backup(train_progress, True)
                        if save:
                            self.__save(train_progress, True)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}-{get_string_timestamp()}.pickle"),
                    TorchProfiler      (enabled=False, filename=f"profile-step{train_progress.global_step}-{get_string_timestamp()}.json"),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)

                    prior_pred_indices = [i for i in range(self.config.batch_size)
                                          if ConceptType(batch['concept_type'][i]) == ConceptType.PRIOR_PREDICTION]
                    if len(prior_pred_indices) > 0 \
                            or (self.config.masked_training
                                and self.config.masked_prior_preservation_weight > 0
                                and self.config.training_method == TrainingMethod.LORA):
                        with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                            #do NOT create a subbatch using the indices, even though it would be more efficient:
                            #different timesteps are used for a smaller subbatch by predict(), but the conditioning must match exactly:
                            prior_model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
                        prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
                        model_output_data['target'][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
                        model_output_data['prior_target'] = prior_model_prediction
                    else:
                        model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

                    loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

                    loss = loss / self.config.gradient_accumulation_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    has_gradient = True
                    detached_loss = loss.detach()
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress):
                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError("Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation.")

                            self.tensorboard.add_scalar("loss/train_step",accumulated_loss_cpu , train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix({
                                'loss': accumulated_loss_cpu,
                                'smooth loss': ema_loss,
                            })
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if self.config.validation and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            self.model.evict()

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)
                if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )

        if self.model is not None:
            self.model.evict()

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
