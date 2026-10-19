# Ручная проверка: release_text_encoder_after_cache (Flux.2 LoRA, staged loading)

## Что должно произойти

При включённом флаге модель грузится **по этапам**, чтобы в каждый момент в памяти жил
только один тяжёлый компонент:

```
metadata → text encoder → прогрев текстового кэша → релиз TE
         → VAE → прогрев image (latent) кэша → релиз VAE
         → transformer (DiT) → LoRA/optimizer/sampler → обучение
```

Смысл: самый большой компонент (DiT ~30+ GiB) грузится только после того, как веса TE и VAE
освобождены — туда, где обычная загрузка всего сразу даёт OOM.

Прогрев кэшей идёт **отдельными slim-пайплайнами**, а не полным тренировочным:
- текстовый этап: enumerate → load → bucketing → crop → augmentations → tokenize → TE →
  text DiskCache → VariationSorting. Нет latent cache, EncodeVAE, сортировки картинок и
  сборки батчей — VAE не касается вообще;
- image-этап: те же модули до EncodeVAE → image DiskCache. Нет токенизации/TE/text cache.

Порядок модулей до прогреваемого кэша идентичен тренировочному пайплайну, поэтому индексы,
balancing и variations штатные. Перед релизом компонента warming-лоадер выбрасывается, а
после загрузки DiT **тренировочный пайплайн создаётся заново** (не перематывается прогретый),
так что первая тренировочная эпоха перемешивает и выбирает вариации точно как без staged-режима.

## Подготовка

1. OneTrainer, модель **Flux.2-dev** (можно с GGUF TE override — фичи совместимы).
2. В настройках обучения включить:
   - **Latent caching** (обязательно)
   - **Release text encoder after cache** (switch рядом с latent caching)
3. Условия валидации (иначе ошибка ДО загрузки модели, сразу после Apply):
   - метод обучения = LoRA (не fine-tune / Distillation)
   - text encoder train = выключен
   - samples во время обучения = выключены (пусто)
   - латентный кэшинг = включён
   - FLUX.2-dev (Klein → ошибка NotImplementedError после чтения metadata)

## Прогон

1. Новый конфиг → Flux.2-dev, небольшой датасет (5-10 картинок), LoRA, latent caching ON,
   release TE ON, samples пусто, epochs 2.
2. В логе на каждом этапе печатается снимок памяти:

```
[staged loading] text encoder load: before: process RAM incl. children: ~X GiB | CUDA 0: ...
[staged loading] text encoder load: after: ...
All text embeddings are cached: releasing the text encoder (release_text_encoder_after_cache)...
  memory before release: ...
  memory after release:  ...
  Text encoder was garbage collected.
[staged loading] text encoder release: after: ...
[staged loading] VAE load: before: ...
All latents are cached: releasing the VAE (release_text_encoder_after_cache)...
  VAE was garbage collected.
[staged loading] VAE release: after: ...
[staged loading] transformer load: before: ...
```

3. Пик RAM/VRAM за весь прогон должен быть **меньше**, чем при обычном режиме: TE и VAE
   никогда не сосуществуют с DiT.
4. После загрузки обучение идёт как обычно; эмбеддинги и латенты читаются с диска.

## Критерии успеха

- [ ] Конфиг с release TE + fine-tune / text_encoder.train / samples → ValueError сразу,
      модель не грузится. Klein → NotImplementedError после metadata.
- [ ] Строки `Text encoder was garbage collected.` и `VAE was garbage collected.`
      (не Warning про "still referenced").
- [ ] Падение RAM/VRAM в логе до/после каждого релиза; пик ниже обычного режима.
- [ ] Эпохи, backup/сейв работают; LoRA сейв не содержит TE/VAE LoRA.
- [ ] Стоп/резюм после релиза работает (кэш на диске, TE/VAE не требуются).
- [ ] После релиза VAE обучение продолжается: `scale_latents` работает на сохранённых BN-статах
      (сохраняются до освобождения: `save_vae_bn_stats()` — detached CPU-клоны running_mean/var + eps).
- [ ] Перемешивание/выбор вариаций в первой реальной эпохе не сдвинуты относительно
      обычного режима (после прогрева кэшей тренировочный пайплайн создаётся заново).

## Негативные проверки

- Выключенный switch → обычный путь загрузки, всё как раньше.
- Старый конфиг JSON без поля → дефолт OFF, грузится без ошибок.
- Release TE на не-Flux.2 модели → ValueError "only supported for Flux.2".
- Если что-то удерживает TE/VAE после релиза → явный RuntimeError
  ("was still referenced after release") вместо тихого OOM на следующем этапе.

## Автотесты (проходят)

```
env -u PYTHONPATH venv/Scripts/python.exe -m pytest tests/ -q
```

50 passed (15 новых staged-loading в tests/test_staged_loading.py + 17 release-te + 18 GGUF).

## Diff-сводка (git diff --stat)

```
modules/dataLoader/BaseDataLoader.py               |   7 +   get_pipeline_modules(), cache_warming_stage
modules/dataLoader/Flux2BaseDataLoader.py          |  43 +-  _image/_text_preparation_modules, slim dispatch
modules/dataLoader/mixin/DataLoaderText2ImageMixin.py | 186 ++  ReadinessDiskCache, slim warming pipelines
modules/model/Flux2Model.py                        |  24 +-  transformer_config, is_dev() до загрузки DiT
modules/modelLoader/Flux2ModelLoader.py            | 205 ++++  staged API: load_metadata/TE/VAE/DiT
modules/modelLoader/GenericLoRAModelLoader.py      |  73 ++   staged load API
modules/modelSetup/BaseFlux2Setup.py               |  44 +-   validate_pre_load, attention guard
modules/modelSetup/BaseModelSetup.py               |  22 ++   optimized_parts (идемпотентность)
modules/modelSetup/Flux2LoRASetup.py               |  24 +-   guards для None TE/VAE/DiT
modules/trainer/BaseTrainer.py                     |   3 +-   cache_warming_stage passthrough
modules/trainer/GenericTrainer.py                  | 295 +++  __init_staged, slim прогрев/релиз TE+VAE
modules/util/create.py                             |   5 +-   cache_warming_stage passthrough
modules/ui/BaseTrainUIView.py                      |   5 +    tooltip staged-режима
modules/util/config/TrainConfig.py                 |  18 +-   флаг release_text_encoder_after_cache
+ modules/util/released_text_encoder.py            (placeholder после релиза TE)
+ modules/util/released_vae.py                     (placeholder после релиза VAE)
+ tests/test_te_release_after_cache.py             (17 тестов)
+ tests/test_staged_loading.py                     (15 тестов)
```
