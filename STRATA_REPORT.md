# Strata: семплы, освобождение TE/VAE и пики памяти при save/backup — отчёт

Commit базы: `36f8d5a3` (ветка `Flux.2D-optimizations`), все изменения — поверх него, в рабочем дереве (не закоммичены).
Окружение агента: Windows 10, RTX 3090 24 GiB, torch 2.12.0+cu130, diffusers 0.39.0.dev0, transformers 5.5.4, safetensors 0.8.0.

## 1. Что подтверждено кодом (не гипотезы)

### 1.1 Что входит в бекап и откуда временные гигабайты
- **Состав INTERNAL-бекапа зависит от training_method.** Для **LoRA** (`Flux2LoRAModelSaver` → `LoRASaverMixin._save_internal`): `lora/lora.safetensors` (только LoRA-веса) + `optimizer/optimizer.pt` + `ema/ema.pt` + `meta.json` + конфиг. **Базовые веса трансформера в LoRA-бекап НЕ пишутся.** Для **FINE_TUNE** (`Flux2ModelSaver`, `case ModelFormat.INTERNAL`) — да, полный diffusers-сейв базы (это нужно для resume).
- **Источник лишнего пика при LoRA-бекапе — не сериализация, а `self.model.evict()` ПЕРЕД ней** (в цикле обучения и в `end()`). При включённом layer-offload `evict()` → `LayerOffloadConductor.evict()`: для каждого слоя `layer.to(temp)` + `offload_quantized(..., allocator=clone_tensor_allocator)` — **клон каждого тензора весов** плюс полный GPU↔CPU round-trip всей модели, а после сейва `setup_train_device` → `conductor.materialize()` гонит всё обратно и заново аллоцирует cache-блоки аллокаторов. Для GGUF Flux.2-dev (~13 GiB весов на CPU) этот цикл сам по себе — многогигабайтный транзиент (порядка +1x байт модели суммарно по фазам), хотя сериализуется при этом только LoRA. **Исправлено:** `__evict_model_for_backup_or_save()` — для `TrainingMethod.LORA` evict не делается вообще (сейв/бекап LoRA не трогают базовые веса), для fine-tune/embedding остаётся как было. Тот же путь в `end()` (финальные backup_before_save + save).
- **Хеш safetensors делал полную копию каждого тензора** (`tobytes()` материализует весь тензор разом). **Исправлено:** хеш стримится чанками по 1 МиБ поверх zero-copy uint8-представления; побайтовая эквивалентность старого/нового хеша покрыта тестом (`tests/test_save_path.py` — эталон = буквально старая реализация).
- **`_convert_state_dict_dtype(state_dict, None)` не копирует** уже лежащие на CPU тензоры (`.to(cpu, dtype=None)` — no-op); тест есть.
- **sample / backup / save исполняются последовательно в одном месте цикла** (`__execute_sample_during_training()` → backup → save), их пики не суммируются.

### 1.2 Освобождение TE/VAE и семплы (новая семантика)
- Staged loading: TE → текстовый кеш → релиз TE; VAE → латентный кеш → релиз VAE; затем transformer+LoRA. Релиз проверяется weakref: не собран — явная ошибка, не идём дальше в OOM. **Полный валидный кеш → компонент не грузится вообще** (стадии зондируют кеш с незагруженным компонентом); BN-статы для скейлинга латентов читаются прямо из файлов VAE (`load_vae_bn_stats_from_disk`, работает и с HF-hub id через subfolder).
- **Семплы кешируются ВМЕСТЕ со всем остальным в начале.** На стадии текстового кеша (`__warm_sample_conditioning`) все включённые фиксированные sample-промпты кодируются, пока TE резидентен, и персистятся в `cache_dir/sample_conditioning.pt` с fingerprint (model_type, base/TE id, dtype TE, sequence length). Ключ словаря точно совпадает с ключом сэмплера: `(prompt, negative_prompt|None, text_encoder_1_sequence_length, batch_size)`, batch_size=2 только при cfg_scale>1 и отсутствии guidance_embeds (флаг берётся из `transformer_config`, загруженного `load_metadata` — трансформер для прогрева не грузится).
  - **После релиза TE не грузится ни на какой стадии**: scheduled-семпл — always cache hit (словарь живёт на модели, `Flux2Sampler` биндит его при конструировании; обновление словаря — in-place). Единственное исключение — **ручной custom-семпл с новым промптом**: его физически нельзя было закодировать заранее, для него работает on-demand reload хуком с последующим повторным релизом.
  - При полностью валидном текстовом кеше и холодном conditioning-кеше TE грузится **один раз за прогон** — на прогрев промптов (это часть фазы кеширования, до релиза). При прогретом `sample_conditioning.pt` — **ноль загрузок TE** (проверено тестом: второй staged-init не вызывает `load_text_encoder`).
  - Смена TE/dtype/модели инвалидирует файл по fingerprint → автоперезапись.
- **RNG:** тренировочный шаг сидится из `train_progress.global_step`, сэмплер — из собственного `torch.Generator`; глобальный RNG не расходуется → resume детерминирован без RNG-файла (тест `test_training_noise_is_derived_from_global_step`).
- После семплов трейнер отпускает подгруженное и восстанавливает train-mode раскладкой `setup_train_device`; VAE для decode грузится только на decode-фазу (`materialize_only("vae")`) и снова релизится.

### 1.3 Метки пиков
- `PeakMemoryRecorder` (`modules/util/profiling_util.py`): опрашивает **private RAM** процесса (с дочерними) на daemon-потоке каждые 100 мс в течение with-блока + `torch.cuda.reset_peak_memory_stats()`/`max_memory_allocated` для VRAM. Обёрнуты: `backup`, `save`, `sampling`. Печатает `[peak_ram] <метка>: N GiB` и `[peak_vram] <метка>: N GiB (reserved M GiB)`.
- `[staged loading] <этап>: <снимок>` — RSS+дети, CUDA allocated/reserved по GPU — на границах: прогрев текстового кеша, **прогрев conditioning семплов (before/after)**, релиз TE, прогрев латентного кеша, релиз VAE, загрузка transformer, reload TE/VAE по требованию, старт эпохи, первый тренировочный шаг.

## 2. Изменения (код)

| Файл | Суть |
|---|---|
| `modules/trainer/GenericTrainer.py` | staged loading: skip-load при валидном кеше, reload hooks, `__warm_sample_conditioning` (+persist/fingerprint), `__evict_model_for_backup_or_save` (LoRA — без evict), `PeakMemoryRecorder` на backup/save/sampling, `[staged loading]` маркеры |
| `modules/model/BaseModel.py` | `component_reload_hooks` + вызов хука в `_move_part` при materialize на train_device |
| `modules/model/Flux2Model.py` | `sample_conditioning_cache` на модели; `load_vae_bn_stats_from_disk`, `save_vae_bn_stats`, fallback `_vae_bn_stats` |
| `modules/modelSampler/Flux2Sampler.py` | conditioning берётся из `model.sample_conditioning_cache` (hit → TE не трогаем вообще); miss → on-demand reload; decode через `model.vae` (актуальный после reload) |
| `modules/modelSaver/mixin/DtypeModelSaverMixin.py` | стриминговый хеш safetensors (без `.tobytes()`) |
| `modules/modelSetup/BaseFlux2Setup.py` | семплы разрешены в режиме релиза; `prepare_text_caching` через materialize-хук |
| `modules/util/component_lifecycle.py` (новый) | `ComponentNotLoadedError` — зонд «кеш неполный» |
| `modules/ui/BaseTrainUIView.py` | tooltip обновлён |
| `tests/*` | 70 лёгких тестов, см. ниже |

## 3. Результаты лёгких тестов (реальный вывод агента)

```
$ venv\Scripts\python.exe -m pytest tests\ -q
70 passed, 16 warnings in 6.67s
```

Покрыто (tiny/stub, без GPU и весов): полный порядок staged-стадий; неполный кеш → on-demand reload и повторный релиз; TE/VAE реально собираются GC (weakref); transformer не грузится при сбое кеша; **conditioning фиксированных промптов греется на стадии текстового кеша, TE грузится ровно один раз и релизится; embedding персистится в cache_dir; повторный staged-init с тем же конфигом не грузит TE вообще**; **LoRA backup/save не вызывает evict, fine-tune — вызывает**; BN-статы побитово идентичны после релиза (fp32/fp16/bf16) и не держат VAE; slim-пайплайны прогрева; стриминговый хеш == старый `.tobytes()`-хеш (вкл. неконтигуальные и fp16/bf16); `_convert_state_dict_dtype(None)` не копирует CPU-тензоры; resume восстанавливает `train_progress`+optimizer+EMA из backup-директории; шум шага детерминирован из `global_step`; фиксированный промпт в сэмплере кодируется один раз, VAE decode — на каждом семпле.

Ad-hoc верификация (одноразовый скрипт, удалён): хеш-эквивалентность, no-copy dtype=None, conditioning cache (1 encode / 2 decode), reload hook срабатывает ровно один раз.

**НЕ ПРОВЕРЕНО агентом:** полный GPU-прогон, качество изображений, реальные пики RAM/VRAM, реальный resume на весах. Цифра «+15 ГБ» — сообщение пользователя; механизм (evict/re-materialize цикл вокруг LoRA-сейва) подтверждён кодом, количественно его меряет только полный прогон.

## 4. Команды для полного прогона (Windows, cmd)

Конфиг: включить `Release text encoder after cache` (вкладка модели), хотя бы один sample (проверить первую и повторную генерацию одним промптом), `Sample after` так, чтобы семпл попал в короткий прогон; `Backup after` / `Save every` — по шагам (например 2 и 3). Экспортировать конфиг (UI → экспорт) в `C:\Games\OneTrainer\run_config.json`.

```bat
cd /d C:\Games\OneTrainer
call venv\Scripts\activate.bat

REM полный прогон: >=2 шага, переход эпохи, семпл, LoRA save, backup; лог в файл
python scripts\train.py --config-path run_config.json > run1.log 2>&1

REM resume из последнего бекапа (в конфиге continue_last_backup=true или
REM указать model_names.lora = путь к backup-папке)
python scripts\train.py --config-path run_config_resume.json > run2.log 2>&1
```

Генерация семплов с LoRA — только внутри тренировочного прогона (вкладка Sampling): `scripts\sample.py` — standalone без LoRA, годится только для «голой» базы. Ожидаемое поведение в логе: строка `Sample conditioning cached for N prompt(s) ...` до релиза TE; после релиза — НИ ОДНОЙ `text encoder reload: before` для scheduled-семплов (только для ручного custom-промпта).

Что собрать из логов (агенту для разбора):
- строки `[peak_ram] backup/save/sampling: ...` и `[peak_vram] ...` — ДО и ПОСЛЕ изменений (для ДО: `git stash` и тот же прогон);
- все строки `[staged loading] ...` — baseline до/после каждого этапа;
- время прогона и наличие `Warning: ... still referenced after release`;
- после resume: совпадает ли `global_step` и численно ли совпадают следующие N шагов с контрольным прогоном (loss по шагам из tensorboard).

Контроль воспроизводимости: два прогона с одинаковым конфигом и seed; сравнить loss шагов 1..N и имена/размеры файлов save/backup.

## 5. Гипотезы (не подтверждены измерениями)

- Ожидаемый эффект skip-evict: пик `[peak_ram] backup` для LoRA падает примерно на размер байтов трансформера в CPU-раскладке (GGUF Flux.2-dev ~13 GiB) — ровно тот транзиент, который убран. Остаток пика — optimizer.pt/ema.pt (для LoRA малые) + сам сейв.
- Для FINE_TUNE INTERNAL backup пик остаётся (~CPU-копия базы, она нужна для resume); следующий кандидат на улучшение (НЕ сделан): shard-сериализация `__save_diffusers` без полной CPU-копии.
- `materialize_only("vae")` в decode-фазе семпла по-прежнему делает conductor-цикл по трансформеру (evict→materialize) — это унаследованное поведение offload-механики, не трогали; его стоимость покажут замеры `[peak_ram] sampling`.

## 6. Разбор логов пользователя от 24.10 и исправления

**Лог 1 (холодный кэш).** Релиз TE отработал (`Text encoder was garbage collected`), но RAM не упала (15.50 → 15.48 GiB), а transformer-стадия упала с `cudaHostRegister error 712` (cudaErrorHostMemoryAlreadyRegistered).
Причина: при layer-offload веса TE живут в pinned-буферах conductor'а (`cudaHostRegister`). Релиз сбрасывал `text_encoder_offload_conductor = None` без `cudaHostUnregister` — страницы оставались запинены драйвером (RAM не возвращается ОС), а следующий `torch.zeros` на том же адресном диапазоне падал с 712.
Исправление: `LayerOffloadConductor.release()` — финальный teardown (deallocate_cache по всем трём аллокаторам, т.е. unpin), вызывается из `__release_text_encoder` перед сбросом conductor'а.

**Лог 2 (тёплый кэш, `text encoder reload` при семпле).** Ключ conditioning-кэша не совпадал с ключом сэмплера: в `config.json` FLUX.2-dev НЕТ ключа `guidance_embeds`, дефолт класса `Flux2Transformer2DModel` — `True` (сэмплер: batch 1, negative не в ключе), а warmup читал с дефолтом `False` (batch 2, negative в ключе) → гарантированный промах и перезагрузка TE на каждом scheduled-семпле.
Исправление: дефолт в `__sample_conditioning_key` приведён к `True` (как в классе модели); в fingerprint conditioning-кэша добавлен `guidance_embeds`, поэтому старый `sample_conditioning.pt` (с кривыми ключами) будет автоматически перекодирован один раз.
Цифра `33.68 → 5.16 GiB` в строке reload — не «освободился TE»: это conductor трансформера отпустил свои кэши при evict перед загрузкой TE; TE поверх этого только добавляет (~12 GiB).

Тесты: `test_sample_conditioning_key_matches_sampler_without_guidance_key`, `test_encoder_collected_and_no_still_referenced_warning` (проверяет `conductor.release()`); всего `71 passed`.
