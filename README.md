# OneTrainer — Flux.2 Optimizations Fork

A fork of [OneTrainer](https://github.com/Nerogar/OneTrainer) focused on making
**FLUX.2-dev LoRA training feasible on consumer hardware** by cutting peak RAM/VRAM.

## What this fork changes

- **Staged model loading** — the text encoder and VAE are loaded only to warm the text
  and latent caches, then fully released *before* the transformer is loaded. Peak RAM no
  longer holds TE + transformer at the same time.
- **GGUF quantized text encoder override** — point `text_encoder_model` at a GGUF file;
  the full BF16 text encoder is never loaded. Tested with Q4_K_M; other quantizations
  are unverified.
- **Sample conditioning cache** — fixed sample prompts are encoded during the text-cache
  stage, so scheduled samples never reload the text encoder.
- **Memory-storm fixes** — idempotent layer-offload materialization (no re-cloning of all
  GGUF weights on every sample), VAE decode without evicting the transformer, LoRA
  backups/saves skip the transformer evict (~15 GB saved), and the working set is trimmed
  after releases so freed RAM is actually returned to the OS.
- **Streaming model saves** — safetensors hashing without full-tensor copies.

## Credits

- **[OneTrainer](https://github.com/Nerogar/OneTrainer)** by Nerogar and contributors —
  the upstream project this fork is based on.
- **[MGDS](https://github.com/Nerogar/mgds)**: A custom dataset implementation for Pytorch
  that is built around the idea of a node based graph.
- **[Stability Matrix](https://github.com/LykosAI/StabilityMatrix)**: A swiss-army knife
  installer which wraps and installs a broad range of diffusion software packages
  including OneTrainer.
- **[Visions of Chaos](https://softology.pro/voc.htm)**: A collection of machine learning
  tools that also includes OneTrainer.
- **[StableTuner](https://github.com/devilismyfriend/StableTuner)**: A now defunct
  (archived) training application for Stable Diffusion. OneTrainer takes a lot of
  inspiration from StableTuner and wouldn't exist without it.
