"""Standalone smoke: real Mistral Q4_K_M GGUF -> hidden states (Task 4 of the plan).

Loads the GGUF TE via GGUFTextEncoderLoaderMixin (packed weights, no full BF16 copy),
tokenizes two captions exactly like Flux2Model.encode_text (dev branch), and checks
the concatenated hidden states from MISTRAL_HIDDEN_STATES_LAYERS.
"""
import glob
import sys
import time

import torch

sys.path.insert(0, r"C:\Games\OneTrainer")

from modules.modelLoader.mixin.GGUFTextEncoderLoaderMixin import GGUFTextEncoderLoaderMixin


import ctypes


def rss_mb():
    # current RSS
    import psutil
    cur = psutil.Process().memory_info().rss / 1e6
    # peak working set on Windows
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return cur, pmc.PeakWorkingSetSize / 1e6


def main():
    base = glob.glob(r"C:\Users\anton\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-dev\snapshots\*")[0]
    gguf_path = r"C:\Games\ComfyUI_windows_portable\ComfyUI\models\text_encoders\Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"

    from transformers import Mistral3Config
    text_config = Mistral3Config.from_pretrained(base, subfolder="text_encoder").text_config

    t0 = time.time()
    mixin = GGUFTextEncoderLoaderMixin()
    te = mixin._load_gguf_text_encoder(gguf_path, text_config, torch.bfloat16)
    cur, peak = rss_mb()
    print(f"loaded in {time.time()-t0:.1f}s, RSS {cur:.0f} MB, peak WS {peak:.0f} MB")

    n_quant = sum(1 for m in te.modules() if type(m).__name__ == "GGUFLinear")
    packed = sum(p.numel() for p in te.parameters() if hasattr(p, "quant_type"))
    print(f"GGUFLinear modules: {n_quant}, packed param bytes: {packed/1e9:.2f} GB")

    from transformers import PixtralProcessor
    tokenizer = PixtralProcessor.from_pretrained(base, subfolder="tokenizer").tokenizer

    # exactly the dev-branch formatting from Flux2Model.encode_text
    from modules.model.Flux2Model import MISTRAL_SYSTEM_MESSAGE, MISTRAL_HIDDEN_STATES_LAYERS
    from diffusers.pipelines.flux2.pipeline_flux2 import format_input as mistral_format_input

    captions = [
        "a weathered metal helmet resting on a mossy stone",
        "neon rain over an empty highway at midnight",
    ]
    outs = []
    for cap in captions:
        messages = mistral_format_input(prompts=[cap], system_message=MISTRAL_SYSTEM_MESSAGE)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tok = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            out = te(tok.input_ids, attention_mask=tok.attention_mask.float(), output_hidden_states=True, use_cache=False)
        cat = torch.cat([out.hidden_states[k] for k in MISTRAL_HIDDEN_STATES_LAYERS], dim=2)
        print(f"caption={cap[:30]!r} hidden_states={len(out.hidden_states)} cat_shape={tuple(cat.shape)} "
              f"finite={torch.isfinite(cat).all().item()} mean={cat.float().mean().item():.4f} "
              f"forward={time.time()-t1:.1f}s RSS {rss_mb()[0]:.0f} MB peak WS {rss_mb()[1]:.0f} MB")
        outs.append(cat)

    same = torch.allclose(outs[0], outs[1])
    diff = (outs[0] - outs[1]).abs().mean().item()
    print(f"different captions -> different conditioning: {not same} (mean abs diff {diff:.4f})")
    cur, peak = rss_mb()
    print(f"FINAL RSS {cur:.0f} MB peak WS {peak:.0f} MB")
    assert len(out.hidden_states) == 41, "expected 41 hidden states (embeddings + 40 layers)"
    assert outs[0].shape == (1, 128, 3 * 5120)
    assert not same


if __name__ == "__main__":
    main()
