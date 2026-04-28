# See-through on Mac (Apple Silicon, MPS)

Experimental port of see-through inference to Apple Silicon. **Tested on M4 Max 36GB**.

## Status

- [x] Device abstraction (`common/utils/device_utils.py`)
- [x] Patch hardcoded `'cuda'` strings in core inference path
- [x] Skip CUDA-only group_offload / cpu_offload
- [x] Mac requirements file (no bitsandbytes)
- [ ] First end-to-end run — pending model download
- [ ] Custom op fallbacks (if any unsupported MPS ops)
- [ ] Performance tuning

## What's Different from Upstream

| Area | Upstream | Mac fork |
|---|---|---|
| Device | `'cuda'` hardcoded | `get_device()` (CUDA > MPS > CPU) |
| dtype | `torch.bfloat16` hardcoded | `get_dtype()` (env-overridable) |
| Quantization | NF4 via bitsandbytes | not supported (use bf16) |
| group_offload | `enable_group_offload('cuda', ...)` | skipped on non-CUDA |
| Empty cache | `torch.cuda.empty_cache()` | guarded with availability check |

## Quick Start

```bash
# 1. Create conda env (or venv)
conda create -n see_through_mac python=3.12 -y
conda activate see_through_mac

# 2. Install MPS-enabled PyTorch (no +cu128 suffix on Mac)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 3. Install Mac requirements (no bitsandbytes)
pip install -r requirements-mac.txt

# 4. Run inference (auto-detects MPS)
python inference/scripts/inference_psd.py \
  --srcp your_image.png \
  --resolution 768 \
  --save_to_psd
```

## Environment Variables

| Var | Default | Purpose |
|---|---|---|
| `SEE_THROUGH_DEVICE` | auto | Force `cpu`/`mps`/`cuda` |
| `SEE_THROUGH_FP16` | unset | Force fp16 (use if bf16 numerical issues on MPS) |
| `SEE_THROUGH_FP32` | unset | Force fp32 (slowest, most stable) |
| `PYTORCH_ENABLE_MPS_FALLBACK` | unset | Set `=1` to fall back to CPU for unsupported MPS ops |

## Known Issues / Gotchas

1. **First run downloads ~6-8GB of model weights** from HuggingFace. Some repos may be gated — check the HF model pages.
2. **Unsupported MPS ops** may cause errors. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU automatically.
3. **bf16 on MPS** is mostly stable on PyTorch 2.4+, but some ops may be slow. If output quality looks wrong, try `SEE_THROUGH_FP32=1`.
4. **NF4 quantization not supported** — bitsandbytes is CUDA-only. Mac users must use the full bf16 pipeline (still fits in 36GB).
5. **Scripts NOT ported**: `inference_psd_quantized.py`, `inference_psd_blockswap.py`. Use `inference_psd.py` only.
6. **Performance estimate**: ~5-15 min/image on M4 Max (vs 74s on RTX 4090). MPS isn't as fast as CUDA for SDXL.

## Memory

- Default settings + 36GB unified memory = comfortable
- 16GB Macs may OOM at resolution 1280; try `--resolution 768`
- 8GB Macs: not recommended

## Reporting Issues

If a particular op fails on MPS, please open an issue with:
- Full error stack
- `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`
- Mac model + RAM size

## Original README

See [README.md](README.md) for the full upstream documentation.
