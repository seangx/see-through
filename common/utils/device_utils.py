"""Cross-backend device helpers (CUDA / MPS / CPU).

Adds a thin abstraction so the rest of the codebase doesn't hardcode 'cuda'.
Kept self-contained — only depends on torch.
"""
from __future__ import annotations
import os
import torch


def get_device() -> str:
    """Pick the best available device.

    Override with env var `SEE_THROUGH_DEVICE` (e.g. 'cpu', 'mps', 'cuda').
    Priority: env > cuda > mps > cpu.
    """
    forced = os.environ.get("SEE_THROUGH_DEVICE", "").lower()
    if forced in ("cpu", "mps", "cuda"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    """Pick a sensible float dtype for the device.

    bf16 is the upstream default (matches CUDA pinned recipe), but on MPS bf16
    is unstable on older PyTorch — fall back to fp16 when SEE_THROUGH_FP16 is set,
    or fp32 with SEE_THROUGH_FP32.
    """
    if os.environ.get("SEE_THROUGH_FP32"):
        return torch.float32
    if os.environ.get("SEE_THROUGH_FP16"):
        return torch.float16
    if device is None:
        device = get_device()
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # MPS bf16 is supported on PyTorch 2.4+; if you hit issues set SEE_THROUGH_FP16=1
        return torch.bfloat16
    return torch.float32


def empty_cache(device: str | None = None) -> None:
    """Backend-aware cache clear. No-op on CPU."""
    if device is None:
        device = get_device()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def reset_peak_memory_stats(device: str | None = None) -> None:
    if device is None:
        device = get_device()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def max_memory_allocated_gb(device: str | None = None) -> float:
    if device is None:
        device = get_device()
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return torch.mps.current_allocated_memory() / 1024**3
        except Exception:
            return 0.0
    return 0.0


def supports_offload(device: str | None = None) -> bool:
    """diffusers' enable_*_offload() shims only work on CUDA."""
    if device is None:
        device = get_device()
    return device == "cuda"


def ipc_collect(device: str | None = None) -> None:
    """Only meaningful on CUDA."""
    if device is None:
        device = get_device()
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
