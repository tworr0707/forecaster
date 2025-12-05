"""Central helpers for detecting and surfacing GPU OOMs."""

import re
import torch


class ForecastingOOMError(RuntimeError):
    """Raised when GPU/allocator OOM is detected in forecasting pipeline."""


_OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"cuda out of memory",
        r"CUDA error: out of memory",
        r"CUDART_STATUS_ALLOC_FAILED",
        r"failed to allocate",
        r"memory allocation error",
    ]
]


def is_oom_error(exc: BaseException) -> bool:
    """Best-effort classification of CUDA/allocator OOMs."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc) if exc else ""
    return any(p.search(msg) for p in _OOM_PATTERNS)


def wrap_oom(exc: BaseException) -> ForecastingOOMError:
    """Return a ForecastingOOMError wrapping the original exception."""
    return ForecastingOOMError(f"GPU OOM: {exc}")
