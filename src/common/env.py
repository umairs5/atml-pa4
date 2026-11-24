"""Shared environment, seeding, hardware metadata, and CUDA timing helpers."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import transformers

COMMON_SEED = 6304


def set_seed(seed: int = COMMON_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_env_metadata(dataset_root: Optional[Path] = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "common_seed": COMMON_SEED,
    }
    if dataset_root is not None:
        lock_path = Path(dataset_root) / "data" / "local_source_revisions.json"
        if lock_path.exists():
            meta["source_revisions"] = json.loads(lock_path.read_text(encoding="utf-8"))
    return meta


def cuda_sync_if_available(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device is None:
            torch.cuda.synchronize()


def measure_cuda_latency_ms(
    fn: Callable[[], Any],
    warmup: int = 3,
    repeats: int = 5,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Run warmed-up, CUDA-synchronized timings and return summary stats in milliseconds."""
    import time

    for _ in range(warmup):
        fn()
    cuda_sync_if_available(device)

    times_ms: list[float] = []
    use_cuda_events = torch.cuda.is_available() and (device is None or device.type == "cuda")
    for _ in range(repeats):
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            cuda_sync_if_available(device)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    return {
        "unit": "ms",
        "warmup": warmup,
        "repeats": repeats,
        "samples_ms": times_ms,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if repeats > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def reset_and_measure_peak_memory_bytes(fn: Callable[[], Any], device: Optional[torch.device] = None) -> tuple[Any, dict[str, int]]:
    """Measure peak allocated and reserved CUDA memory during `fn`."""
    if not torch.cuda.is_available() or (device is not None and device.type != "cuda"):
        out = fn()
        return out, {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = fn()
    torch.cuda.synchronize(device)
    return out, {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }

