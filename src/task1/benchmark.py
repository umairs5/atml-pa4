"""Warmed-up, CUDA-synchronized prefill/decode latency and peak VRAM benchmarks for Task 1."""

from __future__ import annotations

from typing import Any

import torch

from src.common.env import measure_cuda_latency_ms, reset_and_measure_peak_memory_bytes


@torch.inference_mode()
def benchmark_inference(
    model: torch.nn.Module,
    vocab_size: int,
    device: str | torch.device = "cuda",
    warmup: int = 3,
    repeats: int = 5,
    decode_prompt_len: int = 512,
    decode_steps: int = 64,
    seed: int = 6304,
) -> dict[str, Any]:
    """Measure 512-token prefill, 2048-token prefill, 64-token decode latency, and peak VRAM.

    Note: All pruned/fake-quantized FP16 models execute dense FP16 kernels unless an explicit
    sparse/low-bit kernel is used; results must be labeled accordingly.
    """
    model.eval()
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    p512 = torch.randint(100, min(vocab_size, 50000), (1, 512), generator=gen, dtype=torch.long).to(dev)
    p2048 = torch.randint(100, min(vocab_size, 50000), (1, 2048), generator=gen, dtype=torch.long).to(dev)
    p_dec = p512[:, :decode_prompt_len]
    next_tok = torch.randint(100, min(vocab_size, 50000), (1, 1), generator=gen, dtype=torch.long).to(dev)

    def run_prefill_512():
        return model(input_ids=p512, use_cache=False)

    def run_prefill_2048():
        return model(input_ids=p2048, use_cache=False)

    # Precompute fixed KV cache once outside timed 64-token decode, then clone/copy or rebuild?
    # Rebuilding cache per repeat before timed section ensures identical starting cache state.
    def run_decode_64_once() -> float:
        import time

        # Build prompt KV cache outside timer
        out = model(input_ids=p_dec, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:, :].argmax(dim=-1)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(decode_steps):
                step_out = model(input_ids=cur, past_key_values=pkv, use_cache=True)
                pkv = step_out.past_key_values
                cur = step_out.logits[:, -1:, :].argmax(dim=-1)
            end.record()
            torch.cuda.synchronize(dev)
            return float(start.elapsed_time(end))
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            step_out = model(input_ids=cur, past_key_values=pkv, use_cache=True)
            pkv = step_out.past_key_values
            cur = step_out.logits[:, -1:, :].argmax(dim=-1)
        return (time.perf_counter() - t0) * 1000.0

    # Measure prefill 512
    _, mem_512 = reset_and_measure_peak_memory_bytes(run_prefill_512, dev)
    lat_512 = measure_cuda_latency_ms(run_prefill_512, warmup=warmup, repeats=repeats, device=dev)

    # Measure prefill 2048
    _, mem_2048 = reset_and_measure_peak_memory_bytes(run_prefill_2048, dev)
    lat_2048 = measure_cuda_latency_ms(run_prefill_2048, warmup=warmup, repeats=repeats, device=dev)

    # Measure 64-token decode
    for _ in range(warmup):
        run_decode_64_once()
    _, mem_dec64 = reset_and_measure_peak_memory_bytes(run_decode_64_once, dev)
    dec_samples = [run_decode_64_once() for _ in range(repeats)]
    import numpy as np

    dec_arr = np.array(dec_samples, dtype=np.float64)
    lat_dec64 = {
        "unit": "ms",
        "warmup": warmup,
        "repeats": repeats,
        "decode_prompt_len": decode_prompt_len,
        "decode_steps": decode_steps,
        "samples_ms": dec_samples,
        "mean_ms": float(dec_arr.mean()),
        "std_ms": float(dec_arr.std(ddof=1)) if repeats > 1 else 0.0,
        "median_ms": float(np.median(dec_arr)),
        "ms_per_token": float(dec_arr.mean() / decode_steps),
        "tokens_per_sec": float(decode_steps / (dec_arr.mean() / 1000.0)),
    }

    overall_peak_alloc = max(
        mem_512["peak_allocated_bytes"],
        mem_2048["peak_allocated_bytes"],
        mem_dec64["peak_allocated_bytes"],
    )
    overall_peak_reserved = max(
        mem_512["peak_reserved_bytes"],
        mem_2048["peak_reserved_bytes"],
        mem_dec64["peak_reserved_bytes"],
    )

    return {
        "execution_mode": "dense_fp16",
        "prefill_512": {**lat_512, **mem_512},
        "prefill_2048": {**lat_2048, **mem_2048},
        "decode_64": {**lat_dec64, **mem_dec64},
        "peak_allocated_bytes": overall_peak_alloc,
        "peak_reserved_bytes": overall_peak_reserved,
    }

