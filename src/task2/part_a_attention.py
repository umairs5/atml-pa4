"""Task 2 Part A: Dense causal attention, online-softmax tiled attention, and PyTorch SDPA."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.common.env import measure_cuda_latency_ms, reset_and_measure_peak_memory_bytes


def dense_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    softmax_fp32: bool = True,
) -> torch.Tensor:
    """Explicit QK^T / sqrt(d) causal attention with stable softmax.

    q, k, v: [B, H, L, d_h]
    attn_mask: optional boolean mask of shape [..., L, L] where True = allowed, False = masked.
               If None, uses standard causal lower-triangular mask j <= i.
    """
    B, H, L, d_h = q.shape
    scale = 1.0 / math.sqrt(d_h)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is None:
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=q.device))
    else:
        causal = attn_mask.to(dtype=torch.bool, device=q.device)

    neg_inf = torch.finfo(torch.float32 if softmax_fp32 else scores.dtype).min
    work = scores.float() if softmax_fp32 else scores
    work = work.masked_fill(~causal, neg_inf)

    # Guard against all-masked rows (return 0 output if an entire row is masked)
    has_valid = causal.any(dim=-1, keepdim=True)
    row_max = work.max(dim=-1, keepdim=True).values
    row_max = torch.where(has_valid, row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(work - row_max) * has_valid.to(work.dtype)
    exp_scores = torch.where(causal, exp_scores, torch.zeros_like(exp_scores))
    denom = exp_scores.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, exp_scores / denom.clamp(min=1e-12), torch.zeros_like(exp_scores))
    out = torch.matmul(probs.to(v.dtype), v)
    return out


def tiled_online_softmax_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """Exact blockwise causal attention using online softmax without materializing the L x L matrix.

    Note: In Python/PyTorch without a custom CUDA backward kernel, autograd saves per-tile
    intermediates for backward; this limitation is documented explicitly in benchmark reports.
    """
    B, H, L, d_h = q.shape
    scale = 1.0 / math.sqrt(d_h)
    out_chunks = []

    for q_start in range(0, L, block_q):
        q_end = min(q_start + block_q, L)
        q_tile = q[:, :, q_start:q_end, :]  # [B, H, bq, d_h]
        bq = q_end - q_start

        m = torch.full((B, H, bq, 1), -float("inf"), dtype=torch.float32, device=q.device)
        ell = torch.zeros((B, H, bq, 1), dtype=torch.float32, device=q.device)
        acc = torch.zeros((B, H, bq, d_h), dtype=torch.float32, device=q.device)

        q_pos = torch.arange(q_start, q_end, device=q.device)[:, None]
        # Causal skipping: only iterate kv tiles with kv_start < q_end
        max_kv_end = q_end
        for kv_start in range(0, max_kv_end, block_kv):
            kv_end = min(kv_start + block_kv, L)
            k_tile = k[:, :, kv_start:kv_end, :]
            v_tile = v[:, :, kv_start:kv_end, :]

            s = (torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale).float()
            if kv_end > q_start:
                # Boundary tile: apply causal mask j <= i
                kv_pos = torch.arange(kv_start, kv_end, device=q.device)[None, :]
                valid = kv_pos <= q_pos
                s = s.masked_fill(~valid[None, None, :, :], -float("inf"))

            tile_max = s.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, tile_max)
            # Avoid exp(-inf - -inf) = NaN when tile is completely masked (not possible for causal q >= 0, but safe)
            alpha = torch.where(m > -float("inf"), torch.exp(m - m_new), torch.zeros_like(m))
            p_tile = torch.where(m_new > -float("inf"), torch.exp(s - m_new), torch.zeros_like(s))

            ell = alpha * ell + p_tile.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p_tile, v_tile.float())
            m = m_new

        out_tile = (acc / ell.clamp(min=1e-12)).to(q.dtype)
        out_chunks.append(out_tile)

    return torch.cat(out_chunks, dim=-2)


def probe_sdpa_backends(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, Any]:
    """Probe which PyTorch SDPA backends support the given tensors and which one dispatches."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    candidates = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("MATH", SDPBackend.MATH),
    ]
    supported: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for name, backend in candidates:
        if backend is None:
            continue
        try:
            with sdpa_kernel([backend]):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            supported[name] = True
        except Exception as exc:
            supported[name] = False
            errors[name] = f"{type(exc).__name__}: {exc}"

    # Check default dispatch order used by PyTorch on this device/dtype
    dispatched = "UNKNOWN"
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        if supported.get(name, False):
            dispatched = name
            break
    return {"supported": supported, "errors": errors, "inferred_default_backend": dispatched}


def verify_causality_and_edge_cases(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Run causal perturbation check, extreme-score stability, and all-masked edge case."""
    gen = torch.Generator(device="cpu").manual_seed(6304)
    B, H, L, d_h = 2, 4, 32, 64
    q = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(device)
    k = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(device)
    v = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(device)

    out_ref = dense_causal_attention(q, k, v)
    t_split = 16
    k_pert = k.clone()
    v_pert = v.clone()
    k_pert[:, :, t_split:, :] += 50.0
    v_pert[:, :, t_split:, :] -= 50.0
    out_pert = dense_causal_attention(q, k_pert, v_pert)
    prefix_diff = float((out_ref[:, :, :t_split, :] - out_pert[:, :, :t_split, :]).abs().max().item())

    # Extreme score stability test
    q_ext = q * 20.0
    k_ext = k * 20.0
    out_ext_dense = dense_causal_attention(q_ext, k_ext, v)
    out_ext_tiled = tiled_online_softmax_causal_attention(q_ext, k_ext, v, block_q=8, block_kv=8)
    ext_diff = float((out_ext_dense - out_ext_tiled).abs().max().item())

    # All-masked first row edge case
    all_false_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
    all_false_mask[1:, 0] = True
    out_masked = dense_causal_attention(q, k, v, attn_mask=all_false_mask)

    return {
        "causality_prefix_max_abs_diff": prefix_diff,
        "extreme_score_finite": bool(torch.isfinite(out_ext_dense).all() and torch.isfinite(out_ext_tiled).all()),
        "extreme_score_dense_vs_tiled_max_abs": ext_diff,
        "all_masked_row0_max_abs": float(out_masked[:, :, 0, :].abs().max().item()),
        "all_masked_finite": bool(torch.isfinite(out_masked).all()),
    }


def benchmark_attention_suite(
    lengths: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192),
    dtypes: tuple[str, ...] = ("fp16", "fp32"),
    B: int = 2,
    H: int = 4,
    d_h: int = 64,
    repeats: int = 10,
    warmup: int = 3,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Compare dense, tiled (64x64), and PyTorch SDPA on accuracy, gradients, latency, and VRAM."""
    dev = torch.device(device)
    results: list[dict[str, Any]] = []
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}

    for L in lengths:
        # Build FP32 reference first (unless infeasible due to OOM)
        gen = torch.Generator(device="cpu").manual_seed(6304 + L)
        q32 = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(dev)
        k32 = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(dev)
        v32 = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(dev)
        g32 = torch.randn(B, H, L, d_h, generator=gen, dtype=torch.float32).to(dev)

        ref_out32 = None
        ref_grads32 = None
        try:
            qr = q32.clone().requires_grad_(True)
            kr = k32.clone().requires_grad_(True)
            vr = v32.clone().requires_grad_(True)
            ref_out32 = dense_causal_attention(qr, kr, vr)
            (ref_out32 * g32).sum().backward()
            ref_grads32 = (qr.grad.detach().clone(), kr.grad.detach().clone(), vr.grad.detach().clone())
            ref_out32 = ref_out32.detach().clone()
            del qr, kr, vr
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            ref_out32 = None
            ref_grads32 = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        for dt_name in dtypes:
            dt = dtype_map[dt_name]
            q_base = q32.to(dt)
            k_base = k32.to(dt)
            v_base = v32.to(dt)
            g_base = g32.to(dt)

            sdpa_probe = probe_sdpa_backends(q_base[:1, :, :min(L, 128), :], k_base[:1, :, :min(L, 128), :], v_base[:1, :, :min(L, 128), :])

            ops = [
                ("dense", lambda q, k, v: dense_causal_attention(q, k, v)),
                ("tiled_64", lambda q, k, v: tiled_online_softmax_causal_attention(q, k, v, 64, 64)),
                ("sdpa", lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
            ]

            for op_name, op_fn in ops:
                entry: dict[str, Any] = {
                    "L": L,
                    "dtype": dt_name,
                    "operator": op_name,
                    "B": B,
                    "H": H,
                    "d_h": d_h,
                    "sdpa_probe": sdpa_probe if op_name == "sdpa" else None,
                    "fwd_status": "ok",
                    "fwdbwd_status": "ok",
                }
                # 1. Forward-only accuracy, memory, and timing
                try:
                    with torch.no_grad():
                        out_f = op_fn(q_base, k_base, v_base)
                        if ref_out32 is not None:
                            out_err = (out_f.float() - ref_out32).abs()
                            entry["out_max_abs_err"] = float(out_err.max().item())
                            entry["out_mean_abs_err"] = float(out_err.mean().item())
                        del out_f
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

                    def fwd_only():
                        with torch.no_grad():
                            return op_fn(q_base, k_base, v_base)

                    _, fwd_mem = reset_and_measure_peak_memory_bytes(fwd_only, dev)
                    fwd_lat = measure_cuda_latency_ms(fwd_only, warmup=warmup, repeats=repeats, device=dev)
                    entry["fwd_latency"] = fwd_lat
                    entry["fwd_memory"] = fwd_mem
                except torch.cuda.OutOfMemoryError as exc:
                    entry["fwd_status"] = "OOM"
                    entry["fwd_error"] = str(exc)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

                # 2. Forward+backward gradient accuracy, memory, and timing
                try:
                    qt = q_base.detach().clone().requires_grad_(True)
                    kt = k_base.detach().clone().requires_grad_(True)
                    vt = v_base.detach().clone().requires_grad_(True)
                    out_b = op_fn(qt, kt, vt)
                    (out_b * g_base).sum().backward()
                    if ref_grads32 is not None:
                        dq_err = (qt.grad.detach().float() - ref_grads32[0]).abs()
                        dk_err = (kt.grad.detach().float() - ref_grads32[1]).abs()
                        dv_err = (vt.grad.detach().float() - ref_grads32[2]).abs()
                        entry["dq_max_abs_err"] = float(dq_err.max().item())
                        entry["dk_max_abs_err"] = float(dk_err.max().item())
                        entry["dv_max_abs_err"] = float(dv_err.max().item())
                    del qt, kt, vt, out_b
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

                    def fwd_bwd():
                        qf = q_base.detach().clone().requires_grad_(True)
                        kf = k_base.detach().clone().requires_grad_(True)
                        vf = v_base.detach().clone().requires_grad_(True)
                        o = op_fn(qf, kf, vf)
                        (o * g_base).sum().backward()
                        return qf.grad

                    _, bwd_mem = reset_and_measure_peak_memory_bytes(fwd_bwd, dev)
                    bwd_lat = measure_cuda_latency_ms(fwd_bwd, warmup=warmup, repeats=repeats, device=dev)
                    entry["fwdbwd_latency"] = bwd_lat
                    entry["fwdbwd_memory"] = bwd_mem
                except torch.cuda.OutOfMemoryError as exc:
                    entry["fwdbwd_status"] = "OOM"
                    entry["fwdbwd_error"] = str(exc)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

                results.append(entry)

        del q32, k32, v32, g32, ref_out32, ref_grads32
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # Also check tile size sensitivity at L=512 (FP16 & FP32)
    tile_checks = []
    gen_t = torch.Generator(device="cpu").manual_seed(6304)
    q_t = torch.randn(2, 4, 512, 64, generator=gen_t, dtype=torch.float32).to(dev)
    k_t = torch.randn(2, 4, 512, 64, generator=gen_t, dtype=torch.float32).to(dev)
    v_t = torch.randn(2, 4, 512, 64, generator=gen_t, dtype=torch.float32).to(dev)
    ref_t = dense_causal_attention(q_t, k_t, v_t)
    for dt_name, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        for bq, bkv in ((32, 32), (64, 64), (128, 128), (64, 128)):
            o_t = tiled_online_softmax_causal_attention(q_t.to(dt), k_t.to(dt), v_t.to(dt), bq, bkv)
            diff = (o_t.float() - ref_t).abs()
            tile_checks.append({
                "dtype": dt_name,
                "block_q": bq,
                "block_kv": bkv,
                "max_abs_err": float(diff.max().item()),
                "mean_abs_err": float(diff.mean().item()),
            })

    return {
        "causality_and_edge_cases": verify_causality_and_edge_cases(dev),
        "tile_checks_L512": tile_checks,
        "benchmarks": results,
    }
