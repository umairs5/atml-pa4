"""Pruning methods (P1 Magnitude, P2 Wanda, 2:4 Wanda, P3 Nonuniform) and sensitivity probes."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task1.model_utils import EligibleProjection, enumerate_eligible_projections, get_transformer_layers


def load_sequence_manifest(path: str | Path) -> list[list[int]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["token_ids"] for line in f if line.strip()]


@torch.inference_mode()
def collect_calibration_stats(
    model: nn.Module,
    sequences: list[list[int]],
    device: str | torch.device = "cuda",
    batch_size: int = 4,
    collect_block_bi: bool = True,
) -> dict[str, Any]:
    """Collect per-projection input activation L2 norms, mean abs activations,
    and optional ShortGPT Block Influence (BI) scores in one calibration pass.
    """
    model.eval()
    projections = enumerate_eligible_projections(model)
    layers = get_transformer_layers(model)

    sum_sq: dict[str, torch.Tensor] = {
        p.name: torch.zeros(p.d_in, dtype=torch.float64, device=device) for p in projections
    }
    sum_abs: dict[str, torch.Tensor] = {
        p.name: torch.zeros(p.d_in, dtype=torch.float64, device=device) for p in projections
    }
    tok_counts: dict[str, int] = {p.name: 0 for p in projections}
    bi_sums: list[float] = [0.0] * len(layers)
    bi_counts: list[int] = [0] * len(layers)

    handles = []

    def make_proj_hook(pname: str):
        def hook(_mod, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float64)
            sum_sq[pname] += (x * x).sum(dim=0)
            sum_abs[pname] += x.abs().sum(dim=0)
            tok_counts[pname] += int(x.shape[0])

        return hook

    for p in projections:
        handles.append(p.module.register_forward_hook(make_proj_hook(p.name)))

    if collect_block_bi:
        def make_block_hook(bidx: int):
            def hook(_mod, inp, out):
                x_in = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                x_out = (out[0] if isinstance(out, tuple) else out).detach().reshape(-1, x_in.shape[-1]).float()
                cos = F.cosine_similarity(x_in, x_out, dim=-1)
                bi_sums[bidx] += float((1.0 - cos).sum().item())
                bi_counts[bidx] += int(cos.numel())

            return hook

        for bidx, block in enumerate(layers):
            handles.append(block.register_forward_hook(make_block_hook(bidx)))

    try:
        for start in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
            model(input_ids=batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    input_l2 = {k: torch.sqrt(v).cpu().to(torch.float32) for k, v in sum_sq.items()}
    input_mean_abs = {k: (v / max(tok_counts[k], 1)).cpu().to(torch.float32) for k, v in sum_abs.items()}
    block_bi = [bi_sums[i] / max(bi_counts[i], 1) for i in range(len(layers))]

    return {
        "input_l2": input_l2,
        "input_mean_abs": input_mean_abs,
        "token_counts": tok_counts,
        "block_bi": block_bi,
    }


def collect_output_fisher_scales(
    model: nn.Module,
    sequences: list[list[int]],
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Compute mean squared output-gradient (diagonal output Fisher proxy) per projection
    using microbatch=1 with gradient checkpointing over the calibration set.
    """
    model.eval()
    projections = enumerate_eligible_projections(model)
    grad_sq_sum: dict[str, float] = {p.name: 0.0 for p in projections}
    grad_elem_cnt: dict[str, int] = {p.name: 0 for p in projections}
    taylor_sum: dict[str, float] = {p.name: 0.0 for p in projections}

    # Enable gradient checkpointing on transformer if supported to keep VRAM tiny
    had_gc = getattr(model, "is_gradient_checkpointing", False)
    if hasattr(model, "gradient_checkpointing_enable") and not had_gc:
        model.gradient_checkpointing_enable()

    # Keep only eligible weights requiring grad? Actually we only need hooks on module outputs!
    # We must ensure at least input embeddings or one tensor requires grad so autograd traverses blocks;
    # keeping eligible weights requires_grad=True is easy if we zero_grad each step and avoid optimizer states.
    prev_req = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        prev_req[name] = p.requires_grad
        p.requires_grad_(False)
    # Keep input embeddings requiring grad so gradient checkpointing + full backward hooks fire cleanly
    inp_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if inp_emb is not None and hasattr(inp_emb, "weight"):
        inp_emb.weight.requires_grad_(True)
    for proj in projections:
        proj.module.weight.requires_grad_(True)

    handles = []

    def make_bwd_hook(pname: str):
        def bwd_hook(_mod, _grad_in, grad_out):
            g = grad_out[0].detach().float()
            grad_sq_sum[pname] += float((g * g).sum().item())
            grad_elem_cnt[pname] += int(g.numel())

        return bwd_hook

    for proj in projections:
        handles.append(proj.module.register_full_backward_hook(make_bwd_hook(proj.name)))

    try:
        for seq in sequences:
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(input_ids=ids, use_cache=False).logits
            shift_logits = logits[:, :-1, :].float()
            targets = ids[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            for proj in projections:
                if proj.module.weight.grad is not None:
                    taylor_sum[proj.name] += float(
                        (proj.module.weight.data.float() * proj.module.weight.grad.detach().float()).abs().sum().item()
                    )
                    proj.module.weight.grad = None
    finally:
        for h in handles:
            h.remove()
        for name, p in model.named_parameters(remove_duplicate=False):
            p.requires_grad_(prev_req[name])
        if hasattr(model, "gradient_checkpointing_disable") and not had_gc:
            model.gradient_checkpointing_disable()

    return {
        "output_grad_msq": {k: grad_sq_sum[k] / max(grad_elem_cnt[k], 1) for k in grad_sq_sum},
        "weight_taylor_sum": {k: taylor_sum[k] / max(len(sequences), 1) for k in taylor_sum},
    }


def compute_pruning_scores(
    weight: torch.Tensor,
    method: str,
    input_l2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if method == "magnitude":
        return weight.detach().abs().float()
    if method in ("wanda", "wanda_2_4"):
        if input_l2 is None:
            raise ValueError("input_l2 is required for Wanda pruning scores")
        return weight.detach().abs().float() * input_l2.to(weight.device).float().unsqueeze(0)
    raise ValueError(f"Unknown pruning method: {method}")


def make_per_row_mask(scores: torch.Tensor, k_prune_per_row: int) -> torch.Tensor:
    """Return boolean keep-mask (True=keep, False=prune) pruning exact `k_prune_per_row` smallest per row."""
    d_out, d_in = scores.shape
    if k_prune_per_row <= 0:
        return torch.ones_like(scores, dtype=torch.bool)
    if k_prune_per_row >= d_in:
        return torch.zeros_like(scores, dtype=torch.bool)
    # Sort ascending per row and drop lowest k_prune_per_row
    _, idx = torch.topk(scores, k=k_prune_per_row, dim=1, largest=False, sorted=False)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    return mask


def make_2_4_mask(scores: torch.Tensor) -> torch.Tensor:
    """Return 2:4 structured keep-mask keeping top 2 weights in each contiguous group of 4 input channels."""
    d_out, d_in = scores.shape
    if d_in % 4 != 0:
        raise ValueError(f"d_in={d_in} must be divisible by 4 for 2:4 sparsity")
    grouped = scores.reshape(d_out, d_in // 4, 4)
    _, top2_idx = torch.topk(grouped, k=2, dim=-1, largest=True, sorted=False)
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, top2_idx, True)
    return mask.reshape(d_out, d_in)


@torch.inference_mode()
def apply_pruning_to_model(
    model: nn.Module,
    method: str,
    sparsity_map: float | dict[str, float],
    input_l2: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, Any]:
    """Apply in-place pruning to eligible projections and return per-projection diagnostics."""
    projections = enumerate_eligible_projections(model)
    records = []
    for proj in projections:
        w = proj.module.weight.data
        l2 = None if input_l2 is None else input_l2[proj.name]
        scores = compute_pruning_scores(w, method=method, input_l2=l2)
        if method == "wanda_2_4":
            mask = make_2_4_mask(scores)
            target_s = 0.5
        else:
            target_s = float(sparsity_map[proj.name] if isinstance(sparsity_map, dict) else sparsity_map)
            k_prune = int(round(target_s * proj.d_in))
            mask = make_per_row_mask(scores, k_prune)
        w.mul_(mask.to(w.dtype))
        nnz = int(torch.count_nonzero(w).item())
        records.append({
            "name": proj.name,
            "target_sparsity": target_s,
            "achieved_sparsity": 1.0 - (nnz / proj.numel),
            "nnz": nnz,
            "numel": proj.numel,
        })
    return {"method": method, "projections": records}


@torch.inference_mode()
def measure_layer_output_errors(
    ref_model: nn.Module,
    comp_model: nn.Module,
    sequences: list[list[int]],
    device: str | torch.device = "cuda",
    batch_size: int = 4,
    max_sequences: int = 32,
) -> dict[str, Any]:
    """Measure isolated per-projection output MSE/rel-L2 (on clean FP16 inputs)
    and cumulative per-block hidden-state MSE/rel-L2 on calibration sequences.
    """
    ref_model.eval()
    comp_model.eval()
    seqs = sequences[:max_sequences]
    ref_projs = {p.name: p.module for p in enumerate_eligible_projections(ref_model)}
    comp_projs = {p.name: p.module for p in enumerate_eligible_projections(comp_model)}
    ref_layers = get_transformer_layers(ref_model)
    comp_layers = get_transformer_layers(comp_model)

    # 1. Capture reference block inputs/outputs and isolated per-projection errors
    iso_err_sq = {name: 0.0 for name in ref_projs}
    iso_ref_sq = {name: 0.0 for name in ref_projs}
    cum_block_err_sq = [0.0] * len(ref_layers)
    cum_block_ref_sq = [0.0] * len(ref_layers)

    ref_block_outs: list[list[torch.Tensor]] = [[] for _ in range(len(ref_layers))]
    comp_block_outs: list[list[torch.Tensor]] = [[] for _ in range(len(comp_layers))]

    handles = []

    def make_ref_proj_hook(pname: str, comp_mod: nn.Linear):
        def hook(_mod, inp, out):
            x = inp[0].detach()
            y_ref = out.detach().float()
            # Evaluate compressed projection on IDENTICAL clean FP16 input
            y_comp = comp_mod(x.to(comp_mod.weight.device)).detach().float().to(y_ref.device)
            diff = y_ref - y_comp
            iso_err_sq[pname] += float((diff * diff).sum().item())
            iso_ref_sq[pname] += float((y_ref * y_ref).sum().item())

        return hook

    for pname, mod in ref_projs.items():
        handles.append(mod.register_forward_hook(make_ref_proj_hook(pname, comp_projs[pname])))

    def make_block_save_hook(store: list[list[torch.Tensor]], bidx: int):
        def hook(_mod, _inp, out):
            t = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            store[bidx].append(t)

        return hook

    for bidx, blk in enumerate(ref_layers):
        handles.append(blk.register_forward_hook(make_block_save_hook(ref_block_outs, bidx)))

    try:
        for start in range(0, len(seqs), batch_size):
            batch = torch.tensor(seqs[start : start + batch_size], dtype=torch.long, device=device)
            ref_model(input_ids=batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    comp_handles = []
    for bidx, blk in enumerate(comp_layers):
        comp_handles.append(blk.register_forward_hook(make_block_save_hook(comp_block_outs, bidx)))

    try:
        for start in range(0, len(seqs), batch_size):
            batch = torch.tensor(seqs[start : start + batch_size], dtype=torch.long, device=device)
            comp_model(input_ids=batch, use_cache=False)
    finally:
        for h in comp_handles:
            h.remove()

    for bidx in range(len(ref_layers)):
        for r_t, c_t in zip(ref_block_outs[bidx], comp_block_outs[bidx]):
            rf = r_t.float()
            cf = c_t.float()
            d = rf - cf
            cum_block_err_sq[bidx] += float((d * d).sum().item())
            cum_block_ref_sq[bidx] += float((rf * rf).sum().item())

    per_proj = {
        name: {
            "rel_l2": math.sqrt(iso_err_sq[name] / max(iso_ref_sq[name], 1e-12)),
            "rel_mse": iso_err_sq[name] / max(iso_ref_sq[name], 1e-12),
        }
        for name in ref_projs
    }
    per_block = [
        {
            "layer_idx": i,
            "cum_rel_l2": math.sqrt(cum_block_err_sq[i] / max(cum_block_ref_sq[i], 1e-12)),
            "cum_rel_mse": cum_block_err_sq[i] / max(cum_block_ref_sq[i], 1e-12),
        }
        for i in range(len(ref_layers))
    ]
    return {
        "mean_proj_rel_l2": float(np.mean([v["rel_l2"] for v in per_proj.values()])),
        "final_block_cum_rel_l2": per_block[-1]["cum_rel_l2"],
        "per_projection": per_proj,
        "per_block": per_block,
    }


# ---------------------------------------------------------------------------
# P3 Nonuniform Pruning Sensitivity & Allocation Rules
# ---------------------------------------------------------------------------

def allocate_owl_sparsities(
    projections: list[EligibleProjection],
    input_l2: dict[str, torch.Tensor],
    global_sparsity: float,
    outlier_m: float = 5.0,
    lam: float = 0.08,
    s_min: float = 0.10,
    s_max: float = 0.85,
    unit: str = "block",
) -> tuple[dict[str, float], dict[str, float]]:
    """OWL (Yin et al. 2024) outlier-weighted layerwise sparsity allocation."""
    # Compute outlier ratio per projection
    proj_lod: dict[str, float] = {}
    for p in projections:
        scores = compute_pruning_scores(p.module.weight.data, "wanda", input_l2[p.name])
        thresh = outlier_m * scores.mean()
        proj_lod[p.name] = float((scores > thresh).float().mean().item())

    if unit == "block":
        n_layers = max(p.layer_idx for p in projections) + 1
        unit_sens: dict[str, float] = {}
        for l in range(n_layers):
            sub = [p for p in projections if p.layer_idx == l]
            tot_w = sum(p.numel for p in sub)
            unit_sens[f"block_{l}"] = sum(proj_lod[p.name] * p.numel for p in sub) / tot_w
    elif unit == "projection":
        unit_sens = {p.name: proj_lod[p.name] for p in projections}
    else:
        raise ValueError(f"Unknown unit: {unit}")

    vals = np.array(list(unit_sens.values()), dtype=np.float64)
    # Higher outlier ratio -> MORE sensitive -> LOWER sparsity
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        raw_s = {k: global_sparsity for k in unit_sens}
    else:
        normed = (vals - vmin) / (vmax - vmin)  # [0, 1]
        # Map [0, 1] sensitivity to [s_target + lam, s_target - lam]
        mapped = global_sparsity + lam * (1.0 - 2.0 * normed)
        raw_s = {k: float(np.clip(v, s_min, s_max)) for k, v in zip(unit_sens.keys(), mapped)}

    proj_s = _expand_and_match_sparsity_budget(
        projections, raw_s, unit=unit, global_sparsity=global_sparsity, s_min=s_min, s_max=s_max
    )
    return proj_s, unit_sens


def allocate_shortgpt_bi_sparsities(
    projections: list[EligibleProjection],
    block_bi: list[float],
    global_sparsity: float,
    lam: float = 0.15,
    s_min: float = 0.10,
    s_max: float = 0.85,
) -> tuple[dict[str, float], dict[str, float]]:
    """ShortGPT Block Influence (Men et al. 2025): higher BI = more influential = lower sparsity."""
    unit_sens = {f"block_{i}": float(v) for i, v in enumerate(block_bi)}
    vals = np.array(block_bi, dtype=np.float64)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        raw_s = {k: global_sparsity for k in unit_sens}
    else:
        normed = (vals - vmin) / (vmax - vmin)
        mapped = global_sparsity + lam * (1.0 - 2.0 * normed)
        raw_s = {f"block_{i}": float(np.clip(v, s_min, s_max)) for i, v in enumerate(mapped)}
    proj_s = _expand_and_match_sparsity_budget(
        projections, raw_s, unit="block", global_sparsity=global_sparsity, s_min=s_min, s_max=s_max
    )
    return proj_s, unit_sens


def allocate_marginal_distortion_sparsities(
    projections: list[EligibleProjection],
    input_l2: dict[str, torch.Tensor],
    sensitivity_weights: dict[str, float],
    global_sparsity: float,
    s_min: float = 0.10,
    s_max: float = 0.85,
    step: float = 0.05,
    normalize_by_tot_sq: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Constrained water-filling / greedy marginal distortion allocation.

    Estimates D_u(s) = sensitivity_weight_u * pruned_energy_u(s) [/ tot_energy_u]
    over discrete sparsity steps and greedily assigns each increment to the projection with
    smallest marginal loss per pruned weight.
    """
    grid = np.round(np.arange(s_min, s_max + 1e-9, step), 4)
    costs: dict[str, list[float]] = {}
    unit_sens: dict[str, float] = {}

    for p in projections:
        scores = compute_pruning_scores(p.module.weight.data, "wanda", input_l2[p.name])
        sq = (scores * scores).reshape(-1)
        tot_sq = float(sq.sum().item()) + 1e-12
        # Sort each row ascending to evaluate exact per-row Wanda pruned score energy on grid
        row_sq_sorted, _ = torch.sort(scores * scores, dim=1, descending=False)
        cum_row = torch.cumsum(row_sq_sorted, dim=1)
        w_sens = float(sensitivity_weights[p.name])
        denom = tot_sq if normalize_by_tot_sq else 1.0
        curve = []
        for s in grid:
            k = int(round(float(s) * p.d_in))
            pruned_sq = float(cum_row[:, k - 1].sum().item()) if k > 0 else 0.0
            curve.append(w_sens * (pruned_sq / denom))
        costs[p.name] = curve
        unit_sens[p.name] = w_sens

    # Greedy water-filling starting from s_min for every projection
    idx_map = {p.name: 0 for p in projections}
    total_elig = sum(p.numel for p in projections)
    target_zeros = int(round(global_sparsity * total_elig))
    cur_zeros = sum(p.d_out * int(round(float(grid[0]) * p.d_in)) for p in projections)

    import heapq

    heap: list[tuple[float, str, int]] = []
    for p in projections:
        if len(grid) > 1:
            dk = int(round(float(grid[1]) * p.d_in)) - int(round(float(grid[0]) * p.d_in))
            dw = p.d_out * dk
            dc = costs[p.name][1] - costs[p.name][0]
            heapq.heappush(heap, (dc / max(dw, 1), p.name, 0))

    while cur_zeros < target_zeros and heap:
        _, pname, i_cur = heapq.heappop(heap)
        if idx_map[pname] != i_cur or i_cur + 1 >= len(grid):
            continue
        p = next(x for x in projections if x.name == pname)
        k_prev = int(round(float(grid[i_cur]) * p.d_in))
        k_next = int(round(float(grid[i_cur + 1]) * p.d_in))
        dw = p.d_out * (k_next - k_prev)
        if cur_zeros + dw > target_zeros and (target_zeros - cur_zeros) < dw * 0.5:
            continue
        idx_map[pname] = i_cur + 1
        cur_zeros += dw
        if i_cur + 2 < len(grid):
            k2 = int(round(float(grid[i_cur + 2]) * p.d_in))
            dw2 = p.d_out * (k2 - k_next)
            dc2 = costs[pname][i_cur + 2] - costs[pname][i_cur + 1]
            heapq.heappush(heap, (dc2 / max(dw2, 1), pname, i_cur + 1))

    raw_s = {pname: float(grid[i]) for pname, i in idx_map.items()}
    matched_s = _expand_and_match_sparsity_budget(
        projections, raw_s, unit="projection", global_sparsity=global_sparsity, s_min=s_min, s_max=s_max
    )
    return matched_s, unit_sens


def _expand_and_match_sparsity_budget(
    projections: list[EligibleProjection],
    unit_sparsity: dict[str, float],
    unit: str,
    global_sparsity: float,
    s_min: float = 0.10,
    s_max: float = 0.85,
) -> dict[str, float]:
    """Rescale and integer-round per-row sparsities so total pruned weights match global_sparsity."""
    if unit == "block":
        s_map = {p.name: float(unit_sparsity[f"block_{p.layer_idx}"]) for p in projections}
    elif unit == "projection":
        s_map = {p.name: float(unit_sparsity[p.name]) for p in projections}
    else:
        raise ValueError(f"Unknown unit: {unit}")

    total_elig = sum(p.numel for p in projections)
    target_zeros = int(round(global_sparsity * total_elig))

    # Binary search scalar shift delta added before clamping to [s_min, s_max]
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        est = sum(p.d_out * int(round(np.clip(s_map[p.name] + mid, s_min, s_max) * p.d_in)) for p in projections)
        if est < target_zeros:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)

    k_map: dict[str, int] = {}
    for p in projections:
        s_val = float(np.clip(s_map[p.name] + delta, s_min, s_max))
        k_lo = math.ceil(s_min * p.d_in - 1e-9)
        k_hi = math.floor(s_max * p.d_in + 1e-9)
        k_map[p.name] = int(np.clip(round(s_val * p.d_in), k_lo, k_hi))

    # Fine-tune per-row counts by +-1 starting with smallest d_out matrices (e.g. k_proj/v_proj = 256)
    cur_zeros = sum(p.d_out * k_map[p.name] for p in projections)
    diff = target_zeros - cur_zeros
    sorted_projs = sorted(projections, key=lambda p: p.d_out)
    while abs(diff) >= min(p.d_out for p in projections) // 2:
        changed = False
        for p in sorted_projs:
            k_lo = math.ceil(s_min * p.d_in - 1e-9)
            k_hi = math.floor(s_max * p.d_in + 1e-9)
            if diff > 0 and k_map[p.name] + 1 <= k_hi and abs(diff - p.d_out) < abs(diff):
                k_map[p.name] += 1
                diff -= p.d_out
                changed = True
            elif diff < 0 and k_map[p.name] - 1 >= k_lo and abs(diff + p.d_out) < abs(diff):
                k_map[p.name] -= 1
                diff += p.d_out
                changed = True
            if abs(diff) < min(x.d_out for x in projections) // 2:
                break
        if not changed:
            break

    return {p.name: k_map[p.name] / p.d_in for p in projections}
