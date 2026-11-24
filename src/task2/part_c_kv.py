"""Task 2 Part C: Pretrained KV-cache validation, eviction, quantization, hybrid, and packed benchmarks."""

from __future__ import annotations

import gzip
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.common.env import measure_cuda_latency_ms, reset_and_measure_peak_memory_bytes
from src.task1.packing import pack_bits, unpack_bits

QUESTION_MARKER = "\nQUESTION_BEGIN\n"
SINK_TOKENS = 8
RECENT_TOKENS = 64
KV_GROUP_SIZE = 64


def load_kv_retrieval_cases(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_five_digit_answer(text: str) -> Optional[str]:
    s = text.strip()
    if re.fullmatch(r"\d{5}", s):
        return s
    m = re.search(r"(?<!\d)(\d{5})(?!\d)", s)
    return m.group(1) if m else None


def calc_kv_bytes_formula(
    L: int,
    bits: int,
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    group_size: int = KV_GROUP_SIZE,
) -> dict[str, int]:
    """Exact PA4 Task 2C byte formulas M_b(L) and M_16(L)."""
    n_chan = 2 * n_layers * n_kv_heads * head_dim
    n_groups = L // group_size
    rem = L % group_size
    group_bytes = (group_size * bits) // 8 + 4  # 4 metadata bytes: FP16 scale + FP16 offset
    m_b = n_chan * (n_groups * group_bytes + 2 * rem)
    m_16 = 4 * n_layers * n_kv_heads * head_dim * L
    return {
        "M_b_bytes": int(m_b),
        "M_16_bytes": int(m_16),
        "n_complete_groups": int(n_groups),
        "trailing_fp16_tokens": int(rem),
        "fp16_bytes_per_pos": int(4 * n_layers * n_kv_heads * head_dim),
    }


def tokenize_kv_case_split(tokenizer, case: dict[str, Any]) -> dict[str, Any]:
    """Apply chat template and find exact token split before QUESTION_BEGIN + gold evidence token span."""
    user_text = case["user_text"]
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    q_pos_in_full = full_text.index(QUESTION_MARKER)
    user_pos_in_full = full_text.index(user_text)
    ev_c0 = user_pos_in_full + int(case["evidence_char_start"])
    ev_c1 = user_pos_in_full + int(case["evidence_char_end"])

    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]

    # First token extending past q_pos_in_full starts the question suffix
    split_tok = None
    for idx, (c0, c1) in enumerate(offsets):
        if c1 > q_pos_in_full:
            split_tok = idx
            break
    if split_tok is None or split_tok < SINK_TOKENS + RECENT_TOKENS:
        raise ValueError(f"Invalid split_tok={split_tok} for case {case['id']}")

    evidence_tok_indices = [
        idx for idx, (c0, c1) in enumerate(offsets[:split_tok]) if c1 > ev_c0 and c0 < ev_c1
    ]
    return {
        "full_ids": full_ids,
        "ctx_ids": full_ids[:split_tok],
        "q_ids": full_ids[split_tok:],
        "L_ctx": split_tok,
        "L_q": len(full_ids) - split_tok,
        "L_total": len(full_ids),
        "evidence_token_indices": evidence_tok_indices,
    }


# ---------------------------------------------------------------------------
# Group-wise KV quantization (FP16 scale + FP16 offset, group size 64)
# ---------------------------------------------------------------------------

def quantize_pack_group64(x: torch.Tensor, bits: int, axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Quantize and bit-pack along `axis` (must be divisible by 64).

    Convention: offset = x_min (FP16), scale = max(x_max - x_min, 1e-6) / (2^b - 1) (FP16),
    q = clip(round((x - offset) / scale), 0, 2^b - 1).
    """
    if axis < 0:
        axis = x.ndim + axis
    length = x.shape[axis]
    if length % KV_GROUP_SIZE != 0:
        raise ValueError(f"Axis {axis} length {length} must be divisible by {KV_GROUP_SIZE}")
    if length == 0:
        empty_u8 = torch.empty(0, dtype=torch.uint8, device=x.device)
        empty_f16 = torch.empty(0, dtype=torch.float16, device=x.device)
        return empty_u8, empty_f16, empty_f16, tuple(x.shape)

    perm = [i for i in range(x.ndim) if i != axis] + [axis]
    inv_perm = [0] * x.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    xp = x.permute(perm).contiguous()
    lead_shape = xp.shape[:-1]
    n_groups = length // KV_GROUP_SIZE
    xg = xp.reshape(*lead_shape, n_groups, KV_GROUP_SIZE).float()

    x_min = xg.amin(dim=-1, keepdim=True)
    x_max = xg.amax(dim=-1, keepdim=True)
    qmax = (1 << bits) - 1
    offset_f16 = x_min.squeeze(-1).to(torch.float16)
    offset_eff = offset_f16.float().unsqueeze(-1)
    scale_f16 = ((x_max - offset_eff) / qmax).clamp(min=1e-6).squeeze(-1).to(torch.float16)
    scale_eff = scale_f16.float().unsqueeze(-1).clamp(min=1e-6)

    q = torch.clamp(torch.round((xg - offset_eff) / scale_eff), 0, qmax).to(torch.uint8)
    q_packed = pack_bits(q, bits)
    return q_packed, scale_f16, offset_f16, tuple(xp.shape)


def unpack_dequantize_group64(
    q_packed: torch.Tensor,
    scale_f16: torch.Tensor,
    offset_f16: torch.Tensor,
    permuted_shape: tuple[int, ...],
    bits: int,
    axis: int,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if q_packed.numel() == 0:
        if axis < 0:
            axis = len(permuted_shape) + axis
        orig_shape = list(permuted_shape[:-1])
        orig_shape.insert(axis, 0)
        return torch.empty(orig_shape, dtype=out_dtype, device=q_packed.device)

    n_elems = 1
    for s in permuted_shape:
        n_elems *= s
    q = unpack_bits(q_packed, bits, n_elems).reshape(*permuted_shape[:-1], permuted_shape[-1] // KV_GROUP_SIZE, KV_GROUP_SIZE).float()
    s = scale_f16.float().unsqueeze(-1)
    o = offset_f16.float().unsqueeze(-1)
    xg = s * q + o
    xp = xg.reshape(*permuted_shape).to(out_dtype)

    if axis < 0:
        axis = len(permuted_shape) + axis
    # Invert permutation that moved `axis` to the end
    ndim = len(permuted_shape)
    inv_perm = list(range(ndim - 1))
    inv_perm.insert(axis, ndim - 1)
    return xp.permute(inv_perm).contiguous()


@dataclass
class QuantizedLayerKV:
    k_mode: str  # "channel" or "token"
    bits: int
    # Packed complete-group buffers
    k_packed: torch.Tensor
    k_scale: torch.Tensor
    k_offset: torch.Tensor
    k_perm_shape: tuple[int, ...]
    v_packed: torch.Tensor
    v_scale: torch.Tensor
    v_offset: torch.Tensor
    v_perm_shape: tuple[int, ...]
    # Trailing FP16 residual tokens (length < 64)
    k_res: torch.Tensor  # [B, H_kv, r, d_h]
    v_res: torch.Tensor  # [B, H_kv, r, d_h]

    def packed_bytes(self) -> int:
        total = 0
        for t in (self.k_packed, self.k_scale, self.k_offset, self.v_packed, self.v_scale, self.v_offset, self.k_res, self.v_res):
            total += int(t.numel() * t.element_size())
        return total


def compress_kv_tensor_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int,
    layout: str,
) -> QuantizedLayerKV:
    """Compress [B, H_kv, L, d_h] into QuantizedLayerKV using trailing L % 64 FP16 rule."""
    L = k.shape[-2]
    n_comp = (L // KV_GROUP_SIZE) * KV_GROUP_SIZE
    k_main, k_res = k[:, :, :n_comp, :].contiguous(), k[:, :, n_comp:, :].contiguous()
    v_main, v_res = v[:, :, :n_comp, :].contiguous(), v[:, :, n_comp:, :].contiguous()

    k_axis = -2 if layout == "kivi" else -1
    kp, ks, ko, kps = quantize_pack_group64(k_main, bits, axis=k_axis)
    vp, vs, vo, vps = quantize_pack_group64(v_main, bits, axis=-1)
    return QuantizedLayerKV(
        k_mode="channel" if layout == "kivi" else "token",
        bits=bits,
        k_packed=kp, k_scale=ks, k_offset=ko, k_perm_shape=kps,
        v_packed=vp, v_scale=vs, v_offset=vo, v_perm_shape=vps,
        k_res=k_res, v_res=v_res,
    )


def decompress_kv_tensor_pair(qkv: QuantizedLayerKV) -> tuple[torch.Tensor, torch.Tensor]:
    k_axis = -2 if qkv.k_mode == "channel" else -1
    k_main = unpack_dequantize_group64(
        qkv.k_packed, qkv.k_scale, qkv.k_offset, qkv.k_perm_shape, qkv.bits, axis=k_axis, out_dtype=qkv.k_res.dtype
    )
    v_main = unpack_dequantize_group64(
        qkv.v_packed, qkv.v_scale, qkv.v_offset, qkv.v_perm_shape, qkv.bits, axis=-1, out_dtype=qkv.v_res.dtype
    )
    k = torch.cat([k_main, qkv.k_res], dim=-2) if k_main.shape[-2] > 0 else qkv.k_res
    v = torch.cat([v_main, qkv.v_res], dim=-2) if v_main.shape[-2] > 0 else qkv.v_res
    return k, v


# ---------------------------------------------------------------------------
# Question-aware importance scoring using the active SDPA model (no eager swap)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_question_aware_scores(
    model,
    ctx_cache: DynamicCache,
    q_ids: torch.Tensor,
    L_ctx: int,
) -> torch.Tensor:
    """Compute masked FP32 attention importance scores [n_layers, H_kv, L_ctx] from q_ids onto ctx_cache.

    Extracts Q projections inside Qwen2Attention forward hooks while keeping SDPA active.
    """
    n_layers = len(model.model.layers)
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    gqa_groups = n_q // n_kv
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_q)
    device = q_ids.device

    scores_out = torch.zeros((n_layers, n_kv, L_ctx), dtype=torch.float32, device="cpu")
    rotary = model.model.rotary_emb
    L_q = q_ids.shape[1]
    pos_ids = torch.arange(L_ctx, L_ctx + L_q, device=device, dtype=torch.long).unsqueeze(0)
    cos_sin = rotary(ctx_cache.layers[0].keys, pos_ids)

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    # Clone cache so scoring pass does not mutate ctx_cache
    tmp_cache = DynamicCache()
    for lidx in range(n_layers):
        tmp_cache.update(ctx_cache.layers[lidx].keys.clone(), ctx_cache.layers[lidx].values.clone(), lidx)

    handles = []

    def make_hook(lidx: int):
        k_ctx = ctx_cache.layers[lidx].keys  # [1, H_kv, L_ctx, d_h]
        k_ctx_rep = k_ctx.repeat_interleave(gqa_groups, dim=1).float()  # [1, H_q, L_ctx, d_h]
        scaling = head_dim ** -0.5

        def pre_hook(mod, args, kwargs):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            q_proj = mod.q_proj(hs).view(1, L_q, n_q, head_dim).transpose(1, 2)
            k_q_proj = mod.k_proj(hs).view(1, L_q, n_kv, head_dim).transpose(1, 2)
            cos, sin = cos_sin
            q_rot, k_q_rot = apply_rotary_pos_emb(q_proj, k_q_proj, cos, sin)
            k_q_rep = k_q_rot.repeat_interleave(gqa_groups, dim=1).float()

            # Attend over all L_ctx cached keys + causal suffix keys
            k_all = torch.cat([k_ctx_rep, k_q_rep], dim=-2)  # [1, H_q, L_ctx + L_q, d_h]
            logits = torch.matmul(q_rot.float(), k_all.transpose(-2, -1)) * scaling
            # Mask future positions within suffix
            suf_mask = torch.tril(torch.ones((L_q, L_q), dtype=torch.bool, device=device))
            full_mask = torch.cat([torch.ones((L_q, L_ctx), dtype=torch.bool, device=device), suf_mask], dim=-1)
            logits = logits.masked_fill(~full_mask[None, None, :, :], torch.finfo(torch.float32).min)
            probs = F.softmax(logits, dim=-1)[:, :, :, :L_ctx]  # [1, H_q, L_q, L_ctx]
            # Sum across L_q query tokens and GQA query heads per kv_head
            imp = probs.sum(dim=-2).reshape(1, n_kv, gqa_groups, L_ctx).sum(dim=2).squeeze(0)
            scores_out[lidx].copy_(imp.cpu())

        return pre_hook

    for lidx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(lidx), with_kwargs=True))

    try:
        cache_pos = torch.arange(L_ctx, L_ctx + L_q, device=device, dtype=torch.long)
        model(input_ids=q_ids, past_key_values=tmp_cache, position_ids=pos_ids, cache_position=cache_pos, use_cache=True)
    finally:
        for h in handles:
            h.remove()
        del tmp_cache

    return scores_out


# ---------------------------------------------------------------------------
# Budget solvers and Cache Compression Application
# ---------------------------------------------------------------------------

def solve_eviction_kept_count(
    L_ctx: int,
    budget_bytes: int,
    per_head_indexed: bool,
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
) -> tuple[int, int]:
    """Return (max_K_kept, total_bytes) <= budget_bytes."""
    fp16_per_pos = 4 * n_layers * n_kv_heads * head_dim
    idx_per_pos = (2 * n_layers * n_kv_heads) if per_head_indexed else 0
    per_pos = fp16_per_pos + idx_per_pos
    k_kept = min(L_ctx, budget_bytes // per_pos)
    return int(k_kept), int(k_kept * per_pos)


def solve_hybrid_fp16_int2_counts(
    L_ctx: int,
    budget_bytes: int,
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
) -> tuple[int, int, int]:
    """Choose smallest integer g (INT2 groups of 64) maximizing n_fp16 = L_ctx - 64*g
    subject to FP16 + INT2(with 4B metadata/group) + uint16 position indices <= budget_bytes.
    """
    n_chan = 2 * n_layers * n_kv_heads * head_dim
    fp16_per_pos = 2 * n_chan
    idx_bytes = 2 * n_layers * n_kv_heads * L_ctx  # uint16 original positions for all L_ctx tokens
    int2_group_bytes = (KV_GROUP_SIZE * 2) // 8 + 4  # 20 bytes per 64-elem group per channel

    max_g = (L_ctx - (SINK_TOKENS + RECENT_TOKENS)) // KV_GROUP_SIZE
    for g in range(0, max_g + 1):
        n_int2 = g * KV_GROUP_SIZE
        n_fp16 = L_ctx - n_int2
        tot = n_fp16 * fp16_per_pos + n_chan * g * int2_group_bytes + idx_bytes
        if tot <= budget_bytes:
            return int(n_fp16), int(n_int2), int(tot)
    raise RuntimeError(f"Hybrid budget infeasible for L_ctx={L_ctx}, budget={budget_bytes}")


def build_compressed_context_cache(
    policy: str,
    ctx_cache: DynamicCache,
    L_ctx: int,
    budget_bytes: int,
    importance_scores: Optional[torch.Tensor] = None,
    evidence_token_indices: Optional[list[int]] = None,
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
) -> dict[str, Any]:
    """Apply one of the 6 policies to ctx_cache and return cache state + accounting + evidence retention."""
    ev_set = set(evidence_token_indices or [])
    new_cache = DynamicCache()

    if policy == "full_fp16":
        for lidx in range(n_layers):
            new_cache.update(ctx_cache.layers[lidx].keys.clone(), ctx_cache.layers[lidx].values.clone(), lidx)
        m16 = calc_kv_bytes_formula(L_ctx, 4, n_layers, n_kv_heads, head_dim)["M_16_bytes"]
        return {
            "cache": new_cache,
            "physical_ctx_len": L_ctx,
            "ctx_bytes": m16,
            "idx_bytes": 0,
            "retained_evidence_mean": 1.0 if ev_set else None,
            "retained_evidence_max": 1.0 if ev_set else None,
            "selection_mask_sample": None,
            "quant_layout": None,
        }

    if policy == "evict_sink_recent":
        k_kept, tot_bytes = solve_eviction_kept_count(L_ctx, budget_bytes, False, n_layers, n_kv_heads, head_dim)
        recent_n = k_kept - SINK_TOKENS
        kept_idx = list(range(SINK_TOKENS)) + list(range(L_ctx - recent_n, L_ctx))
        idx_t = torch.tensor(kept_idx, dtype=torch.long, device=ctx_cache.layers[0].keys.device)
        for lidx in range(n_layers):
            k_sel = ctx_cache.layers[lidx].keys[:, :, idx_t, :].clone()
            v_sel = ctx_cache.layers[lidx].values[:, :, idx_t, :].clone()
            new_cache.update(k_sel, v_sel, lidx)
        ev_ret = (len(ev_set.intersection(kept_idx)) / len(ev_set)) if ev_set else None
        return {
            "cache": new_cache,
            "physical_ctx_len": k_kept,
            "ctx_bytes": tot_bytes,
            "idx_bytes": 0,
            "retained_evidence_mean": ev_ret,
            "retained_evidence_max": ev_ret,
            "selection_mask_sample": kept_idx,
            "quant_layout": None,
        }

    if policy == "evict_question_aware":
        assert importance_scores is not None
        k_kept, tot_bytes = solve_eviction_kept_count(L_ctx, budget_bytes, True, n_layers, n_kv_heads, head_dim)
        n_mid = k_kept - (SINK_TOKENS + RECENT_TOKENS)
        sink_idx = torch.arange(0, SINK_TOKENS)
        rec_idx = torch.arange(L_ctx - RECENT_TOKENS, L_ctx)
        mid_lo, mid_hi = SINK_TOKENS, L_ctx - RECENT_TOKENS

        ev_fracs = []
        sample_masks = {}
        dev = ctx_cache.layers[0].keys.device
        for lidx in range(n_layers):
            k_layer = ctx_cache.layers[lidx].keys
            v_layer = ctx_cache.layers[lidx].values
            k_heads, v_heads = [], []
            for h in range(n_kv_heads):
                if n_mid > 0:
                    mid_scores = importance_scores[lidx, h, mid_lo:mid_hi]
                    _, top_rel = torch.topk(mid_scores, k=n_mid, largest=True, sorted=False)
                    mid_idx = top_rel + mid_lo
                    kept = torch.sort(torch.cat([sink_idx, mid_idx, rec_idx])).values
                else:
                    kept = torch.cat([sink_idx, rec_idx])
                if ev_set:
                    ev_fracs.append(len(ev_set.intersection(kept.tolist())) / len(ev_set))
                if lidx in (0, n_layers // 2, n_layers - 1):
                    sample_masks[f"L{lidx}_H{h}"] = kept.tolist()
                kept_dev = kept.to(dev)
                k_heads.append(k_layer[:, h : h + 1, kept_dev, :])
                v_heads.append(v_layer[:, h : h + 1, kept_dev, :])
            new_cache.update(torch.cat(k_heads, dim=1), torch.cat(v_heads, dim=1), lidx)

        idx_bytes = 2 * n_layers * n_kv_heads * k_kept
        return {
            "cache": new_cache,
            "physical_ctx_len": k_kept,
            "ctx_bytes": tot_bytes,
            "idx_bytes": idx_bytes,
            "retained_evidence_mean": float(np.mean(ev_fracs)) if ev_fracs else None,
            "retained_evidence_max": float(np.max(ev_fracs)) if ev_fracs else None,
            "selection_mask_sample": sample_masks,
            "quant_layout": None,
        }

    if policy in ("quant_int4_tok", "quant_int4_kivi"):
        layout = "tok" if policy == "quant_int4_tok" else "kivi"
        total_bytes = 0
        q_layers: list[QuantizedLayerKV] = []
        for lidx in range(n_layers):
            qkv = compress_kv_tensor_pair(ctx_cache.layers[lidx].keys, ctx_cache.layers[lidx].values, bits=4, layout=layout)
            total_bytes += qkv.packed_bytes()
            k_hat, v_hat = decompress_kv_tensor_pair(qkv)
            new_cache.update(k_hat, v_hat, lidx)
            q_layers.append(qkv)
        return {
            "cache": new_cache,
            "physical_ctx_len": L_ctx,
            "ctx_bytes": total_bytes,
            "idx_bytes": 0,
            "retained_evidence_mean": 1.0 if ev_set else None,
            "retained_evidence_max": 1.0 if ev_set else None,
            "selection_mask_sample": None,
            "quant_layout": layout,
            "quant_bits": 4,
            "q_layers": q_layers,
        }

    if policy == "hybrid_fp16_int2":
        assert importance_scores is not None
        n_fp16, n_int2, expected_bytes = solve_hybrid_fp16_int2_counts(L_ctx, budget_bytes, n_layers, n_kv_heads, head_dim)
        n_mid = n_fp16 - (SINK_TOKENS + RECENT_TOKENS)
        sink_idx = torch.arange(0, SINK_TOKENS)
        rec_idx = torch.arange(L_ctx - RECENT_TOKENS, L_ctx)
        mid_lo, mid_hi = SINK_TOKENS, L_ctx - RECENT_TOKENS
        dev = ctx_cache.layers[0].keys.device

        ev_fracs_fp16 = []
        sample_masks = {}
        actual_payload_bytes = 0
        for lidx in range(n_layers):
            k_layer = ctx_cache.layers[lidx].keys
            v_layer = ctx_cache.layers[lidx].values
            k_heads, v_heads = [], []
            for h in range(n_kv_heads):
                mid_scores = importance_scores[lidx, h, mid_lo:mid_hi]
                if n_mid > 0:
                    _, top_rel = torch.topk(mid_scores, k=n_mid, largest=True, sorted=False)
                    fp16_idx = torch.sort(torch.cat([sink_idx, top_rel + mid_lo, rec_idx])).values
                else:
                    fp16_idx = torch.cat([sink_idx, rec_idx])

                fp16_mask = torch.zeros(L_ctx, dtype=torch.bool)
                fp16_mask[fp16_idx] = True
                int2_idx = torch.nonzero(~fp16_mask, as_tuple=False).squeeze(-1)
                assert int2_idx.numel() == n_int2

                if ev_set:
                    ev_fracs_fp16.append(len(ev_set.intersection(fp16_idx.tolist())) / len(ev_set))
                if lidx in (0, n_layers // 2, n_layers - 1):
                    sample_masks[f"L{lidx}_H{h}"] = fp16_idx.tolist()

                k_fp16 = k_layer[:, h : h + 1, fp16_idx.to(dev), :]
                v_fp16 = v_layer[:, h : h + 1, fp16_idx.to(dev), :]
                k_lo = k_layer[:, h : h + 1, int2_idx.to(dev), :]
                v_lo = v_layer[:, h : h + 1, int2_idx.to(dev), :]

                qkv_lo = compress_kv_tensor_pair(k_lo, v_lo, bits=2, layout="kivi")
                actual_payload_bytes += int(k_fp16.numel() * 2 + v_fp16.numel() * 2 + qkv_lo.packed_bytes())
                k_lo_hat, v_lo_hat = decompress_kv_tensor_pair(qkv_lo)

                k_heads.append(torch.cat([k_fp16, k_lo_hat], dim=-2))
                v_heads.append(torch.cat([v_fp16, v_lo_hat], dim=-2))
            new_cache.update(torch.cat(k_heads, dim=1), torch.cat(v_heads, dim=1), lidx)

        idx_bytes = 2 * n_layers * n_kv_heads * L_ctx
        total_bytes = actual_payload_bytes + idx_bytes
        if total_bytes > budget_bytes:
            raise RuntimeError(f"Hybrid exceeded budget: {total_bytes} > {budget_bytes}")
        return {
            "cache": new_cache,
            "physical_ctx_len": L_ctx,
            "ctx_bytes": total_bytes,
            "idx_bytes": idx_bytes,
            "n_fp16": n_fp16,
            "n_int2": n_int2,
            "retained_evidence_mean": float(np.mean(ev_fracs_fp16)) if ev_fracs_fp16 else None,
            "retained_evidence_max": float(np.max(ev_fracs_fp16)) if ev_fracs_fp16 else None,
            "selection_mask_sample": sample_masks,
            "quant_layout": "hybrid_fp16_int2_kivi",
        }

    raise ValueError(f"Unknown policy: {policy}")


# ---------------------------------------------------------------------------
# End-to-end Case Evaluation (Split Prefill -> Compression -> Suffix -> Decode)
# ---------------------------------------------------------------------------

def _clone_cache(cache: DynamicCache) -> DynamicCache:
    out = DynamicCache()
    for lidx, layer in enumerate(cache.layers):
        out.update(layer.keys.clone(), layer.values.clone(), lidx)
    return out


def _append_and_quantize_if_needed(
    cache: DynamicCache,
    already_quant_len: int,
    bits: Optional[int],
    layout: Optional[str],
) -> int:
    """For C3 policies, quantize newly completed 64-token blocks dynamically as decode appends tokens."""
    if bits is None or layout is None:
        return already_quant_len
    cur_L = cache.layers[0].get_seq_length()
    target_quant = (cur_L // KV_GROUP_SIZE) * KV_GROUP_SIZE
    if target_quant > already_quant_len:
        for lidx in range(len(cache.layers)):
            k_slice = cache.layers[lidx].keys[:, :, already_quant_len:target_quant, :].contiguous()
            v_slice = cache.layers[lidx].values[:, :, already_quant_len:target_quant, :].contiguous()
            qkv = compress_kv_tensor_pair(k_slice, v_slice, bits=bits, layout=layout)
            kh, vh = decompress_kv_tensor_pair(qkv)
            cache.layers[lidx].keys[:, :, already_quant_len:target_quant, :].copy_(kh)
            cache.layers[lidx].values[:, :, already_quant_len:target_quant, :].copy_(vh)
        return target_quant
    return already_quant_len


@torch.inference_mode()
def evaluate_single_kv_case(
    model,
    tokenizer,
    case: dict[str, Any],
    tok_split: dict[str, Any],
    policy: str,
    fp16_ref_logits_last: Optional[torch.Tensor] = None,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Run one prompt under `policy` and return retrieval, NLL, KL, memory, and timing."""
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)

    ctx_ids = torch.tensor([tok_split["ctx_ids"]], dtype=torch.long, device=device)
    q_ids = torch.tensor([tok_split["q_ids"]], dtype=torch.long, device=device)
    L_ctx = tok_split["L_ctx"]
    L_q = tok_split["L_q"]
    budget_bytes = calc_kv_bytes_formula(L_ctx, 4, n_layers, n_kv, head_dim)["M_b_bytes"]

    # 1. Context prefill
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    ctx_out = model(input_ids=ctx_ids, use_cache=True)
    ctx_cache = ctx_out.past_key_values
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_sec = time.perf_counter() - t0

    # 2. Importance scoring (only for question-aware or hybrid) + compression
    t1 = time.perf_counter()
    imp_scores = None
    if policy in ("evict_question_aware", "hybrid_fp16_int2"):
        imp_scores = compute_question_aware_scores(model, ctx_cache, q_ids, L_ctx)
    comp = build_compressed_context_cache(
        policy=policy,
        ctx_cache=ctx_cache,
        L_ctx=L_ctx,
        budget_bytes=budget_bytes,
        importance_scores=imp_scores,
        evidence_token_indices=tok_split["evidence_token_indices"],
        n_layers=n_layers,
        n_kv_heads=n_kv,
        head_dim=head_dim,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    score_compress_sec = time.perf_counter() - t1

    cache = comp["cache"]
    L_phys = comp["physical_ctx_len"]

    # 3. Question suffix forward (preserving original absolute rotary position_ids!) + greedy decode
    t2 = time.perf_counter()
    q_pos_ids = torch.arange(L_ctx, L_ctx + L_q, dtype=torch.long, device=device).unsqueeze(0)
    q_cache_pos = torch.arange(L_phys, L_phys + L_q, dtype=torch.long, device=device)
    q_out = model(
        input_ids=q_ids,
        past_key_values=cache,
        position_ids=q_pos_ids,
        cache_position=q_cache_pos,
        use_cache=True,
    )
    logits_last = q_out.logits[:, -1, :].float()

    # Keep a lightweight copy ONLY of cache state right after question suffix if we need ans_nll separately
    # Note: clone after greedy decode? No, greedy decode mutates cur_cache in-place!
    # So measure suffix+greedy decode timing cleanly excluding ans_nll clone/eval:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    suffix_sec = time.perf_counter() - t2

    # Post-prefill KL from FP16 reference
    kl_from_fp16 = None
    if fp16_ref_logits_last is not None:
        p_ref = F.log_softmax(fp16_ref_logits_last.float(), dim=-1)
        p_cur = F.log_softmax(logits_last, dim=-1)
        kl_from_fp16 = float(F.kl_div(p_cur, p_ref.exp(), reduction="sum", log_target=False).item())

    # 4. Greedy generation using absolute rotary position_ids and physical cache length
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    gen_ids: list[int] = []
    next_tok = logits_last.argmax(dim=-1, keepdim=True)
    cur_abs_pos = L_ctx + L_q
    cur_phys_pos = L_phys + L_q
    eos_id = tokenizer.eos_token_id
    cur_cache = q_out.past_key_values
    quant_boundary = (L_ctx // KV_GROUP_SIZE) * KV_GROUP_SIZE

    for _ in range(max_new_tokens):
        tid = int(next_tok.item())
        gen_ids.append(tid)
        if eos_id is not None and tid == eos_id:
            break
        if policy in ("quant_int4_tok", "quant_int4_kivi"):
            quant_boundary = _append_and_quantize_if_needed(
                cur_cache, quant_boundary, 4, comp["quant_layout"]
            )

        p_id = torch.tensor([[cur_abs_pos]], dtype=torch.long, device=device)
        c_pos = torch.tensor([cur_phys_pos], dtype=torch.long, device=device)
        step_out = model(
            input_ids=next_tok,
            past_key_values=cur_cache,
            position_ids=p_id,
            cache_position=c_pos,
            use_cache=True,
        )
        cur_cache = step_out.past_key_values
        next_tok = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur_abs_pos += 1
        cur_phys_pos += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_sec = suffix_sec + (time.perf_counter() - t3)
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    # 5. Teacher-forced reference answer NLL outside timed decode / peak memory (crop cache back to L_phys + L_q)
    ans_ids_list = tokenizer.encode(case["code"], add_special_tokens=False)
    ans_nll = None
    if ans_ids_list:
        for layer in cur_cache.layers:
            layer.crop(L_phys + L_q)
        ans_t = torch.tensor([ans_ids_list], dtype=torch.long, device=device)
        total_ans_nll = float(-F.log_softmax(logits_last, dim=-1)[0, ans_ids_list[0]].item())
        if len(ans_ids_list) > 1:
            a_in = ans_t[:, :-1]
            a_pos_ids = torch.arange(L_ctx + L_q, L_ctx + L_q + a_in.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            a_cpos = torch.arange(L_phys + L_q, L_phys + L_q + a_in.shape[1], dtype=torch.long, device=device)
            a_out = model(
                input_ids=a_in,
                past_key_values=cur_cache,
                position_ids=a_pos_ids,
                cache_position=a_cpos,
                use_cache=True,
            )
            a_lp = F.log_softmax(a_out.logits.float(), dim=-1)
            for j in range(1, len(ans_ids_list)):
                total_ans_nll += float(-a_lp[0, j - 1, ans_ids_list[j]].item())
        ans_nll = total_ans_nll / len(ans_ids_list)

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred_code = extract_five_digit_answer(gen_text)
    exact_match = int(pred_code == case["code"])

    # Compute decode-time shared FP16 memory (question suffix + generated tokens)
    shared_decode_tokens = L_q + len(gen_ids)
    shared_decode_bytes = 4 * n_layers * n_kv * head_dim * shared_decode_tokens
    m16_ctx = calc_kv_bytes_formula(L_ctx, 4, n_layers, n_kv, head_dim)["M_16_bytes"]

    return {
        "id": case["id"],
        "policy": policy,
        "target_tokens": case["target_tokens"],
        "position": case["position"],
        "distractors": case["distractors"],
        "L_ctx": L_ctx,
        "L_q": L_q,
        "n_gen": len(gen_ids),
        "gold_code": case["code"],
        "pred_code": pred_code,
        "generated_text": gen_text,
        "exact_match": exact_match,
        "reference_answer_nll": ans_nll,
        "post_prefill_kl_fp16": kl_from_fp16,
        "retained_evidence_mean": comp["retained_evidence_mean"],
        "retained_evidence_max": comp["retained_evidence_max"],
        "ctx_packed_bytes": comp["ctx_bytes"],
        "ctx_budget_int4_bytes": budget_bytes,
        "ctx_full_fp16_bytes": m16_ctx,
        "realized_ctx_fraction_fp16": comp["ctx_bytes"] / m16_ctx,
        "shared_decode_fp16_bytes": shared_decode_bytes,
        "total_live_cache_bytes": comp["ctx_bytes"] + shared_decode_bytes,
        "prefill_sec": prefill_sec,
        "score_compress_sec": score_compress_sec,
        "decode_sec": decode_sec,
        "peak_allocated_bytes": peak_alloc,
        "logits_last_cpu": logits_last.cpu() if policy == "full_fp16" else None,
        "selection_mask_sample": comp["selection_mask_sample"],
    }


@torch.inference_mode()
def validate_c1_full_cache_equivalence(model, tokenizer, case: dict[str, Any], tok_split: dict[str, Any]) -> dict[str, Any]:
    """Verify C1 split-prefill/decode reproduces direct full-forward logits within FP16 tolerance."""
    device = next(model.parameters()).device
    full_ids = torch.tensor([tok_split["full_ids"]], dtype=torch.long, device=device)
    ctx_ids = torch.tensor([tok_split["ctx_ids"]], dtype=torch.long, device=device)
    q_ids = torch.tensor([tok_split["q_ids"]], dtype=torch.long, device=device)
    L_ctx = tok_split["L_ctx"]
    L_q = tok_split["L_q"]

    full_out = model(input_ids=full_ids, use_cache=False)
    direct_q_logits = full_out.logits[:, L_ctx:, :].float()

    ctx_out = model(input_ids=ctx_ids, use_cache=True)
    pos_ids = torch.arange(L_ctx, L_ctx + L_q, dtype=torch.long, device=device).unsqueeze(0)
    c_pos = torch.arange(L_ctx, L_ctx + L_q, dtype=torch.long, device=device)
    split_out = model(
        input_ids=q_ids,
        past_key_values=ctx_out.past_key_values,
        position_ids=pos_ids,
        cache_position=c_pos,
        use_cache=True,
    )
    split_q_logits = split_out.logits.float()

    diff = (direct_q_logits - split_q_logits).abs()
    all_finite = bool(torch.isfinite(direct_q_logits).all() and torch.isfinite(split_q_logits).all())
    return {
        "case_id": case["id"],
        "all_finite": all_finite,
        "q_logits_max_abs_diff": float(diff.max().item()),
        "q_logits_mean_abs_diff": float(diff.mean().item()),
        "top1_match_last": bool(direct_q_logits[:, -1].argmax().item() == split_q_logits[:, -1].argmax().item()),
    }


@torch.inference_mode()
def benchmark_c5_packed_encode_decode(
    batch_sizes: tuple[int, ...] = (1, 8),
    lengths: tuple[int, ...] = (2048, 4096),
    n_layers: int = 28,
    n_kv_heads: int = 2,
    n_q_heads: int = 12,
    head_dim: int = 128,
    repeats: int = 10,
    device: str | torch.device = "cuda",
) -> list[dict[str, Any]]:
    """Standalone C5 benchmark: packing, unpacking/dequantization, serialized bytes,
    and decode-step SDPA latency with unpacking vs FP16 reference at matched shapes.
    """
    dev = torch.device(device)
    results = []

    for B in batch_sizes:
        for L in lengths:
            gen = torch.Generator(device="cpu").manual_seed(6304 + B * 10000 + L)
            # Benchmark single-layer packed encode/decode scaled or full 28 layers:
            # Wait: full 28 layers at B=8, L=4096 in FP16 takes 28 * 8 * 2 * 4096 * 128 * 4 = 0.94 GB (fits on L4/T4!)
            k_full = torch.randn(n_layers, B, n_kv_heads, L, head_dim, generator=gen, dtype=torch.float16).to(dev)
            v_full = torch.randn(n_layers, B, n_kv_heads, L, head_dim, generator=gen, dtype=torch.float16).to(dev)
            q_step = torch.randn(n_layers, B, n_q_heads, 1, head_dim, generator=gen, dtype=torch.float16).to(dev)
            gqa_rep = n_q_heads // n_kv_heads

            # FP16 reference decode step across all layers
            def fp16_decode_step():
                out = None
                for lidx in range(n_layers):
                    kr = k_full[lidx].repeat_interleave(gqa_rep, dim=1)
                    vr = v_full[lidx].repeat_interleave(gqa_rep, dim=1)
                    out = F.scaled_dot_product_attention(q_step[lidx], kr, vr, is_causal=False)
                return out

            _, fp16_mem = reset_and_measure_peak_memory_bytes(fp16_decode_step, dev)
            fp16_lat = measure_cuda_latency_ms(fp16_decode_step, warmup=3, repeats=repeats, device=dev)
            fp16_bytes = int(k_full.numel() * 2 + v_full.numel() * 2)
            results.append({
                "B": B,
                "L": L,
                "format": "fp16_ref",
                "serialized_bytes": fp16_bytes,
                "pack_latency": None,
                "unpack_latency": None,
                "decode_step_with_unpack_latency": fp16_lat,
                "decode_step_memory": fp16_mem,
            })

            for bits, fmt_name in ((4, "int4_kivi"), (2, "int2_kivi")):
                def pack_all():
                    return [compress_kv_tensor_pair(k_full[l], v_full[l], bits=bits, layout="kivi") for l in range(n_layers)]

                packed_layers = pack_all()
                ser_bytes = sum(ql.packed_bytes() for ql in packed_layers)
                pack_lat = measure_cuda_latency_ms(pack_all, warmup=3, repeats=repeats, device=dev)

                def unpack_all():
                    return [decompress_kv_tensor_pair(ql) for ql in packed_layers]

                _, unpack_mem = reset_and_measure_peak_memory_bytes(unpack_all, dev)
                unpack_lat = measure_cuda_latency_ms(unpack_all, warmup=3, repeats=repeats, device=dev)

                def decode_with_unpack():
                    out = None
                    for lidx in range(n_layers):
                        kh, vh = decompress_kv_tensor_pair(packed_layers[lidx])
                        kr = kh.repeat_interleave(gqa_rep, dim=1)
                        vr = vh.repeat_interleave(gqa_rep, dim=1)
                        out = F.scaled_dot_product_attention(q_step[lidx], kr, vr, is_causal=False)
                    return out

                _, dec_mem = reset_and_measure_peak_memory_bytes(decode_with_unpack, dev)
                dec_lat = measure_cuda_latency_ms(decode_with_unpack, warmup=3, repeats=repeats, device=dev)

                results.append({
                    "B": B,
                    "L": L,
                    "format": fmt_name,
                    "serialized_bytes": ser_bytes,
                    "formula_bytes": calc_kv_bytes_formula(L, bits, n_layers, n_kv_heads, head_dim)["M_b_bytes"] * B,
                    "pack_latency": pack_lat,
                    "unpack_latency": unpack_lat,
                    "unpack_memory": unpack_mem,
                    "decode_step_with_unpack_latency": dec_lat,
                    "decode_step_memory": dec_mem,
                })
                del packed_layers
            del k_full, v_full, q_step
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    return results
