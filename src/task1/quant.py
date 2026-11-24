"""Weight-only PTQ (Q1 uniform, Q2 AWQ-style activation scaling, Q3 mixed precision)."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task1.model_utils import (
    EligibleProjection,
    compute_unique_parameter_accounting,
    enumerate_eligible_projections,
    get_transformer_layers,
)
from src.task1.packing import (
    DEFAULT_GROUP_SIZE,
    PackedQuantizedWeight,
    calc_quantized_matrix_bytes,
    pack_bits,
)

DEFAULT_ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def quantize_weight_asymmetric(
    weight: torch.Tensor,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    channel_inv_scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, PackedQuantizedWeight]:
    """Group-wise asymmetric uniform weight-only quantization along input columns.

    Conventions (fixed across Q1/Q2/Q3):
    - Group size g=128 along input columns (padded on right with zeros if needed).
    - Zero is included in [w_min, w_max] so integer zero-point z in [0, 2^b - 1]
      and 0.0 maps losslessly.
    - scale = max(w_max - w_min, 1e-8) / (2^b - 1), quantized to FP16.
    - z = clip(round(-w_min / scale), 0, 2^b - 1) as uint8, packed at b bits.
    - q = clip(round(w / scale) + z, 0, 2^b - 1) as uint8, packed at b bits.
    - Reconstructed weight uses exact FP16-stored scale and unpacked q, z.
    """
    d_out, d_in = weight.shape
    n_groups_per_row = math.ceil(d_in / group_size)
    d_pad = n_groups_per_row * group_size
    w = weight.detach().to(torch.float32)
    if d_pad > d_in:
        w = F.pad(w, (0, d_pad - d_in), value=0.0)
    w_grp = w.reshape(d_out, n_groups_per_row, group_size)

    w_min = torch.clamp(w_grp.amin(dim=-1, keepdim=True), max=0.0)
    w_max = torch.clamp(w_grp.amax(dim=-1, keepdim=True), min=0.0)
    qmax = (1 << bits) - 1
    scale_fp32 = torch.clamp((w_max - w_min) / qmax, min=1e-8)
    scales_fp16 = scale_fp32.squeeze(-1).to(torch.float16)
    scale_eff = scales_fp16.to(torch.float32).unsqueeze(-1).clamp(min=1e-8)

    zp = torch.clamp(torch.round(-w_min / scale_eff), 0, qmax).to(torch.uint8)
    q = torch.clamp(torch.round(w_grp / scale_eff) + zp.to(torch.float32), 0, qmax).to(torch.uint8)

    packed = PackedQuantizedWeight(
        d_out=d_out,
        d_in=d_in,
        bits=bits,
        group_size=group_size,
        n_groups_per_row=n_groups_per_row,
        qweight=pack_bits(q.cpu(), bits),
        scales=scales_fp16.cpu(),
        qzeros=pack_bits(zp.cpu(), bits),
        channel_inv_scales=channel_inv_scales.detach().to(torch.float16).cpu()
        if channel_inv_scales is not None
        else None,
    )
    w_hat_scaled = (scale_eff * (q.to(torch.float32) - zp.to(torch.float32))).reshape(d_out, d_pad)[:, :d_in]
    return w_hat_scaled.to(weight.dtype), packed


def _compute_quant_storage_summary(
    model: nn.Module,
    packed_map: dict[str, PackedQuantizedWeight],
) -> dict[str, Any]:
    acct = compute_unique_parameter_accounting(model)
    projections = enumerate_eligible_projections(model)

    tot_w_bytes = 0
    tot_s_bytes = 0
    tot_z_bytes = 0
    tot_c_bytes = 0
    weighted_bits = 0
    per_proj = []

    for p in projections:
        pw = packed_map[p.name]
        has_c = pw.channel_inv_scales is not None
        spec = calc_quantized_matrix_bytes(p.d_out, p.d_in, pw.bits, pw.group_size, has_c)
        actual = pw.actual_bytes()
        if actual != spec.total_bytes:
            raise RuntimeError(f"Packed byte mismatch on {p.name}: {actual} != {spec.total_bytes}")
        tot_w_bytes += spec.weight_bytes
        tot_s_bytes += spec.scale_bytes
        tot_z_bytes += spec.zero_bytes
        tot_c_bytes += spec.channel_scale_bytes
        weighted_bits += p.numel * pw.bits
        per_proj.append({
            "name": p.name,
            "layer_idx": p.layer_idx,
            "proj_type": p.proj_type,
            "bits": pw.bits,
            "numel": p.numel,
            "weight_bytes": spec.weight_bytes,
            "scale_bytes": spec.scale_bytes,
            "zero_bytes": spec.zero_bytes,
            "channel_scale_bytes": spec.channel_scale_bytes,
            "total_bytes": actual,
        })

    elig_packed_bytes = tot_w_bytes + tot_s_bytes + tot_z_bytes + tot_c_bytes
    total_packed_bytes = elig_packed_bytes + acct["uncompressed_fp16_bytes"]
    return {
        "eligible_params": acct["eligible_params"],
        "total_unique_params": acct["total_unique_params"],
        "baseline_fp16_bytes": acct["baseline_fp16_bytes"],
        "uncompressed_fp16_bytes": acct["uncompressed_fp16_bytes"],
        "weight_bytes": tot_w_bytes,
        "scale_bytes": tot_s_bytes,
        "zero_bytes": tot_z_bytes,
        "channel_scale_bytes": tot_c_bytes,
        "eligible_packed_bytes": elig_packed_bytes,
        "total_packed_bytes": total_packed_bytes,
        "compression_ratio": acct["baseline_fp16_bytes"] / total_packed_bytes,
        "nominal_bits_per_eligible_weight": weighted_bits / acct["eligible_params"],
        "effective_bits_per_eligible_weight": elig_packed_bytes * 8.0 / acct["eligible_params"],
        "per_projection": per_proj,
    }


@torch.inference_mode()
def apply_q1_uniform(
    model: nn.Module,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    """Apply uniform group-wise asymmetric PTQ in-place and return packed-byte summary."""
    projections = enumerate_eligible_projections(model)
    packed_map: dict[str, PackedQuantizedWeight] = {}
    for p in projections:
        w_hat, packed = quantize_weight_asymmetric(p.module.weight.data, bits=bits, group_size=group_size)
        p.module.weight.data.copy_(w_hat)
        packed_map[p.name] = packed
    summary = _compute_quant_storage_summary(model, packed_map)
    summary["quantizer"] = "Q1_uniform"
    summary["bits"] = bits
    summary["group_size"] = group_size
    return summary


def _make_awq_scale(x_mean_abs: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute bounded per-channel AWQ scale s_j proportional to (x_mean_abs_j)^alpha."""
    if alpha == 0.0:
        return torch.ones_like(x_mean_abs, dtype=torch.float32)
    x = x_mean_abs.detach().float().clamp(min=1e-4)
    s = x.pow(alpha)
    denom = torch.sqrt(s.max() * s.min()).clamp(min=1e-6)
    return (s / denom).clamp(1e-2, 1e2)


def _attach_input_channel_scale_hook(mod: nn.Linear, inv_scale: torch.Tensor) -> None:
    """Apply compensating per-channel input scaling (x * inv_scale) before quantized linear weight."""
    mod.register_buffer("awq_inv_scale", inv_scale.to(mod.weight.dtype).to(mod.weight.device), persistent=False)

    def pre_hook(module, inputs):
        return (inputs[0] * module.awq_inv_scale,) + inputs[1:]

    handle = mod.register_forward_pre_hook(pre_hook)
    if not hasattr(mod, "_awq_hook_handles"):
        mod._awq_hook_handles = []
    mod._awq_hook_handles.append(handle)


def clear_awq_hooks_and_buffers(model: nn.Module) -> None:
    """Remove any AWQ pre-hooks and non-persistent channel-scale buffers before restoring base weights."""
    for mod in model.modules():
        if hasattr(mod, "_awq_hook_handles"):
            for h in mod._awq_hook_handles:
                h.remove()
            del mod._awq_hook_handles
        if hasattr(mod, "awq_inv_scale"):
            delattr(mod, "awq_inv_scale")


@torch.inference_mode()
def apply_q2_awq(
    model: nn.Module,
    sequences: list[list[int]],
    bits_map: int | dict[str, int],
    device: str | torch.device = "cuda",
    group_size: int = DEFAULT_GROUP_SIZE,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    batch_size: int = 4,
    verify_rescaling: bool = True,
) -> dict[str, Any]:
    """Apply AWQ-style activation-dependent per-channel scaling + Q1 quantizer block-by-block.

    Folding rules:
    - input_layernorm -> [q_proj, k_proj, v_proj]: folded into input_layernorm.weight (0 extra bytes).
    - o_proj: v_proj has GQA (2->12) + bias, so compensating FP16 channel scales are serialized
      and included in packed bytes (2 * d_in bytes per layer).
    - up_proj -> down_proj: down_proj inverse scale folded into up_proj.weight (compatible linear,
      bias=False, 0 extra bytes).
    - post_attention_layernorm -> [gate_proj, up_proj]: folded into post_attention_layernorm.weight (0 extra bytes).
    """
    model.eval()
    layers = get_transformer_layers(model)
    projections = enumerate_eligible_projections(model)
    bmap = {
        p.name: int(bits_map[p.name] if isinstance(bits_map, dict) else bits_map)
        for p in projections
    }

    # Capture initial hidden states entering block 0 on CPU in FP16 (~200 MB for 128x512x1536)
    hidden_cpu: list[torch.Tensor] = []
    attn_masks_cpu: list[Optional[torch.Tensor]] = []
    pos_ids_cpu: list[torch.Tensor] = []

    def capture_block0_hook(_mod, args, kwargs):
        hidden_cpu.append(args[0].detach().cpu())
        pos_ids_cpu.append(kwargs["position_ids"].detach().cpu() if "position_ids" in kwargs and kwargs["position_ids"] is not None else torch.arange(args[0].shape[1]).unsqueeze(0))
        attn_masks_cpu.append(kwargs.get("attention_mask").detach().cpu() if kwargs.get("attention_mask") is not None else None)
        raise _StopForward()

    class _StopForward(Exception):
        pass

    h0 = layers[0].register_forward_pre_hook(capture_block0_hook, with_kwargs=True)
    try:
        for start in range(0, len(sequences), batch_size):
            ids = torch.tensor(sequences[start : start + batch_size], dtype=torch.long, device=device)
            try:
                model(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
    finally:
        h0.remove()

    packed_map: dict[str, PackedQuantizedWeight] = {}
    search_records: list[dict[str, Any]] = []
    max_rescale_rel_err = 0.0

    # Helper to compute rotary embeddings identically to Qwen2Model
    rotary_emb = model.model.rotary_emb if hasattr(model, "model") and hasattr(model.model, "rotary_emb") else None

    for lidx, block in enumerate(layers):
        attn = block.self_attn
        mlp = block.mlp
        dev = next(block.parameters()).device

        # 1. Run current FP16 block across calibration batches to gather inputs to:
        #    (a) input_layernorm -> x_qkv
        #    (b) o_proj -> x_o
        #    (c) post_attention_layernorm -> x_mlp
        #    (d) down_proj -> x_down
        x_qkv_list, x_o_list, x_mlp_list, x_down_list = [], [], [], []
        h_q = attn.q_proj.register_forward_pre_hook(lambda _m, inp: x_qkv_list.append(inp[0].detach().cpu()))
        h_o = attn.o_proj.register_forward_pre_hook(lambda _m, inp: x_o_list.append(inp[0].detach().cpu()))
        h_g = mlp.gate_proj.register_forward_pre_hook(lambda _m, inp: x_mlp_list.append(inp[0].detach().cpu()))
        h_d = mlp.down_proj.register_forward_pre_hook(lambda _m, inp: x_down_list.append(inp[0].detach().cpu()))
        try:
            for b_i, h_cpu in enumerate(hidden_cpu):
                h_dev = h_cpu.to(dev)
                pids = pos_ids_cpu[b_i].to(dev)
                amask = attn_masks_cpu[b_i].to(dev) if attn_masks_cpu[b_i] is not None else None
                pos_emb = rotary_emb(h_dev, pids) if rotary_emb is not None else None
                kw = {"position_ids": pids, "attention_mask": amask, "use_cache": False}
                if pos_emb is not None:
                    kw["position_embeddings"] = pos_emb
                _ = block(h_dev, **kw)
        finally:
            h_q.remove(); h_o.remove(); h_g.remove(); h_d.remove()

        def _search_joint_scales(
            projs: list[tuple[str, nn.Linear]],
            x_batches_cpu: list[torch.Tensor],
        ) -> tuple[float, torch.Tensor, float]:
            # Compute channel mean abs across all batches
            abs_sum = None
            cnt = 0
            for xb in x_batches_cpu:
                xf = xb.reshape(-1, xb.shape[-1]).float()
                s = xf.abs().sum(dim=0)
                abs_sum = s if abs_sum is None else (abs_sum + s)
                cnt += int(xf.shape[0])
            x_mean = (abs_sum / max(cnt, 1)).to(dev)

            # Precompute quantized scaled weights for every candidate alpha ONCE (bias cancels in y_ref - y_hat)
            cand_data: list[tuple[float, torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
            for alpha in alpha_grid:
                s = _make_awq_scale(x_mean, float(alpha))
                inv_s = (1.0 / s).to(projs[0][1].weight.dtype)
                w_hats = []
                for pname, mod in projs:
                    w_s = mod.weight.data * s.to(mod.weight.dtype).unsqueeze(0)
                    w_hat, _ = quantize_weight_asymmetric(w_s, bits=bmap[pname], group_size=group_size)
                    w_hats.append(w_hat)
                cand_data.append((float(alpha), s, inv_s, w_hats))

            losses = [0.0] * len(cand_data)
            for xb in x_batches_cpu:
                xd = xb.to(dev)
                refs_b = [F.linear(xd, mod.weight) for _, mod in projs]
                for a_idx, (_, _, inv_s, w_hats) in enumerate(cand_data):
                    xd_scaled = xd * inv_s
                    for p_j, w_hat in enumerate(w_hats):
                        y_hat = F.linear(xd_scaled, w_hat)
                        diff = (refs_b[p_j] - y_hat).float()
                        losses[a_idx] += float((diff * diff).sum().item())

            best_idx = int(np.argmin(losses))
            best_alpha, best_scale, _, _ = cand_data[best_idx]
            return best_alpha, best_scale, float(losses[best_idx])

        # Verify rescaling invariance on batch 0 before modifying weights
        if verify_rescaling:
            h0_dev = hidden_cpu[0].to(dev)
            pids0 = pos_ids_cpu[0].to(dev)
            am0 = attn_masks_cpu[0].to(dev) if attn_masks_cpu[0] is not None else None
            pe0 = rotary_emb(h0_dev, pids0) if rotary_emb is not None else None
            kw0 = {"position_ids": pids0, "attention_mask": am0, "use_cache": False}
            if pe0 is not None:
                kw0["position_embeddings"] = pe0
            raw_ref = block(h0_dev, **kw0)
            ref_blk_out = (raw_ref[0] if isinstance(raw_ref, tuple) else raw_ref).float()

        # --- Sub-block A: input_layernorm -> q, k, v ---
        qkv_projs = [
            (f"model.layers.{lidx}.self_attn.q_proj", attn.q_proj),
            (f"model.layers.{lidx}.self_attn.k_proj", attn.k_proj),
            (f"model.layers.{lidx}.self_attn.v_proj", attn.v_proj),
        ]
        a_qkv, s_qkv, loss_qkv = _search_joint_scales(qkv_projs, x_qkv_list)
        block.input_layernorm.weight.data.div_(s_qkv.to(block.input_layernorm.weight.dtype))
        for _, mod in qkv_projs:
            mod.weight.data.mul_(s_qkv.to(mod.weight.dtype).unsqueeze(0))

        # --- Sub-block B: o_proj (serialized FP16 channel scales due to GQA + v_proj.bias) ---
        o_name = f"model.layers.{lidx}.self_attn.o_proj"
        a_o, s_o, loss_o = _search_joint_scales([(o_name, attn.o_proj)], x_o_list)
        inv_s_o = (1.0 / s_o).to(attn.o_proj.weight.dtype)

        # Free attention activation buffers before MLP search
        del x_qkv_list, x_o_list

        # --- Sub-block C: up_proj -> down_proj (fold 1/s_down into up_proj.weight BEFORE quantizing up_proj!) ---
        d_name = f"model.layers.{lidx}.mlp.down_proj"
        a_down, s_down, loss_down = _search_joint_scales([(d_name, mlp.down_proj)], x_down_list)
        if mlp.up_proj.bias is not None:
            raise RuntimeError("Unexpected bias on up_proj during AWQ folding")
        mlp.up_proj.weight.data.div_(s_down.to(mlp.up_proj.weight.dtype).unsqueeze(1))
        mlp.down_proj.weight.data.mul_(s_down.to(mlp.down_proj.weight.dtype).unsqueeze(0))

        # --- Sub-block D: post_attention_layernorm -> gate_proj, up_proj ---
        mlp_projs = [
            (f"model.layers.{lidx}.mlp.gate_proj", mlp.gate_proj),
            (f"model.layers.{lidx}.mlp.up_proj", mlp.up_proj),
        ]
        a_mlp, s_mlp, loss_mlp = _search_joint_scales(mlp_projs, x_mlp_list)
        block.post_attention_layernorm.weight.data.div_(s_mlp.to(block.post_attention_layernorm.weight.dtype))
        for _, mod in mlp_projs:
            mod.weight.data.mul_(s_mlp.to(mod.weight.dtype).unsqueeze(0))

        # Attach o_proj compensating inverse scale BEFORE rescaling invariance check & quantization
        attn.o_proj.weight.data.mul_(s_o.to(attn.o_proj.weight.dtype).unsqueeze(0))
        _attach_input_channel_scale_hook(attn.o_proj, inv_s_o)

        if verify_rescaling:
            raw_res = block(h0_dev, **kw0)
            rescaled_blk_out = (raw_res[0] if isinstance(raw_res, tuple) else raw_res).float()
            rel_err = float(
                ((ref_blk_out - rescaled_blk_out).norm() / ref_blk_out.norm().clamp(min=1e-8)).item()
            )
            max_rescale_rel_err = max(max_rescale_rel_err, rel_err)
            if rel_err > 5e-2:
                raise RuntimeError(f"Full-precision AWQ rescaling failed on block {lidx}: rel_err={rel_err:.4e}")

        # Now quantize all 7 projections in block `lidx`
        for pname, mod in qkv_projs + mlp_projs + [(d_name, mlp.down_proj)]:
            w_hat, packed = quantize_weight_asymmetric(mod.weight.data, bits=bmap[pname], group_size=group_size)
            mod.weight.data.copy_(w_hat)
            packed_map[pname] = packed

        w_hat_o, packed_o = quantize_weight_asymmetric(
            attn.o_proj.weight.data,
            bits=bmap[o_name],
            group_size=group_size,
            channel_inv_scales=inv_s_o,
        )
        attn.o_proj.weight.data.copy_(w_hat_o)
        packed_map[o_name] = packed_o

        search_records.append({
            "layer_idx": lidx,
            "alpha_qkv": a_qkv,
            "alpha_o": a_o,
            "alpha_mlp": a_mlp,
            "alpha_down": a_down,
        })

        # Propagate quantized block outputs sequentially to form inputs for block lidx + 1
        if lidx + 1 < len(layers):
            next_hidden_cpu = []
            for b_i, h_cpu in enumerate(hidden_cpu):
                h_dev = h_cpu.to(dev)
                pids = pos_ids_cpu[b_i].to(dev)
                amask = attn_masks_cpu[b_i].to(dev) if attn_masks_cpu[b_i] is not None else None
                pos_emb = rotary_emb(h_dev, pids) if rotary_emb is not None else None
                kw = {"position_ids": pids, "attention_mask": amask, "use_cache": False}
                if pos_emb is not None:
                    kw["position_embeddings"] = pos_emb
                raw_out = block(h_dev, **kw)
                out = (raw_out[0] if isinstance(raw_out, tuple) else raw_out).detach().cpu()
                next_hidden_cpu.append(out)
            hidden_cpu = next_hidden_cpu

    summary = _compute_quant_storage_summary(model, packed_map)
    summary["quantizer"] = "Q2_AWQ"
    summary["alpha_grid"] = list(alpha_grid)
    summary["group_size"] = group_size
    summary["max_full_precision_rescale_rel_l2"] = max_rescale_rel_err
    summary["awq_search_alphas"] = search_records
    return summary


# ---------------------------------------------------------------------------
# Q3 Mixed-Precision Sensitivity & Constrained Bit Allocation
# ---------------------------------------------------------------------------

def measure_quantization_distortion_matrix(
    model: nn.Module,
    sequences: list[list[int]],
    candidate_bits: Sequence[int] = (2, 3, 4, 8),
    device: str | torch.device = "cuda",
    group_size: int = DEFAULT_GROUP_SIZE,
    batch_size: int = 4,
    max_sequences: int = 64,
) -> dict[str, dict[int, float]]:
    """Measure per-projection local output MSE ||(W - W_hat^(b)) X||_F^2 on calibration inputs.

    Streams block-by-block so only one Transformer block's quantized candidates sit in VRAM.
    """
    model.eval()
    projections = enumerate_eligible_projections(model)
    layers = get_transformer_layers(model)
    seqs = sequences[:max_sequences]

    err_matrix: dict[str, dict[int, float]] = {p.name: {b: 0.0 for b in candidate_bits} for p in projections}

    hidden_cpu: list[torch.Tensor] = []
    pos_ids_cpu: list[torch.Tensor] = []
    attn_masks_cpu: list[Optional[torch.Tensor]] = []

    class _StopForward(Exception):
        pass

    def capture_block0(_mod, args, kwargs):
        hidden_cpu.append(args[0].detach().cpu())
        pos_ids_cpu.append(
            kwargs["position_ids"].detach().cpu()
            if "position_ids" in kwargs and kwargs["position_ids"] is not None
            else torch.arange(args[0].shape[1]).unsqueeze(0)
        )
        attn_masks_cpu.append(
            kwargs.get("attention_mask").detach().cpu() if kwargs.get("attention_mask") is not None else None
        )
        raise _StopForward()

    h0 = layers[0].register_forward_pre_hook(capture_block0, with_kwargs=True)
    try:
        for start in range(0, len(seqs), batch_size):
            ids = torch.tensor(seqs[start : start + batch_size], dtype=torch.long, device=device)
            try:
                model(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
    finally:
        h0.remove()

    rotary_emb = model.model.rotary_emb if hasattr(model, "model") and hasattr(model.model, "rotary_emb") else None
    projs_by_layer: dict[int, list[EligibleProjection]] = {}
    for p in projections:
        projs_by_layer.setdefault(p.layer_idx, []).append(p)

    for lidx, block in enumerate(layers):
        dev = next(block.parameters()).device
        layer_projs = projs_by_layer[lidx]
        q_weights: dict[str, dict[int, torch.Tensor]] = {
            p.name: {b: quantize_weight_asymmetric(p.module.weight.data, bits=b, group_size=group_size)[0] for b in candidate_bits}
            for p in layer_projs
        }
        handles = []

        def make_hook(proj: EligibleProjection):
            def hook(_mod, inp, out):
                x = inp[0].detach()
                y_ref = out.detach()
                for b in candidate_bits:
                    y_q = F.linear(x, q_weights[proj.name][b].to(x.device))
                    # Note: bias cancels in y_ref_no_bias - y_q_no_bias vs F.linear(x, W) - F.linear(x, W_q)
                    # Wait: out includes bias if mod.bias is not None, so pass mod.bias when computing y_q:
                    if proj.module.bias is not None:
                        y_q = y_q + proj.module.bias
                    d = (y_ref - y_q).float()
                    err_matrix[proj.name][b] += float((d * d).sum().item())

            return hook

        for p in layer_projs:
            handles.append(p.module.register_forward_hook(make_hook(p)))

        next_hidden: list[torch.Tensor] = []
        try:
            for b_i, h_cpu in enumerate(hidden_cpu):
                h_dev = h_cpu.to(dev)
                pids = pos_ids_cpu[b_i].to(dev)
                am = attn_masks_cpu[b_i].to(dev) if attn_masks_cpu[b_i] is not None else None
                pe = rotary_emb(h_dev, pids) if rotary_emb is not None else None
                kw = {"position_ids": pids, "attention_mask": am, "use_cache": False}
                if pe is not None:
                    kw["position_embeddings"] = pe
                raw_out = block(h_dev, **kw)
                out = (raw_out[0] if isinstance(raw_out, tuple) else raw_out).detach().cpu()
                if lidx + 1 < len(layers):
                    next_hidden.append(out)
        finally:
            for h in handles:
                h.remove()
            del q_weights

        hidden_cpu = next_hidden

    return err_matrix


def allocate_mixed_precision_milp(
    projections: list[EligibleProjection],
    distortion_matrix: dict[str, dict[int, float]],
    sensitivity_weights: dict[str, float],
    packed_byte_budget: int,
    uncompressed_fp16_bytes: int,
    candidate_bits: Sequence[int] = (2, 3, 4, 8),
    group_size: int = DEFAULT_GROUP_SIZE,
    unfolded_proj_types: Sequence[str] = ("o_proj",),
) -> tuple[dict[str, int], dict[str, Any]]:
    """Solve exact multiple-choice knapsack (MILP) assigning b_u in {2,3,4,8} per projection
    to minimize sum_u sensitivity_u * distortion_u(b_u) subject to exact total packed bytes <= budget.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_proj = len(projections)
    n_bits = len(candidate_bits)
    n_vars = n_proj * n_bits

    c = np.zeros(n_vars, dtype=np.float64)
    byte_weights = np.zeros(n_vars, dtype=np.float64)

    for i, p in enumerate(projections):
        w_sens = float(sensitivity_weights[p.name])
        has_c = p.proj_type in unfolded_proj_types
        for j, b in enumerate(candidate_bits):
            idx = i * n_bits + j
            c[idx] = w_sens * float(distortion_matrix[p.name][b])
            byte_weights[idx] = float(
                calc_quantized_matrix_bytes(p.d_out, p.d_in, b, group_size, has_c).total_bytes
            )

    # Constraint 1: exactly one bit width selected per projection (sum_j x_{ij} = 1)
    A_one = np.zeros((n_proj, n_vars), dtype=np.float64)
    for i in range(n_proj):
        A_one[i, i * n_bits : (i + 1) * n_bits] = 1.0

    # Constraint 2: total packed bytes <= packed_byte_budget
    elig_budget = float(packed_byte_budget - uncompressed_fp16_bytes)
    A = np.vstack([A_one, byte_weights.reshape(1, -1)])
    lb = np.concatenate([np.ones(n_proj), [-np.inf]])
    ub = np.concatenate([np.ones(n_proj), [elig_budget]])

    res = milp(
        c=c,
        integrality=np.ones(n_vars),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, lb, ub),
    )
    if not res.success:
        raise RuntimeError(f"Mixed-precision MILP failed: {res.message}")

    x_sol = np.round(res.x).astype(int)
    bit_map: dict[str, int] = {}
    used_elig_bytes = 0
    for i, p in enumerate(projections):
        sel = int(np.argmax(x_sol[i * n_bits : (i + 1) * n_bits]))
        b = int(candidate_bits[sel])
        bit_map[p.name] = b
        used_elig_bytes += int(byte_weights[i * n_bits + sel])

    return bit_map, {
        "milp_objective": float(res.fun),
        "budget_total_bytes": int(packed_byte_budget),
        "allocated_total_bytes": int(used_elig_bytes + uncompressed_fp16_bytes),
        "headroom_bytes": int(packed_byte_budget - (used_elig_bytes + uncompressed_fp16_bytes)),
        "bit_histogram": {int(b): sum(1 for v in bit_map.values() if v == b) for b in candidate_bits},
    }
