"""Model loading, eligible projection enumeration, and tied-weight accounting for Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from src.task1.packing import calc_sparse_matrix_bytes

ELIGIBLE_PROJ_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class EligibleProjection:
    name: str          # e.g., "model.layers.0.self_attn.q_proj"
    layer_idx: int
    proj_type: str     # one of ELIGIBLE_PROJ_NAMES
    module: nn.Linear

    @property
    def d_out(self) -> int:
        return int(self.module.out_features)

    @property
    def d_in(self) -> int:
        return int(self.module.in_features)

    @property
    def numel(self) -> int:
        return int(self.module.weight.numel())


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Could not locate decoder layers on model")


def enumerate_eligible_projections(model: nn.Module) -> list[EligibleProjection]:
    """Return the exact ordered list of eligible Linear projections across all Transformer blocks.

    Asserts that no other nn.Linear layers inside Transformer blocks are silently omitted.
    """
    layers = get_transformer_layers(model)
    projections: list[EligibleProjection] = []
    eligible_ids: set[int] = set()

    for idx, block in enumerate(layers):
        attn = block.self_attn
        mlp = block.mlp
        mapping = {
            "q_proj": (f"model.layers.{idx}.self_attn.q_proj", attn.q_proj),
            "k_proj": (f"model.layers.{idx}.self_attn.k_proj", attn.k_proj),
            "v_proj": (f"model.layers.{idx}.self_attn.v_proj", attn.v_proj),
            "o_proj": (f"model.layers.{idx}.self_attn.o_proj", attn.o_proj),
            "gate_proj": (f"model.layers.{idx}.mlp.gate_proj", mlp.gate_proj),
            "up_proj": (f"model.layers.{idx}.mlp.up_proj", mlp.up_proj),
            "down_proj": (f"model.layers.{idx}.mlp.down_proj", mlp.down_proj),
        }
        for proj_type in ELIGIBLE_PROJ_NAMES:
            full_name, mod = mapping[proj_type]
            if not isinstance(mod, nn.Linear):
                raise TypeError(f"Expected nn.Linear at {full_name}, got {type(mod)}")
            projections.append(
                EligibleProjection(name=full_name, layer_idx=idx, proj_type=proj_type, module=mod)
            )
            eligible_ids.add(id(mod))

        # Verify no extra nn.Linear modules exist inside the decoder block
        for sub_name, sub_mod in block.named_modules():
            if isinstance(sub_mod, nn.Linear) and id(sub_mod) not in eligible_ids:
                raise RuntimeError(f"Unexpected nn.Linear inside block {idx}: {sub_name}")

    return projections


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (int(tensor.untyped_storage().data_ptr()), int(tensor.storage_offset()), tuple(tensor.shape))


def compute_unique_parameter_accounting(model: nn.Module) -> dict[str, int]:
    """Count unique parameters and FP16 bytes, deduplicating tied/shared weights.

    Separates eligible projection weights from excluded uncompressed parameters
    (embeddings, norms, biases, and untied lm_head).
    """
    projections = enumerate_eligible_projections(model)
    eligible_weight_ids = {id(p.module.weight) for p in projections}
    eligible_storage_keys = {_tensor_storage_key(p.module.weight) for p in projections}

    seen_storage: set[tuple[int, int, tuple[int, ...]]] = set()
    eligible_params = 0
    uncompressed_params = 0
    tied_duplicates = 0
    tied_duplicate_params = 0
    breakdown = {
        "embeddings": 0,
        "norms": 0,
        "biases": 0,
        "lm_head_untied": 0,
        "other_uncompressed": 0,
    }

    for name, param in model.named_parameters(remove_duplicate=False):
        key = _tensor_storage_key(param)
        if key in seen_storage:
            tied_duplicates += 1
            tied_duplicate_params += int(param.numel())
            continue
        seen_storage.add(key)

        n = int(param.numel())
        if id(param) in eligible_weight_ids or key in eligible_storage_keys:
            eligible_params += n
        else:
            uncompressed_params += n
            if "embed" in name:
                breakdown["embeddings"] += n
            elif "norm" in name:
                breakdown["norms"] += n
            elif name.endswith(".bias"):
                breakdown["biases"] += n
            elif "lm_head" in name:
                breakdown["lm_head_untied"] += n
            else:
                breakdown["other_uncompressed"] += n

    total_unique = eligible_params + uncompressed_params
    return {
        "num_layers": len(get_transformer_layers(model)),
        "num_eligible_matrices": len(projections),
        "eligible_params": eligible_params,
        "uncompressed_params": uncompressed_params,
        "total_unique_params": total_unique,
        "tied_duplicate_tensors": tied_duplicates,
        "tied_duplicate_params": tied_duplicate_params,
        "baseline_fp16_bytes": total_unique * 2,
        "eligible_fp16_bytes": eligible_params * 2,
        "uncompressed_fp16_bytes": uncompressed_params * 2,
        **{f"uncompressed_{k}_params": v for k, v in breakdown.items()},
    }


def compute_pruned_model_storage(model: nn.Module) -> dict[str, float | int]:
    """Measure exact nonzero counts, achieved sparsity, and 1-bit-mask+FP16 sparse bytes."""
    acct = compute_unique_parameter_accounting(model)
    projections = enumerate_eligible_projections(model)

    total_elig = 0
    total_elig_nnz = 0
    total_mask_bytes = 0
    total_val_bytes = 0
    per_layer_stats = []

    for p in projections:
        w = p.module.weight.data
        numel = int(w.numel())
        nnz = int(torch.count_nonzero(w).item())
        mask_b, val_b, tot_b = calc_sparse_matrix_bytes(p.d_out, p.d_in, nnz, align_per_row=False)
        total_elig += numel
        total_elig_nnz += nnz
        total_mask_bytes += mask_b
        total_val_bytes += val_b
        per_layer_stats.append({
            "name": p.name,
            "layer_idx": p.layer_idx,
            "proj_type": p.proj_type,
            "numel": numel,
            "nnz": nnz,
            "sparsity": 1.0 - (nnz / numel),
            "mask_bytes": mask_b,
            "value_bytes": val_b,
        })

    if total_elig != acct["eligible_params"]:
        raise RuntimeError(f"Eligible mismatch: {total_elig} != {acct['eligible_params']}")

    uncompressed_bytes = acct["uncompressed_fp16_bytes"]
    sparse_total_bytes = total_mask_bytes + total_val_bytes + uncompressed_bytes
    nonzero_frac = total_elig_nnz / total_elig
    achieved_sparsity = 1.0 - nonzero_frac
    # Effective nominal bits per eligible weight (1 mask bit + 16*nonzero_frac)
    nominal_bits_per_elig = (total_mask_bytes + total_val_bytes) * 8.0 / total_elig

    return {
        "eligible_params": total_elig,
        "eligible_nnz": total_elig_nnz,
        "eligible_zeros": total_elig - total_elig_nnz,
        "eligible_nonzero_fraction": nonzero_frac,
        "achieved_eligible_sparsity": achieved_sparsity,
        "total_unique_params": acct["total_unique_params"],
        "total_unique_nnz": total_elig_nnz + acct["uncompressed_params"],
        "dense_fp16_bytes": acct["baseline_fp16_bytes"],
        "sparse_mask_bytes": total_mask_bytes,
        "sparse_value_bytes": total_val_bytes,
        "uncompressed_fp16_bytes": uncompressed_bytes,
        "sparse_total_bytes": sparse_total_bytes,
        "sparse_compression_ratio": acct["baseline_fp16_bytes"] / sparse_total_bytes,
        "nominal_bits_per_eligible_weight": nominal_bits_per_elig,
        "per_projection": per_layer_stats,
    }


def load_qwen_base_fp16(
    model_id: str = "Qwen/Qwen2.5-1.5B",
    revision: Optional[str] = None,
    device: str | torch.device = "cuda",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = model.to(device).eval()
    return model, tok

