"""Locked-window perplexity evaluation for Task 1 (WikiText-2 test & C4 validation)."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_locked_windows(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def evaluate_perplexity_windows(
    model: torch.nn.Module,
    windows: list[dict[str, Any]],
    device: str | torch.device = "cuda",
    batch_size: int = 8,
    expected_total_tokens: int | None = 50000,
) -> dict[str, Any]:
    """Evaluate exact non-overlapping window NLL and perplexity.

    Groups sequences of equal length into batches (no padding needed, exact causal match)
    and computes FP32 cross-entropy over `score_target_offsets`.
    """
    model.eval()
    # Preserve original order indices while batching identical lengths
    indexed = list(enumerate(windows))
    groups: dict[tuple[int, int, int], list[tuple[int, dict[str, Any]]]] = {}
    for idx, rec in indexed:
        seq_len = len(rec["token_ids"])
        s0, s1 = rec["score_target_offsets"]
        groups.setdefault((seq_len, s0, s1), []).append((idx, rec))

    per_window_nll: list[float] = [0.0] * len(windows)
    per_window_tokens: list[int] = [0] * len(windows)

    for (seq_len, s0, s1), items in groups.items():
        if s0 < 1 or s1 > seq_len or s1 <= s0:
            raise ValueError(f"Invalid score_target_offsets {[s0, s1]} for len {seq_len}")
        n_scored = s1 - s0
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch_ids = torch.tensor(
                [rec["token_ids"] for _, rec in chunk],
                dtype=torch.long,
                device=device,
            )
            outputs = model(input_ids=batch_ids, use_cache=False)
            # Logits at position t predict token t+1
            logits_slice = outputs.logits[:, s0 - 1 : s1 - 1, :].float()
            targets = batch_ids[:, s0:s1]
            losses = F.cross_entropy(
                logits_slice.reshape(-1, logits_slice.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(len(chunk), n_scored)
            sums = losses.sum(dim=1).tolist()
            for (orig_idx, rec), nll_sum in zip(chunk, sums):
                if rec["scored_tokens"] != n_scored:
                    raise ValueError(f"scored_tokens mismatch: {rec['scored_tokens']} != {n_scored}")
                per_window_nll[orig_idx] = float(nll_sum)
                per_window_tokens[orig_idx] = int(n_scored)

    total_tokens = int(sum(per_window_tokens))
    if expected_total_tokens is not None and total_tokens != expected_total_tokens:
        raise ValueError(f"Expected {expected_total_tokens} scored tokens, got {total_tokens}")
    total_nll = float(sum(per_window_nll))
    mean_nll = total_nll / total_tokens
    ppl = float(math.exp(mean_nll))

    return {
        "n_windows": len(windows),
        "scored_tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "per_window_nll_sum": per_window_nll,
        "per_window_scored_tokens": per_window_tokens,
    }

