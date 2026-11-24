"""Zero-shot HellaSwag evaluation following lm-evaluation-harness conventions."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HELLASWAG_PROMPT_FORMAT = "lm_eval_v0.4_activity_colon_ctx_space_ending"


def preprocess_hellaswag_text(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def format_hellaswag_example(record: dict[str, Any]) -> dict[str, Any]:
    ex = record["example"]
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    context = preprocess_hellaswag_text(ex["activity_label"] + ": " + ctx)
    endings = [preprocess_hellaswag_text(e) for e in ex["endings"]]
    gold = int(ex["label"])
    return {
        "order": int(record["order"]),
        "source_row_index": int(record["source_row_index"]),
        "ind": ex.get("ind"),
        "activity_label": ex["activity_label"],
        "context": context,
        "endings": endings,
        "gold": gold,
    }


def load_hellaswag_examples(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(format_hellaswag_example(json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def encode_context_continuation(tokenizer, context: str, ending: str) -> tuple[list[int], int]:
    """Encode `context` and `context + ' ' + ending`, returning `(full_ids, ctx_len)`.

    Mirrors lm-evaluation-harness `_encode_pair` with single-space delimiter.
    """
    continuation = " " + ending
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    if len(ctx_ids) == 0:
        raise ValueError("Empty context encoding encountered in HellaSwag")
    if len(whole_ids) <= len(ctx_ids):
        # Edge case if BPE merged across boundary; fall back safely
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        whole_ids = ctx_ids + cont_ids
    return whole_ids, len(ctx_ids)


@torch.inference_mode()
def evaluate_hellaswag(
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict[str, Any]],
    device: str | torch.device = "cuda",
    batch_size_candidates: int = 32,
) -> dict[str, Any]:
    """Score all 4 candidate endings zero-shot by conditional log-likelihood.

    Reports raw accuracy (`acc`) and character-length-normalized accuracy (`acc_norm`).
    """
    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Flatten 4 candidates per example
    items: list[tuple[int, int, list[int], int, int]] = []
    for ex_idx, ex in enumerate(examples):
        for cand_idx, ending in enumerate(ex["endings"]):
            full_ids, ctx_len = encode_context_continuation(tokenizer, ex["context"], ending)
            items.append((ex_idx, cand_idx, full_ids, ctx_len, len(ending)))

    # Sort by length to minimize padding overhead, restore by (ex_idx, cand_idx)
    items.sort(key=lambda x: len(x[2]))
    raw_lls: list[list[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(len(examples))]
    char_lens: list[list[int]] = [[0, 0, 0, 0] for _ in range(len(examples))]

    for start in range(0, len(items), batch_size_candidates):
        chunk = items[start : start + batch_size_candidates]
        max_len = max(len(x[2]) for x in chunk)
        bsz = len(chunk)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for row_i, (_, _, full_ids, _, _) in enumerate(chunk):
            L = len(full_ids)
            input_ids[row_i, :L] = torch.tensor(full_ids, dtype=torch.long, device=device)
            attn_mask[row_i, :L] = 1

        logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        # Shift for next-token log-probabilities in FP32
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        tok_lls = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        for row_i, (ex_idx, cand_idx, full_ids, ctx_len, clen) in enumerate(chunk):
            L = len(full_ids)
            # Tokens at positions ctx_len .. L-1 are predicted by logits at ctx_len-1 .. L-2
            cand_ll = float(tok_lls[row_i, ctx_len - 1 : L - 1].sum().item())
            raw_lls[ex_idx][cand_idx] = cand_ll
            char_lens[ex_idx][cand_idx] = int(clen)

    per_example = []
    correct_raw = []
    correct_norm = []
    for ex_idx, ex in enumerate(examples):
        lls = raw_lls[ex_idx]
        clens = char_lens[ex_idx]
        norm_lls = [ll / max(cl, 1) for ll, cl in zip(lls, clens)]
        pred_raw = int(max(range(4), key=lambda k: lls[k]))
        pred_norm = int(max(range(4), key=lambda k: norm_lls[k]))
        gold = int(ex["gold"])
        ok_raw = int(pred_raw == gold)
        ok_norm = int(pred_norm == gold)
        correct_raw.append(ok_raw)
        correct_norm.append(ok_norm)
        per_example.append({
            "order": ex["order"],
            "source_row_index": ex["source_row_index"],
            "gold": gold,
            "raw_lls": lls,
            "char_lens": clens,
            "norm_lls": norm_lls,
            "pred_raw": pred_raw,
            "pred_norm": pred_norm,
            "correct_raw": ok_raw,
            "correct_norm": ok_norm,
        })

    n = len(examples)
    return {
        "prompt_format": HELLASWAG_PROMPT_FORMAT,
        "n_examples": n,
        "acc": float(sum(correct_raw) / n),
        "acc_norm": float(sum(correct_norm) / n),
        "correct_raw": correct_raw,
        "correct_norm": correct_norm,
        "per_example": per_example,
    }

