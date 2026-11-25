"""Task 3 evaluation: prefix banks, distribution metrics, speculative benchmarks, and standalone accuracy."""

from __future__ import annotations

import gzip
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.common.env import measure_cuda_latency_ms
from src.task3.distill import format_chat_prompt_ids
from src.task3.speculative import (
    compute_analytical_speculative_model,
    speculative_generate,
    target_only_generate,
)


def load_task3_test_prompts(dataset_root: Path) -> list[dict[str, Any]]:
    pub = dataset_root / "data" / "public_locked"
    prompts: list[dict[str, Any]] = []
    with gzip.open(pub / "gsm8k_test_locked.jsonl.gz", "rt", encoding="utf-8") as f:
        for r in f:
            if not r.strip():
                continue
            obj = json.loads(r)
            gold_ans = extract_gsm8k_number(obj["example"]["answer"])
            prompts.append({
                "prompt_id": f"GSM8K-{obj['order']:04d}",
                "domain": "math",
                "subsplit": "math",
                "family": None,
                "prompt": obj["example"]["question"],
                "gold_answer": gold_ans,
            })

    with open(dataset_root / "data" / "short_form" / "short_test.jsonl", "r", encoding="utf-8") as f:
        for r in f:
            if not r.strip():
                continue
            obj = json.loads(r)
            prompts.append({
                "prompt_id": obj["id"],
                "domain": "short",
                "subsplit": f"short_{obj['split']}",  # short_familiar or short_shifted
                "family": obj["family"],
                "prompt": obj["input"],
                "gold_answer": obj["answer"],
            })
    return prompts


def load_prefix_bank_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _sample_bank_positions(pid: str, bank: str, seq_len: int) -> list[int]:
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    positions: list[int] = []
    for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
        cands = list(range(lo, min(hi, seq_len) + 1))
        positions.extend(sorted(rng.sample(cands, min(n, len(cands)))))
    return positions


@torch.inference_mode()
def generate_student_or_spec_prefix_bank(
    bank_name: str,
    prompts: list[dict[str, Any]],
    tokenizer,
    draft_model,
    target_model=None,
    gamma: int = 4,
    use_speculative: bool = False,
    seed_offset: int = 0,
    device: torch.device = torch.device("cuda"),
) -> list[dict[str, Any]]:
    """Generate frozen response-prefix bank matching setup_data.py sampling conventions."""
    records: list[dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id
    draft_model.eval()
    if target_model is not None:
        target_model.eval()

    for j, item in enumerate(prompts):
        pid = item["prompt_id"]
        p_ids = format_chat_prompt_ids(tokenizer, item["prompt"])
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        gen_seed = 6304 + 10007 * j + seed_offset

        if use_speculative:
            assert target_model is not None
            out = speculative_generate(
                target_model, draft_model, p_tensor,
                max_new_tokens=192, gamma=gamma, eos_token_id=eos_id,
                temperature=1.0, top_p=1.0, seed=gen_seed,
            )
            seq = out["generated_ids"]
        else:
            gen = torch.Generator(device=device).manual_seed(gen_seed)
            seq = draft_model.generate(
                input_ids=p_tensor,
                max_new_tokens=192,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                generator=gen,
                pad_token_id=eos_id,
                eos_token_id=eos_id,
            )[0, len(p_ids) :].tolist()

        for pos in _sample_bank_positions(pid, bank_name, len(seq)):
            records.append({
                "prompt_id": pid,
                "domain": item["domain"],
                "subsplit": item["subsplit"],
                "bank": bank_name,
                "position": pos,
                "prompt": item["prompt"],
                "response_prefix_token_ids": seq[: pos - 1],
                "source_length": len(seq),
                "source_generation_seed": gen_seed,
            })
    return records


def _pos_bin(pos: int) -> str:
    if pos <= 16:
        return "[1,16]"
    if pos <= 32:
        return "[17,32]"
    if pos <= 64:
        return "[33,64]"
    return "[65,192]"


@torch.inference_mode()
def evaluate_distribution_alignment_on_banks(
    bank_records: list[dict[str, Any]],
    prompt_subsplit_map: dict[str, str],
    teacher_model,
    draft_models: dict[str, Any],
    tokenizer,
    device: torch.device,
    max_disagreements: int = 12,
) -> dict[str, Any]:
    """Compute forward KL, reverse KL, TV, greedy agreement, and 1-TV on common prefix banks.

    Groups prefixes belonging to the same (bank, prompt_id) and evaluates teacher/draft logits
    in a single forward pass per sequence whenever prefixes form a nested prefix chain.
    """
    teacher_model.eval()
    for m in draft_models.values():
        m.eval()

    # Group by (bank, prompt_id)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in bank_records:
        grouped.setdefault((rec["bank"], rec["prompt_id"]), []).append(rec)

    # Accumulators keyed by (draft_name, bank, subsplit, pos_bin)
    stats: dict[tuple[str, str, str, str], list[float]] = {}  # [sum_fkl, sum_rkl, sum_tv, sum_acc, sum_greedy, count]
    disagreements: list[dict[str, Any]] = []

    def _add(draft_name: str, bank: str, subsplit: str, pbin: str, fkl: float, rkl: float, tv: float, greedy: float):
        key = (draft_name, bank, subsplit, pbin)
        acc = stats.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        acc[0] += fkl
        acc[1] += rkl
        acc[2] += tv
        acc[3] += (1.0 - tv)
        acc[4] += greedy
        acc[5] += 1.0

    for (bank, pid), recs in grouped.items():
        recs.sort(key=lambda r: r["position"])
        max_pref = recs[-1]["response_prefix_token_ids"]
        is_nested = all(r["response_prefix_token_ids"] == max_pref[: len(r["response_prefix_token_ids"])] for r in recs)
        p_ids = format_chat_prompt_ids(tokenizer, recs[0]["prompt"])
        L_p = len(p_ids)
        subsplit = prompt_subsplit_map.get(pid, recs[0].get("domain", "unknown"))

        if is_nested:
            full_seq = p_ids + max_pref
            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            # Token at response position `pos` (1-indexed) is predicted at index (L_p - 1) + (pos - 1)
            indices = [L_p - 1 + len(r["response_prefix_token_ids"]) for r in recs]
            t_logits = teacher_model(input_ids=inp, use_cache=False).logits[0, indices, :].float()
            log_pT = F.log_softmax(t_logits, dim=-1)
            pT = log_pT.exp()
            t_top = t_logits.argmax(dim=-1)

            for d_name, d_model in draft_models.items():
                d_logits = d_model(input_ids=inp, use_cache=False).logits[0, indices, :].float()
                log_qS = F.log_softmax(d_logits, dim=-1)
                qS = log_qS.exp()
                d_top = d_logits.argmax(dim=-1)

                fkl_vec = (pT * (log_pT - log_qS)).sum(dim=-1)
                rkl_vec = (qS * (log_qS - log_pT)).sum(dim=-1)
                tv_vec = 0.5 * (pT - qS).abs().sum(dim=-1)
                greedy_vec = (t_top == d_top).float()

                for j, r in enumerate(recs):
                    pbin = _pos_bin(int(r["position"]))
                    fkl_v = float(fkl_vec[j].item())
                    rkl_v = float(rkl_vec[j].item())
                    tv_v = float(tv_vec[j].item())
                    gr_v = float(greedy_vec[j].item())
                    _add(d_name, bank, subsplit, pbin, fkl_v, rkl_v, tv_v, gr_v)

                    if d_name == "original" and bank == "teacher" and len(disagreements) < max_disagreements and tv_v > 0.5:
                        t_tok_id = int(t_top[j].item())
                        d_tok_id = int(d_top[j].item())
                        disagreements.append({
                            "prompt_id": pid,
                            "subsplit": subsplit,
                            "bank": bank,
                            "position": int(r["position"]),
                            "teacher_top_token": tokenizer.decode([t_tok_id]),
                            "orig_draft_top_token": tokenizer.decode([d_tok_id]),
                            "teacher_prob": float(pT[j, t_tok_id].item()),
                            "orig_draft_prob_on_teacher_top": float(qS[j, t_tok_id].item()),
                            "orig_tv": tv_v,
                        })
        else:
            for r in recs:
                full_seq = p_ids + r["response_prefix_token_ids"]
                inp = torch.tensor([full_seq], dtype=torch.long, device=device)
                t_l = teacher_model(input_ids=inp, use_cache=False).logits[0, -1, :].float()
                log_pT = F.log_softmax(t_l, dim=-1)
                pT = log_pT.exp()
                t_top = int(t_l.argmax().item())
                pbin = _pos_bin(int(r["position"]))
                for d_name, d_model in draft_models.items():
                    d_l = d_model(input_ids=inp, use_cache=False).logits[0, -1, :].float()
                    log_qS = F.log_softmax(d_l, dim=-1)
                    qS = log_qS.exp()
                    d_top = int(d_l.argmax().item())
                    fkl_v = float((pT * (log_pT - log_qS)).sum().item())
                    rkl_v = float((qS * (log_qS - log_pT)).sum().item())
                    tv_v = float((0.5 * (pT - qS).abs().sum()).item())
                    _add(d_name, bank, subsplit, pbin, fkl_v, rkl_v, tv_v, float(t_top == d_top))

    rows = []
    for (d_name, bank, subsplit, pbin), (sf, sr, stv, sacc, sg, cnt) in sorted(stats.items()):
        rows.append({
            "draft": d_name,
            "bank": bank,
            "subsplit": subsplit,
            "pos_bin": pbin,
            "count": int(cnt),
            "forward_kl": sf / cnt,
            "reverse_kl": sr / cnt,
            "tv": stv / cnt,
            "expected_acceptance": sacc / cnt,
            "greedy_agreement": sg / cnt,
        })
    return {"metrics_by_cell": rows, "sample_disagreements": disagreements}


@torch.inference_mode()
def benchmark_isolated_step_costs(
    target_model,
    draft_model,
    prompt_lengths: tuple[int, ...] = (64, 128, 256),
    gammas: tuple[int, ...] = (2, 4),
    device: torch.device = torch.device("cuda"),
    repeats: int = 10,
) -> dict[str, Any]:
    """Measure isolated draft step, target single step, and (gamma+1)-token verify step across prompt lengths."""
    out_rows = []
    for L_p in prompt_lengths:
        ids = torch.randint(100, 10000, (1, L_p), dtype=torch.long, device=device)
        t_pref = target_model(input_ids=ids, use_cache=True).past_key_values
        d_pref = draft_model(input_ids=ids, use_cache=True).past_key_values
        tok1 = torch.tensor([[100]], dtype=torch.long, device=device)

        def _clone_crop_step(model, base_cache, n_toks: int):
            from transformers import DynamicCache

            c = DynamicCache()
            for lidx, layer in enumerate(base_cache.layers):
                c.update(layer.keys.clone(), layer.values.clone(), lidx)
            q = torch.full((1, n_toks), 100, dtype=torch.long, device=device)
            return model(input_ids=q, past_key_values=c, use_cache=True)

        d_lat = measure_cuda_latency_ms(lambda: _clone_crop_step(draft_model, d_pref, 1), warmup=3, repeats=repeats, device=device)
        t_lat = measure_cuda_latency_ms(lambda: _clone_crop_step(target_model, t_pref, 1), warmup=3, repeats=repeats, device=device)

        verify_lats = {}
        for g in gammas:
            # Verify pass feeds [pending_tok, x_1..x_g] = g+1 tokens (or g tokens); report exact (g+1) pass ratio
            v_lat = measure_cuda_latency_ms(
                lambda g_len=g + 1: _clone_crop_step(target_model, t_pref, g_len),
                warmup=3,
                repeats=repeats,
                device=device,
            )
            verify_lats[str(g)] = {
                **v_lat,
                "v_gamma": v_lat["mean_ms"] / t_lat["mean_ms"],
            }

        c_ratio = d_lat["mean_ms"] / t_lat["mean_ms"]
        out_rows.append({
            "prompt_len": L_p,
            "draft_step_ms": d_lat["mean_ms"],
            "target_step_ms": t_lat["mean_ms"],
            "c_ratio": c_ratio,
            "verify_passes": verify_lats,
        })
    return {"by_prompt_len": out_rows}


@torch.inference_mode()
def benchmark_speculative_120_prompts(
    draft_name: str,
    draft_model,
    target_model,
    tokenizer,
    prompts: list[dict[str, Any]],
    device: torch.device,
    gammas: tuple[int, ...] = (2, 4),
    repeats: int = 2,
    max_new_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Run counterbalanced 120-prompt EOS-aware speculative vs adjacent target-only benchmark."""
    draft_model.eval()
    target_model.eval()
    eos_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []

    # Warm up both models once on prompt 0
    warm_ids = torch.tensor([format_chat_prompt_ids(tokenizer, prompts[0]["prompt"])], dtype=torch.long, device=device)
    _ = target_only_generate(target_model, warm_ids, max_new_tokens=8, eos_token_id=eos_id, seed=6304)
    _ = speculative_generate(target_model, draft_model, warm_ids, max_new_tokens=8, gamma=2, eos_token_id=eos_id, seed=6304)

    for gamma in gammas:
        for rep in range(repeats):
            for j, item in enumerate(prompts):
                p_ids = format_chat_prompt_ids(tokenizer, item["prompt"])
                p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
                run_seed = 6304 + 10007 * j + 1009 * rep + 31 * gamma

                # Counterbalance execution order across reps
                if (j + rep) % 2 == 0:
                    t_res = target_only_generate(
                        target_model, p_tensor, max_new_tokens=max_new_tokens, eos_token_id=eos_id, seed=run_seed
                    )
                    s_res = speculative_generate(
                        target_model, draft_model, p_tensor, max_new_tokens=max_new_tokens, gamma=gamma, eos_token_id=eos_id, seed=run_seed
                    )
                else:
                    s_res = speculative_generate(
                        target_model, draft_model, p_tensor, max_new_tokens=max_new_tokens, gamma=gamma, eos_token_id=eos_id, seed=run_seed
                    )
                    t_res = target_only_generate(
                        target_model, p_tensor, max_new_tokens=max_new_tokens, eos_token_id=eos_id, seed=run_seed
                    )

                rows.append({
                    "draft": draft_name,
                    "gamma": gamma,
                    "rep": rep,
                    "prompt_id": item["prompt_id"],
                    "domain": item["domain"],
                    "subsplit": item["subsplit"],
                    "prompt_tokens": len(p_ids),
                    "target_only": {k: v for k, v in t_res.items() if k != "generated_ids"},
                    "speculative": {k: v for k, v in s_res.items() if k not in ("generated_ids", "round_traces")},
                    "speedup_total": t_res["total_ms"] / max(s_res["total_ms"], 1e-6),
                    "speedup_decode": t_res["decode_ms"] / max(s_res["decode_ms"], 1e-6),
                    "target_text": tokenizer.decode(t_res["generated_ids"], skip_special_tokens=True) if rep == 0 and j < 10 else None,
                    "spec_text": tokenizer.decode(s_res["generated_ids"], skip_special_tokens=True) if rep == 0 and j < 10 else None,
                })
    return rows


# ---------------------------------------------------------------------------
# Answer Extraction & Standalone Evaluation (Step 7)
# ---------------------------------------------------------------------------

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def extract_gsm8k_number(text: str) -> Optional[str]:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m_box = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m_box:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", m_box[-1])
        if nums:
            return nums[-1].replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "").strip() if nums else None


def check_short_compliance(pred_text: str, family: Optional[int]) -> bool:
    s = pred_text.strip()
    if family == 0:
        return s.lower() in MONTHS
    if family == 1:
        return bool(re.fullmatch(r"-?\d+", s))
    if family == 2:
        return bool(re.fullmatch(r"[A-Z]{2}", s))
    return len(s) > 0


@torch.inference_mode()
def evaluate_standalone_accuracy(
    model_name: str,
    model,
    tokenizer,
    prompts: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 12,
) -> list[dict[str, Any]]:
    """Evaluate standalone model answer correctness with domain caps (512 math, 32 short)."""
    model.eval()
    eos_id = tokenizer.eos_token_id
    prev_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    rows_by_pid: dict[str, dict[str, Any]] = {}
    try:
        for domain, cap in (("math", 512), ("short", 32)):
            sub = [p for p in prompts if p["domain"] == domain]
            for start in range(0, len(sub), batch_size):
                chunk = sub[start : start + batch_size]
                p_lists = [format_chat_prompt_ids(tokenizer, x["prompt"]) for x in chunk]
                max_l = max(len(x) for x in p_lists)
                inp = torch.full((len(chunk), max_l), eos_id, dtype=torch.long, device=device)
                amask = torch.zeros((len(chunk), max_l), dtype=torch.long, device=device)
                for i, seq in enumerate(p_lists):
                    L = len(seq)
                    inp[i, max_l - L :] = torch.tensor(seq, dtype=torch.long, device=device)
                    amask[i, max_l - L :] = 1

                outs = model.generate(
                    input_ids=inp,
                    attention_mask=amask,
                    max_new_tokens=cap,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                )
                for i, item in enumerate(chunk):
                    raw_gen = outs[i, max_l:].tolist()
                    out_seq = []
                    hit_eos = False
                    for tid in raw_gen:
                        out_seq.append(tid)
                        if tid == eos_id:
                            hit_eos = True
                            break
                    truncated = (not hit_eos) and (len(out_seq) >= cap)
                    text = tokenizer.decode(out_seq, skip_special_tokens=True).strip()

                    if domain == "math":
                        pred = extract_gsm8k_number(text)
                        unextractable = pred is None
                        try:
                            correct = int(pred is not None and abs(float(pred) - float(item["gold_answer"])) < 1e-4)
                        except ValueError:
                            correct = 0
                        compliant = not unextractable
                    else:
                        pred = text
                        unextractable = len(text) == 0
                        correct = int(text.strip() == str(item["gold_answer"]).strip())
                        compliant = check_short_compliance(text, item.get("family"))

                    rows_by_pid[item["prompt_id"]] = {
                        "model": model_name,
                        "prompt_id": item["prompt_id"],
                        "domain": item["domain"],
                        "subsplit": item["subsplit"],
                        "n_generated": len(out_seq),
                        "truncated": truncated,
                        "unextractable": unextractable,
                        "format_compliant": compliant,
                        "correct": correct,
                        "gold": item["gold_answer"],
                        "pred": pred,
                        "text": text,
                    }
    finally:
        tokenizer.padding_side = prev_pad_side

    return [rows_by_pid[p["prompt_id"]] for p in prompts]
