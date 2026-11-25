"""Task 3: Off-policy and on-policy LoRA forward-KL distillation with exact token quotas."""

from __future__ import annotations

import gzip
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from src.common.env import set_seed

MATH_QUOTA = 30000
SHORT_QUOTA = 10000
MATH_CAP = 192
SHORT_CAP = 32
ACCUM_TOKENS = 256
LR = 3e-4
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_task3_training_prompts(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pub = dataset_root / "data" / "public_locked"
    with gzip.open(pub / "gsm8k_train_locked.jsonl.gz", "rt", encoding="utf-8") as f:
        gsm_rows = [json.loads(l) for l in f if l.strip()]
    math_prompts = [
        {"prompt_id": f"GSM8K-train-{r['order']:04d}", "domain": "math", "prompt": r["example"]["question"]}
        for r in gsm_rows
    ]
    with open(dataset_root / "data" / "short_form" / "short_train.jsonl", "r", encoding="utf-8") as f:
        short_rows = [json.loads(l) for l in f if l.strip()]
    short_prompts = [
        {"prompt_id": r["id"], "domain": "short", "prompt": r["input"]}
        for r in short_rows
    ]
    return math_prompts, short_prompts


def format_chat_prompt_ids(tokenizer, prompt_text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
    )


def build_lora_student(
    draft_model_id: str,
    draft_revision: str | None,
    device: torch.device,
    seed: int,
) -> tuple[PeftModel, dict[str, int]]:
    """Load fresh FP16 0.5B backbone, attach r=8 alpha=16 FP32 LoRA adapters."""
    set_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        revision=draft_revision,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    for p in base.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)
    trainable_params = 0
    frozen_params = 0
    for name, p in peft_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # FP32 adapters for AMP + FP32 optimizer states
            trainable_params += int(p.numel())
        else:
            frozen_params += int(p.numel())

    return peft_model, {
        "frozen_backbone_params": frozen_params,
        "trainable_lora_params": trainable_params,
        "total_params": frozen_params + trainable_params,
    }


def merge_and_verify_lora_adapter(
    adapter_dir: Path,
    draft_model_id: str,
    draft_revision: str | None,
    tokenizer,
    verify_prompts: list[str],
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    """Merge saved LoRA adapter into a fresh FP16 0.5B backbone and verify logit equivalence."""
    t0 = time.perf_counter()
    base_for_peft = AutoModelForCausalLM.from_pretrained(
        draft_model_id, revision=draft_revision, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    peft_m = PeftModel.from_pretrained(base_for_peft, str(adapter_dir)).to(device).eval()

    # Check unmerged logits before merging
    batch_ids = [format_chat_prompt_ids(tokenizer, p) for p in verify_prompts[:4]]
    max_l = max(len(x) for x in batch_ids)
    pad_id = tokenizer.eos_token_id
    inp = torch.full((len(batch_ids), max_l), pad_id, dtype=torch.long, device=device)
    amask = torch.zeros((len(batch_ids), max_l), dtype=torch.long, device=device)
    for i, seq in enumerate(batch_ids):
        inp[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        amask[i, : len(seq)] = 1

    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            unmerged_logits = peft_m(input_ids=inp, attention_mask=amask, use_cache=False).logits.float()

    merged = peft_m.merge_and_unload().to(torch.float16).eval()
    merge_sec = time.perf_counter() - t0

    with torch.inference_mode():
        merged_logits = merged(input_ids=inp, attention_mask=amask, use_cache=False).logits.float()

    # Compare only non-padded positions
    valid = amask.bool().unsqueeze(-1).expand_as(unmerged_logits)
    diff = (unmerged_logits - merged_logits).abs()[valid]
    return merged, {
        "merge_time_sec": merge_sec,
        "merged_vs_unmerged_max_abs_diff": float(diff.max().item()),
        "merged_vs_unmerged_mean_abs_diff": float(diff.mean().item()),
    }


@torch.inference_mode()
def prepare_offpolicy_teacher_pool(
    teacher_model,
    tokenizer,
    math_prompts: list[dict[str, Any]],
    short_prompts: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    seed: int = 6304,
) -> dict[str, Any]:
    """Generate frozen teacher responses for all 500 math + 300 short training prompts."""
    teacher_model.eval()
    eos_id = tokenizer.eos_token_id
    prev_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    t0 = time.perf_counter()
    pool: dict[str, list[dict[str, Any]]] = {"math": [], "short": []}

    for domain, items, cap in (("math", math_prompts, MATH_CAP), ("short", short_prompts, SHORT_CAP)):
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            prompt_lists = [format_chat_prompt_ids(tokenizer, x["prompt"]) for x in chunk]
            max_l = max(len(x) for x in prompt_lists)
            bsz = len(chunk)
            inp = torch.full((bsz, max_l), eos_id, dtype=torch.long, device=device)
            amask = torch.zeros((bsz, max_l), dtype=torch.long, device=device)
            for i, seq in enumerate(prompt_lists):
                L = len(seq)
                inp[i, max_l - L :] = torch.tensor(seq, dtype=torch.long, device=device)
                amask[i, max_l - L :] = 1

            rollout_seed = seed + (0 if domain == "math" else 50000) + start
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(rollout_seed)
                out_ids = teacher_model.generate(
                    input_ids=inp,
                    attention_mask=amask,
                    max_new_tokens=cap,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                )
            for i, x in enumerate(chunk):
                resp = out_ids[i, max_l:].tolist()
                # Trim trailing pad/eos past first EOS
                trimmed = []
                term_reason = "length_cap"
                for tid in resp:
                    trimmed.append(tid)
                    if tid == eos_id:
                        term_reason = "eos"
                        break
                pool[domain].append({
                    "prompt_id": x["prompt_id"],
                    "domain": domain,
                    "prompt_ids": prompt_lists[i],
                    "response_ids": trimmed,
                    "termination_reason": term_reason,
                    "rollout_seed": seed + (0 if domain == "math" else 50000) + start,
                })

    tokenizer.padding_side = prev_pad_side
    prep_sec = time.perf_counter() - t0
    return {
        "prep_time_sec": prep_sec,
        "math_total_tokens": sum(len(x["response_ids"]) for x in pool["math"]),
        "short_total_tokens": sum(len(x["response_ids"]) for x in pool["short"]),
        "pool": pool,
    }


def _build_domain_schedule(n_math: int, n_short: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    m_idx = list(range(n_math))
    s_idx = list(range(n_short))
    rng.shuffle(m_idx)
    rng.shuffle(s_idx)
    return m_idx, s_idx


def run_distillation(
    mode: str,
    seed: int,
    teacher_model,
    tokenizer,
    math_prompts: list[dict[str, Any]],
    short_prompts: list[dict[str, Any]],
    offpolicy_pool: dict[str, Any] | None,
    draft_model_id: str,
    draft_revision: str | None,
    save_adapter_dir: Path,
    device: torch.device,
    math_quota: int = MATH_QUOTA,
    short_quota: int = SHORT_QUOTA,
) -> dict[str, Any]:
    """Run off-policy or on-policy LoRA distillation to exact 30k math + 10k short teacher-scored tokens."""
    assert mode in ("offpolicy", "onpolicy")
    if mode == "offpolicy" and offpolicy_pool is None:
        raise ValueError("offpolicy_pool is required for off-policy distillation")

    student, param_counts = build_lora_student(draft_model_id, draft_revision, device, seed)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    eos_id = tokenizer.eos_token_id
    m_order, s_order = _build_domain_schedule(len(math_prompts), len(short_prompts), seed)
    m_ptr, s_ptr = 0, 0
    math_rem, short_rem = math_quota, short_quota
    scored_total = 0
    step_count = 0
    skipped_steps = 0

    accum_tokens = 0
    accum_kl_sum = 0.0
    optimizer.zero_grad(set_to_none=True)

    teacher_score_sec = 0.0
    student_rollout_sec = 0.0
    student_bwd_sec = 0.0

    rollout_logs: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_run_start = time.perf_counter()

    # Interleave 3 math : 1 short while both quotas remain
    turn_counter = 0
    while math_rem > 0 or short_rem > 0:
        if math_rem > 0 and (short_rem == 0 or (turn_counter % 4 != 3)):
            domain = "math"
            idx = m_order[m_ptr % len(m_order)]
            m_ptr += 1
            cap = MATH_CAP
            dom_rem = math_rem
        else:
            domain = "short"
            idx = s_order[s_ptr % len(s_order)]
            s_ptr += 1
            cap = SHORT_CAP
            dom_rem = short_rem
        turn_counter += 1

        if mode == "offpolicy":
            entry = offpolicy_pool["pool"][domain][idx]
            prompt_id = entry["prompt_id"]
            p_ids = entry["prompt_ids"]
            raw_resp = entry["response_ids"]
            term_reason = entry["termination_reason"]
            r_seed = entry["rollout_seed"]
        else:
            item = math_prompts[idx] if domain == "math" else short_prompts[idx]
            prompt_id = item["prompt_id"]
            p_ids = format_chat_prompt_ids(tokenizer, item["prompt"])
            r_seed = seed + 100000 + turn_counter
            t_gen0 = time.perf_counter()
            student.eval()
            p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                        torch.manual_seed(r_seed)
                        out_seq = student.generate(
                            input_ids=p_tensor,
                            max_new_tokens=cap,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            pad_token_id=eos_id,
                            eos_token_id=eos_id,
                        )[0, len(p_ids) :].tolist()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            student_rollout_sec += time.perf_counter() - t_gen0
            raw_resp = []
            term_reason = "length_cap"
            for tid in out_seq:
                raw_resp.append(tid)
                if tid == eos_id:
                    term_reason = "eos"
                    break

        if len(raw_resp) == 0:
            continue

        truncated_quota = False
        if len(raw_resp) > dom_rem:
            scored_resp = raw_resp[:dom_rem]
            truncated_quota = True
        else:
            scored_resp = raw_resp

        n_scored = len(scored_resp)
        if domain == "math":
            math_rem -= n_scored
        else:
            short_rem -= n_scored
        scored_total += n_scored

        # Concatenate [prompt, scored_resp[:-1]] to predict scored_resp[0..n_scored-1]
        seq = p_ids + scored_resp[:-1]
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        L_p = len(p_ids)

        # Teacher scoring (no_grad)
        t_t0 = time.perf_counter()
        with torch.inference_mode():
            t_logits = teacher_model(input_ids=inp, use_cache=False).logits[:, L_p - 1 : L_p - 1 + n_scored, :]
            log_pT = F.log_softmax(t_logits.float(), dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        teacher_score_sec += time.perf_counter() - t_t0

        # Student forward + backward forward-KL
        t_b0 = time.perf_counter()
        student.train()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            s_logits = student(input_ids=inp, use_cache=False).logits[:, L_p - 1 : L_p - 1 + n_scored, :]
        log_qS = F.log_softmax(s_logits.float(), dim=-1)
        kl_sum = F.kl_div(log_qS, log_pT, reduction="none", log_target=True).sum()
        # Scale by 1 / ACCUM_TOKENS during backward; rescale before unscale if actual accum != ACCUM_TOKENS
        scaler.scale(kl_sum / float(ACCUM_TOKENS)).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        student_bwd_sec += time.perf_counter() - t_b0

        accum_tokens += n_scored
        accum_kl_sum += float(kl_sum.item())

        rollout_logs.append({
            "presentation": turn_counter,
            "prompt_id": prompt_id,
            "domain": domain,
            "rollout_seed": r_seed,
            "raw_len": len(raw_resp),
            "scored_tokens": n_scored,
            "termination_reason": term_reason,
            "truncated_by_quota": truncated_quota,
            "response_token_ids": scored_resp,
        })

        done = (math_rem == 0 and short_rem == 0)
        if accum_tokens >= ACCUM_TOKENS or done:
            t_opt0 = time.perf_counter()
            if accum_tokens != ACCUM_TOKENS:
                factor = float(ACCUM_TOKENS) / float(accum_tokens)
                for p in trainable:
                    if p.grad is not None:
                        p.grad.mul_(factor)

            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            overflowed = bool(new_scale < prev_scale)
            if overflowed:
                skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            student_bwd_sec += time.perf_counter() - t_opt0

            step_count += 1
            mean_kl = accum_kl_sum / float(accum_tokens)
            curve.append({
                "optimizer_step": step_count,
                "teacher_scored_tokens": scored_total,
                "step_tokens": accum_tokens,
                "mean_kl_loss": mean_kl,
                "lr": LR,
                "grad_norm": grad_norm,
                "overflow_skipped": overflowed,
                "elapsed_sec": time.perf_counter() - t_run_start,
            })
            if step_count % 25 == 0 or done:
                print(
                    f"  [{mode} s={seed}] step={step_count} | scored={scored_total}/{math_quota+short_quota} | "
                    f"KL={mean_kl:.4f} | skipped={skipped_steps}",
                    flush=True,
                )
            accum_tokens = 0
            accum_kl_sum = 0.0

    save_adapter_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(save_adapter_dir))
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total_sec = time.perf_counter() - t_run_start
    del student, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": mode,
        "seed": seed,
        "param_counts": param_counts,
        "peak_allocated_bytes": peak_alloc,
        "math_scored_tokens": math_quota - math_rem,
        "short_scored_tokens": short_quota - short_rem,
        "total_scored_tokens": scored_total,
        "optimizer_steps": step_count,
        "skipped_overflow_steps": skipped_steps,
        "mean_raw_math_len": float(sum(x["raw_len"] for x in rollout_logs if x["domain"] == "math") / max(1, sum(1 for x in rollout_logs if x["domain"] == "math"))),
        "mean_raw_short_len": float(sum(x["raw_len"] for x in rollout_logs if x["domain"] == "short") / max(1, sum(1 for x in rollout_logs if x["domain"] == "short"))),
        "timing_sec": {
            "total_train_loop_sec": total_sec,
            "teacher_scoring_sec": teacher_score_sec,
            "student_rollout_sec": student_rollout_sec,
            "student_fwd_bwd_opt_sec": student_bwd_sec,
            "teacher_pool_prep_sec": offpolicy_pool["prep_time_sec"] if mode == "offpolicy" else 0.0,
        },
        "learning_curve": curve,
        "rollout_logs": rollout_logs,
    }
