"""Task 3: Exact speculative sampling, Markov chi-squared validation, cache sync, and cost model."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache


def _crop_dynamic_cache(cache: DynamicCache, length: int) -> None:
    """Crop every layer in a transformers DynamicCache to physical length `length`."""
    for layer in cache.layers:
        layer.crop(length)


def _sample_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    """Sample one token index from a 1D probability vector."""
    if generator is not None and generator.device != probs.device:
        cdf = torch.cumsum(probs.float(), dim=-1)
        u = float(torch.rand((), generator=generator, device=generator.device).item())
        u_t = torch.tensor(u, device=probs.device, dtype=cdf.dtype)
        idx = int(torch.searchsorted(cdf, u_t).item())
        return min(idx, probs.shape[-1] - 1)
    return int(torch.multinomial(probs.float(), num_samples=1, generator=generator).item())


def _sample_uniform(generator: Optional[torch.Generator] = None, device: str | torch.device = "cpu") -> float:
    dev = generator.device if generator is not None else device
    return float(torch.rand((), generator=generator, device=dev).item())


def _logits_to_probs(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """Convert 1D logits to temperature-adjusted (and optionally nucleus-filtered) probabilities."""
    if temperature <= 0.0:
        out = torch.zeros_like(logits, dtype=torch.float32)
        out[int(logits.argmax().item())] = 1.0
        return out
    scaled = logits.float() / float(temperature)
    probs = F.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = (cum - sorted_probs) > top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
        out = torch.zeros_like(probs)
        out.scatter_(0, sorted_idx, sorted_probs)
        return out
    return probs


# ---------------------------------------------------------------------------
# Exact Speculative Sampling on Finite Categorical / Markov Toy Models (Step 1)
# ---------------------------------------------------------------------------

def sample_markov_exact_speculative(
    target_init: np.ndarray,
    target_trans: np.ndarray,
    draft_init: np.ndarray,
    draft_trans: np.ndarray,
    seq_len: int,
    gamma: int,
    n_trials: int,
    seed: int = 6304,
    use_broken_residual: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample `n_trials` sequences of length `seq_len` from a finite-state Markov chain
    using exact speculative sampling (or deliberately broken `r = p` negative control).
    """
    rng = np.random.default_rng(seed)
    V = len(target_init)
    seqs = np.empty((n_trials, seq_len), dtype=np.int64)
    total_proposed = 0
    total_accepted = 0
    total_bonus = 0

    for t in range(n_trials):
        out: list[int] = []
        while len(out) < seq_len:
            rem = seq_len - len(out)
            m = min(gamma, rem)
            # Draft proposes m tokens autoregressively given current prefix
            draft_toks = []
            draft_dists = []
            prev = out[-1] if out else None
            for _ in range(m):
                q = draft_init if prev is None else draft_trans[prev]
                x = int(rng.choice(V, p=q))
                draft_toks.append(x)
                draft_dists.append(q)
                prev = x

            total_proposed += m
            # Target distributions at positions len(out) .. len(out) + m - 1 (plus bonus at +m)
            all_accepted = True
            prev_t = out[-1] if out else None
            for j in range(m):
                p_j = target_init if prev_t is None else target_trans[prev_t]
                q_j = draft_dists[j]
                x_j = draft_toks[j]
                acc_prob = min(1.0, float(p_j[x_j]) / max(float(q_j[x_j]), 1e-15))
                if rng.random() <= acc_prob:
                    out.append(x_j)
                    total_accepted += 1
                    prev_t = x_j
                else:
                    if use_broken_residual:
                        res_p = p_j
                    else:
                        diff = np.maximum(p_j - q_j, 0.0)
                        s = diff.sum()
                        res_p = (diff / s) if s > 0 else p_j
                    x_res = int(rng.choice(V, p=res_p))
                    out.append(x_res)
                    all_accepted = False
                    break

            if all_accepted and len(out) < seq_len:
                p_bonus = target_init if prev_t is None else target_trans[prev_t]
                x_b = int(rng.choice(V, p=p_bonus))
                out.append(x_b)
                total_bonus += 1

        seqs[t] = out[:seq_len]

    return seqs, {
        "n_trials": n_trials,
        "accepted_over_proposed": total_accepted / max(total_proposed, 1),
        "total_bonus": total_bonus,
    }


def validate_markov_chi2(n_trials: int = 100000, seed: int = 6304, gamma: int = 2) -> dict[str, Any]:
    """Run the 100,000-trial 3-symbol Markov chain Pearson chi-squared validation
    and deliberately broken residual negative control (all expected counts >= 10).
    """
    from scipy.stats import chisquare

    # 3-symbol Markov chain with all transition probs bounded away from 0 so expected counts >> 10
    target_init = np.array([0.50, 0.30, 0.20], dtype=np.float64)
    target_trans = np.array(
        [
            [0.55, 0.30, 0.15],
            [0.20, 0.50, 0.30],
            [0.35, 0.25, 0.40],
        ],
        dtype=np.float64,
    )
    # Intentionally mismatched draft so rejections + residual sampling occur frequently
    draft_init = np.array([0.25, 0.45, 0.30], dtype=np.float64)
    draft_trans = np.array(
        [
            [0.25, 0.45, 0.30],
            [0.45, 0.25, 0.30],
            [0.20, 0.50, 0.30],
        ],
        dtype=np.float64,
    )

    # Exact joint probability over all 27 length-3 sequences
    expected_probs = np.zeros(27, dtype=np.float64)
    idx = 0
    labels = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                expected_probs[idx] = target_init[a] * target_trans[a, b] * target_trans[b, c]
                labels.append(f"{a}{b}{c}")
                idx += 1
    expected_counts = expected_probs * n_trials
    assert float(expected_counts.min()) >= 10.0

    def count_seqs(seqs: np.ndarray) -> np.ndarray:
        codes = seqs[:, 0] * 9 + seqs[:, 1] * 3 + seqs[:, 2]
        return np.bincount(codes, minlength=27).astype(np.float64)

    seqs_ok, stats_ok = sample_markov_exact_speculative(
        target_init, target_trans, draft_init, draft_trans, seq_len=3, gamma=gamma, n_trials=n_trials, seed=seed, use_broken_residual=False
    )
    obs_ok = count_seqs(seqs_ok)
    chi_ok = chisquare(obs_ok, expected_counts)

    seqs_bad, stats_bad = sample_markov_exact_speculative(
        target_init, target_trans, draft_init, draft_trans, seq_len=3, gamma=gamma, n_trials=n_trials, seed=seed, use_broken_residual=True
    )
    obs_bad = count_seqs(seqs_bad)
    chi_bad = chisquare(obs_bad, expected_counts)

    return {
        "n_trials": n_trials,
        "gamma": gamma,
        "min_expected_count": float(expected_counts.min()),
        "correct_sampler": {
            "chi2": float(chi_ok.statistic),
            "pvalue": float(chi_ok.pvalue),
            "passed_p_ge_0_01": bool(chi_ok.pvalue >= 0.01),
            **stats_ok,
        },
        "broken_residual_negative_control": {
            "chi2": float(chi_bad.statistic),
            "pvalue": float(chi_bad.pvalue),
            "rejected_p_lt_0_01": bool(chi_bad.pvalue < 0.01),
            **stats_bad,
        },
        "joint_sequence_table": [
            {
                "seq": labels[i],
                "expected_prob": float(expected_probs[i]),
                "correct_empirical_prob": float(obs_ok[i] / n_trials),
                "broken_empirical_prob": float(obs_bad[i] / n_trials),
            }
            for i in range(27)
        ],
    }


# ---------------------------------------------------------------------------
# Real-Model Exact Speculative Sampling with Synchronized 4-Part Timers
# ---------------------------------------------------------------------------

def _sync_ms(t0: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def target_only_generate(
    target_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    temperature: float = 1.0,
    top_p: float = 1.0,
    ignore_eos: bool = False,
    seed: int = 6304,
) -> dict[str, Any]:
    """Baseline autoregressive generation using only the target model with aligned timing."""
    device = prompt_ids.device
    gen = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    out = target_model(input_ids=prompt_ids, use_cache=True)
    prefill_ms = _sync_ms(t0, device)

    cache = out.past_key_values
    logits = out.logits[:, -1, :].squeeze(0)
    generated: list[int] = []
    decode_t0 = time.perf_counter()
    target_passes = 1  # prefill

    for step in range(max_new_tokens):
        probs = _logits_to_probs(logits, temperature=temperature, top_p=top_p)
        tok = _sample_from_probs(probs, generator=gen)
        generated.append(tok)
        if not ignore_eos and eos_token_id is not None and tok == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break
        nxt = torch.tensor([[tok]], dtype=torch.long, device=device)
        out = target_model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].squeeze(0)
        target_passes += 1

    decode_ms = _sync_ms(decode_t0, device)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    terminated_by_eos = bool((not ignore_eos) and eos_token_id is not None and generated and generated[-1] == eos_token_id)

    return {
        "generated_ids": generated,
        "n_generated": len(generated),
        "terminated_by_eos": terminated_by_eos,
        "truncated_by_cap": not terminated_by_eos and len(generated) >= max_new_tokens,
        "target_passes_total": target_passes,
        "target_verify_passes": target_passes - 1,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": prefill_ms + decode_ms,
        "decode_tokens_per_sec": len(generated) / max(decode_ms / 1000.0, 1e-9),
        "total_tokens_per_sec": len(generated) / max((prefill_ms + decode_ms) / 1000.0, 1e-9),
        "peak_allocated_bytes": peak_mem,
    }


@torch.inference_mode()
def speculative_generate(
    target_model,
    draft_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    eos_token_id: Optional[int],
    temperature: float = 1.0,
    top_p: float = 1.0,
    ignore_eos: bool = False,
    seed: int = 6304,
    collect_round_traces: bool = False,
    return_final_caches: bool = False,
) -> dict[str, Any]:
    """Exact speculative sampling with DynamicCache rollback/advancement and 4-part round timers."""
    device = prompt_ids.device
    gen = torch.Generator(device=device).manual_seed(seed)
    L_p = int(prompt_ids.shape[1])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Prefill [0 .. L_p - 2] into both caches; keep prompt_ids[:, -1:] as initial pending token
    t_pre = time.perf_counter()
    if L_p > 1:
        prefix = prompt_ids[:, :-1]
        t_out = target_model(input_ids=prefix, use_cache=True)
        d_out = draft_model(input_ids=prefix, use_cache=True)
        t_cache = t_out.past_key_values
        d_cache = d_out.past_key_values
        l_cached = L_p - 1
    else:
        t_cache = DynamicCache()
        d_cache = DynamicCache()
        l_cached = 0
    prefill_ms = _sync_ms(t_pre, device)

    target_pending = prompt_ids[:, -1:]  # [1, 1] at position l_cached
    draft_pending = prompt_ids[:, -1:]   # [1, 1] at position l_cached (becomes [x_m, x_bonus] after full accept + bonus)

    generated: list[int] = []
    rounds = 0
    draft_passes = 0
    target_verify_passes = 0
    proposed_tokens = 0
    examined_tokens = 0
    accepted_proposed = 0
    bonus_tokens = 0
    accepted_prefix_lengths: list[int] = []
    round_traces = []

    draft_proposal_ms = 0.0
    target_verify_ms = 0.0
    accept_residual_ms = 0.0
    cache_update_ms = 0.0
    t_decode_start = time.perf_counter()

    while len(generated) < max_new_tokens:
        rounds += 1
        rem_cap = max_new_tokens - len(generated)
        m_cap = min(gamma, rem_cap)

        # 1. Draft proposal passes
        t0 = time.perf_counter()
        d_toks: list[int] = []
        d_probs: list[torch.Tensor] = []
        cur_in = draft_pending
        for j in range(m_cap):
            d_out = draft_model(input_ids=cur_in, past_key_values=d_cache, use_cache=True)
            d_cache = d_out.past_key_values
            draft_passes += 1
            q_j = _logits_to_probs(d_out.logits[:, -1, :].squeeze(0), temperature=temperature, top_p=top_p)
            x_next = _sample_from_probs(q_j, generator=gen)
            d_toks.append(x_next)
            d_probs.append(q_j)
            if not ignore_eos and eos_token_id is not None and x_next == eos_token_id:
                break
            cur_in = torch.tensor([[x_next]], dtype=torch.long, device=device)
        draft_proposal_ms += _sync_ms(t0, device)

        m = len(d_toks)
        proposed_tokens += m

        # 2. Single target verification forward pass on [target_pending, x_1, ..., x_m]
        t1 = time.perf_counter()
        verify_seq = torch.cat(
            [target_pending, torch.tensor([d_toks], dtype=torch.long, device=device)],
            dim=1,
        )
        t_out = target_model(input_ids=verify_seq, past_key_values=t_cache, use_cache=True)
        t_cache = t_out.past_key_values
        target_verify_passes += 1
        target_verify_ms += _sync_ms(t1, device)

        # 3. Exact stochastic accept/reject + positive residual + bonus
        t2 = time.perf_counter()
        t_logits = t_out.logits.squeeze(0)  # [m + 1, V]
        n_acc = 0
        all_accepted = True
        hit_eos = False
        res_tok: Optional[int] = None
        bonus_tok: Optional[int] = None

        for j in range(m):
            examined_tokens += 1
            p_j = _logits_to_probs(t_logits[j], temperature=temperature, top_p=top_p)
            q_j = d_probs[j]
            x_j = d_toks[j]
            acc_prob = min(1.0, float(p_j[x_j].item()) / max(float(q_j[x_j].item()), 1e-15))
            if _sample_uniform(generator=gen) <= acc_prob:
                generated.append(x_j)
                n_acc += 1
                accepted_proposed += 1
                if not ignore_eos and eos_token_id is not None and x_j == eos_token_id:
                    hit_eos = True
                    all_accepted = False  # stop cleanly without bonus
                    break
            else:
                diff = torch.clamp(p_j - q_j, min=0.0)
                s = float(diff.sum().item())
                res_probs = (diff / s) if s > 0.0 else p_j
                res_tok = _sample_from_probs(res_probs, generator=gen)
                generated.append(res_tok)
                if not ignore_eos and eos_token_id is not None and res_tok == eos_token_id:
                    hit_eos = True
                all_accepted = False
                break

        if all_accepted and len(generated) < max_new_tokens:
            p_bonus = _logits_to_probs(t_logits[m], temperature=temperature, top_p=top_p)
            bonus_tok = _sample_from_probs(p_bonus, generator=gen)
            generated.append(bonus_tok)
            bonus_tokens += 1
            if not ignore_eos and eos_token_id is not None and bonus_tok == eos_token_id:
                hit_eos = True

        accepted_prefix_lengths.append(n_acc)
        accept_residual_ms += _sync_ms(t2, device)

        if collect_round_traces:
            round_traces.append({
                "round": rounds,
                "proposed": d_toks,
                "n_accepted": n_acc,
                "residual_tok": res_tok,
                "bonus_tok": bonus_tok,
                "all_accepted": all_accepted,
            })

        if hit_eos or len(generated) >= max_new_tokens:
            if return_final_caches:
                if res_tok is not None:
                    new_cached = l_cached + 1 + n_acc
                    _crop_dynamic_cache(t_cache, new_cached)
                    _crop_dynamic_cache(d_cache, new_cached)
                    l_cached = new_cached
                    target_pending = torch.tensor([[res_tok]], dtype=torch.long, device=device)
                elif bonus_tok is not None:
                    new_cached = l_cached + 1 + m
                    l_cached = new_cached
                    target_pending = torch.tensor([[bonus_tok]], dtype=torch.long, device=device)
                else:
                    # Stopped on accepted token without bonus: crop any trailing rejected/unaccepted target tokens
                    new_cached = l_cached + 1 + n_acc
                    _crop_dynamic_cache(t_cache, new_cached)
                    _crop_dynamic_cache(d_cache, new_cached)
                    l_cached = new_cached
                    target_pending = None
            break

        # 4. Cache rollback / advancement to preserve exact aligned state
        t3 = time.perf_counter()
        if res_tok is not None:
            # Rejected at n_acc: keep [target_pending, x_1..x_{n_acc}] in both caches (length l_cached + 1 + n_acc)
            new_cached = l_cached + 1 + n_acc
            _crop_dynamic_cache(t_cache, new_cached)
            _crop_dynamic_cache(d_cache, new_cached)
            l_cached = new_cached
            tok_t = torch.tensor([[res_tok]], dtype=torch.long, device=device)
            target_pending = tok_t
            draft_pending = tok_t
        else:
            # Full acceptance of m tokens + bonus_tok:
            # t_cache already holds [target_pending, x_1..x_m] (len l_cached + 1 + m)
            # d_cache currently holds [draft_pending, x_1..x_{m-1}] (len l_cached + m)
            new_cached = l_cached + 1 + m
            l_cached = new_cached
            b_t = torch.tensor([[bonus_tok]], dtype=torch.long, device=device)
            xm_t = torch.tensor([[d_toks[-1]]], dtype=torch.long, device=device)
            target_pending = b_t
            draft_pending = torch.cat([xm_t, b_t], dim=1)
        cache_update_ms += _sync_ms(t3, device)

    decode_total_ms = _sync_ms(t_decode_start, device)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    host_overhead_ms = max(0.0, decode_total_ms - (draft_proposal_ms + target_verify_ms + accept_residual_ms + cache_update_ms))
    terminated_by_eos = bool((not ignore_eos) and eos_token_id is not None and generated and generated[-1] == eos_token_id)

    res = {
        "generated_ids": generated,
        "n_generated": len(generated),
        "terminated_by_eos": terminated_by_eos,
        "truncated_by_cap": not terminated_by_eos and len(generated) >= max_new_tokens,
        "gamma": gamma,
        "rounds": rounds,
        "proposed_tokens": proposed_tokens,
        "examined_tokens": examined_tokens,
        "accepted_proposed": accepted_proposed,
        "bonus_tokens": bonus_tokens,
        "accepted_over_examined": accepted_proposed / max(examined_tokens, 1),
        "accepted_over_proposed": accepted_proposed / max(proposed_tokens, 1),
        "mean_accepted_prefix_per_verify": float(np.mean(accepted_prefix_lengths)) if accepted_prefix_lengths else 0.0,
        "draft_passes": draft_passes,
        "target_verify_passes": target_verify_passes,
        "target_passes_total": target_verify_passes + (1 if L_p > 1 else 0),
        "target_passes_per_output_token": target_verify_passes / max(len(generated), 1),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_total_ms,
        "total_ms": prefill_ms + decode_total_ms,
        "decode_tokens_per_sec": len(generated) / max(decode_total_ms / 1000.0, 1e-9),
        "total_tokens_per_sec": len(generated) / max((prefill_ms + decode_total_ms) / 1000.0, 1e-9),
        "round_timing_ms": {
            "draft_proposal_ms": draft_proposal_ms,
            "target_verify_ms": target_verify_ms,
            "accept_residual_ms": accept_residual_ms,
            "cache_update_ms": cache_update_ms,
            "host_overhead_ms": host_overhead_ms,
        },
        "peak_allocated_bytes": peak_mem,
        "round_traces": round_traces if collect_round_traces else None,
    }
    if return_final_caches:
        res["_t_cache"] = t_cache
        res["_d_cache"] = d_cache
        res["_l_cached"] = l_cached
        res["_target_pending"] = target_pending
    return res


def compute_analytical_speculative_model(alpha: float, c: float, v_gamma: float, gamma: int) -> dict[str, Any]:
    """Evaluate S_hat_gamma = (1 + sum_{j=1}^gamma alpha^j) / (gamma*c + v_gamma) and break-even alpha."""
    denom = gamma * c + v_gamma
    expected_tokens = 1.0 + sum(alpha ** j for j in range(1, gamma + 1))
    s_hat = expected_tokens / max(denom, 1e-12)

    # Solve 1 + sum_{j=1}^gamma a^j = denom on a in [0, 1]
    break_even = None
    if denom <= 1.0:
        break_even = 0.0
    elif 1.0 + gamma <= denom:
        break_even = None  # impossible even at alpha = 1
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = 1.0 + sum(mid ** j for j in range(1, gamma + 1))
            if val < denom:
                lo = mid
            else:
                hi = mid
        break_even = 0.5 * (lo + hi)

    return {
        "gamma": gamma,
        "alpha": alpha,
        "c": c,
        "v_gamma": v_gamma,
        "predicted_speedup_S_hat": s_hat,
        "break_even_alpha": break_even,
    }
