"""End-to-end Task 3 runner with Drive checkpointing and resumability."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.env import COMMON_SEED, get_env_metadata, set_seed
from src.task3.distill import (
    format_chat_prompt_ids,
    load_task3_training_prompts,
    merge_and_verify_lora_adapter,
    prepare_offpolicy_teacher_pool,
    run_distillation,
)
from src.task3.eval_task3 import (
    benchmark_isolated_step_costs,
    benchmark_speculative_120_prompts,
    evaluate_distribution_alignment_on_banks,
    evaluate_standalone_accuracy,
    generate_student_or_spec_prefix_bank,
    load_prefix_bank_jsonl_gz,
    load_task3_test_prompts,
)
from src.task3.speculative import (
    speculative_generate,
    validate_markov_chi2,
)

DISTILL_SEEDS = (6304, 6305)
TARGET_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task3")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["all"],
        choices=["all", "validate", "offpolicy", "onpolicy", "banks", "benchmark", "standalone"],
    )
    args = parser.parse_args()
    resume = not args.no_resume
    run_all = "all" in args.phases
    phases = set(args.phases)

    set_seed(COMMON_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distill").mkdir(exist_ok=True)
    (args.output_dir / "adapters").mkdir(exist_ok=True)
    (args.output_dir / "banks").mkdir(exist_ok=True)
    (args.output_dir / "spec_bench").mkdir(exist_ok=True)
    (args.output_dir / "standalone").mkdir(exist_ok=True)

    env_meta = get_env_metadata(args.dataset_root)
    revs = env_meta.get("source_revisions", {}).get("revisions", {})
    t_rev = revs.get("model_target")
    d_rev = revs.get("model_draft")
    save_json(args.output_dir / "env.json", env_meta)

    t_tok = AutoTokenizer.from_pretrained(TARGET_ID, revision=t_rev, use_fast=True)
    d_tok = AutoTokenizer.from_pretrained(DRAFT_ID, revision=d_rev, use_fast=True)

    test_prompts = load_task3_test_prompts(args.dataset_root)
    prompt_subsplit_map = {p["prompt_id"]: p["subsplit"] for p in test_prompts}

    # Check tokenizer & chat template agreement on sample prompts
    tok_agrees = all(
        format_chat_prompt_ids(t_tok, p["prompt"]) == format_chat_prompt_ids(d_tok, p["prompt"])
        for p in test_prompts[:10]
    )
    if not tok_agrees:
        raise RuntimeError("Target and draft tokenizers/chat templates disagree!")

    # Load target and initial undistilled draft
    print(f"Loading target {TARGET_ID} and draft {DRAFT_ID} on {env_meta['device_name']}...", flush=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, revision=t_rev, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()
    orig_draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, revision=d_rev, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(device).eval()

    # -----------------------------------------------------------------------
    # Phase 1: Step 1 Sampler Validation (Markov chi2 + real-model cache consistency)
    # -----------------------------------------------------------------------
    val_path = args.output_dir / "step1_validation.json"
    if (run_all or "validate" in phases) and not (resume and val_path.exists()):
        print("Running Step 1 Markov chi2 validation and real-model cache checks...", flush=True)
        chi2_res = validate_markov_chi2(n_trials=100000, seed=COMMON_SEED, gamma=2)
        assert chi2_res["correct_sampler"]["passed_p_ge_0_01"]
        assert chi2_res["broken_residual_negative_control"]["rejected_p_lt_0_01"]

        # Self-speculative consistency check (target == draft -> 100% acceptance)
        p_ids = torch.tensor([format_chat_prompt_ids(t_tok, test_prompts[0]["prompt"])], dtype=torch.long, device=device)
        self_spec = speculative_generate(
            target_model, target_model, p_ids, max_new_tokens=16, gamma=4,
            eos_token_id=t_tok.eos_token_id, seed=COMMON_SEED, return_final_caches=True,
        )
        # Real-model cache-position consistency check with original draft
        draft_spec = speculative_generate(
            target_model, orig_draft, p_ids, max_new_tokens=16, gamma=4,
            eos_token_id=t_tok.eos_token_id, seed=COMMON_SEED, return_final_caches=True,
        )
        full_seq = torch.cat([p_ids, torch.tensor([draft_spec["generated_ids"]], dtype=torch.long, device=device)], dim=1)
        with torch.inference_mode():
            direct_logits = target_model(input_ids=full_seq, use_cache=False).logits[:, -1, :].float()
            tc = draft_spec["_t_cache"]
            t_pend = draft_spec["_target_pending"]
            l_c = draft_spec["_l_cached"]
            if t_pend is not None:
                cache_len_matches = (tc.layers[0].get_seq_length() == l_c == full_seq.shape[1] - 1)
                cached_logits = target_model(input_ids=t_pend, past_key_values=tc, use_cache=True).logits[:, -1, :].float()
            else:
                cache_len_matches = (tc.layers[0].get_seq_length() == l_c == full_seq.shape[1])
                from transformers import DynamicCache

                tc_crop = DynamicCache()
                for li, lay in enumerate(tc.layers):
                    tc_crop.update(lay.keys[:, :, : full_seq.shape[1] - 1, :].clone(), lay.values[:, :, : full_seq.shape[1] - 1, :].clone(), li)
                cached_logits = target_model(input_ids=full_seq[:, -1:], past_key_values=tc_crop, use_cache=True).logits[:, -1, :].float()
            cache_logit_max_diff = float((direct_logits - cached_logits).abs().max().item())

        val_res = {
            "tokenizer_and_chat_template_agree": tok_agrees,
            "markov_chi2": chi2_res,
            "self_speculative_acc_over_examined": self_spec["accepted_over_examined"],
            "real_model_cache_len_consistent": cache_len_matches,
            "real_model_cache_logit_max_abs_diff": cache_logit_max_diff,
            "real_model_spec_n_gen": draft_spec["n_generated"],
            "direct_logits_finite": bool(torch.isfinite(direct_logits).all()),
        }
        save_json(val_path, val_res)
        print(
            f"  Markov chi2 p={chi2_res['correct_sampler']['pvalue']:.4f} (broken p={chi2_res['broken_residual_negative_control']['pvalue']:.2e}) | "
            f"self-spec acc={self_spec['accepted_over_examined']:.3f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Phase 2 & 3: Off-policy and On-policy Distillation
    # -----------------------------------------------------------------------
    math_train, short_train = load_task3_training_prompts(args.dataset_root)

    if run_all or "offpolicy" in phases:
        pool_path = args.output_dir / "distill" / "offpolicy_teacher_pool.json.gz"
        if resume and pool_path.exists():
            print("Loading cached off-policy teacher rollout pool...", flush=True)
            with gzip.open(pool_path, "rt", encoding="utf-8") as f:
                pool_data = json.load(f)
        else:
            print("Generating off-policy frozen teacher rollout pool (500 math + 300 short)...", flush=True)
            pool_data = prepare_offpolicy_teacher_pool(target_model, t_tok, math_train, short_train, device=device, batch_size=16, seed=COMMON_SEED)
            save_jsonl_gz_obj = pool_path
            with gzip.open(save_jsonl_gz_obj, "wt", encoding="utf-8") as f:
                json.dump(pool_data, f)
            print(
                f"  Pool ready in {pool_data['prep_time_sec']:.1f}s (math={pool_data['math_total_tokens']:,}, short={pool_data['short_total_tokens']:,} toks)",
                flush=True,
            )

        for seed in DISTILL_SEEDS:
            out_p = args.output_dir / "distill" / f"offpolicy_s{seed}.json"
            if resume and out_p.exists():
                continue
            print(f"\nRunning off-policy LoRA distillation (seed={seed})...", flush=True)
            res = run_distillation(
                mode="offpolicy", seed=seed, teacher_model=target_model, tokenizer=t_tok,
                math_prompts=math_train, short_prompts=short_train, offpolicy_pool=pool_data,
                draft_model_id=DRAFT_ID, draft_revision=d_rev,
                save_adapter_dir=args.output_dir / "adapters" / f"offpolicy_s{seed}",
                device=device,
            )
            save_jsonl_gz(args.output_dir / "distill" / f"offpolicy_s{seed}_rollouts.jsonl.gz", res.pop("rollout_logs"))
            save_json(out_p, res)

    if run_all or "onpolicy" in phases:
        for seed in DISTILL_SEEDS:
            out_p = args.output_dir / "distill" / f"onpolicy_s{seed}.json"
            if resume and out_p.exists():
                continue
            print(f"\nRunning on-policy LoRA distillation (seed={seed})...", flush=True)
            res = run_distillation(
                mode="onpolicy", seed=seed, teacher_model=target_model, tokenizer=t_tok,
                math_prompts=math_train, short_prompts=short_train, offpolicy_pool=None,
                draft_model_id=DRAFT_ID, draft_revision=d_rev,
                save_adapter_dir=args.output_dir / "adapters" / f"onpolicy_s{seed}",
                device=device,
            )
            save_jsonl_gz(args.output_dir / "distill" / f"onpolicy_s{seed}_rollouts.jsonl.gz", res.pop("rollout_logs"))
            save_json(out_p, res)

    # Load and merge all 4 trained adapters cleanly into fresh FP16 backbones
    draft_names = ["original", "offpolicy_s6304", "offpolicy_s6305", "onpolicy_s6304", "onpolicy_s6305"]
    merged_drafts: dict[str, Any] = {"original": orig_draft}
    merge_checks: dict[str, Any] = {}
    verify_prompt_strs = [p["prompt"] for p in test_prompts[:4]]

    for dname in draft_names[1:]:
        adir = args.output_dir / "adapters" / dname
        if adir.exists():
            m, chk = merge_and_verify_lora_adapter(adir, DRAFT_ID, d_rev, t_tok, verify_prompt_strs, device)
            merged_drafts[dname] = m
            merge_checks[dname] = chk
    if merge_checks:
        save_json(args.output_dir / "merge_verification.json", merge_checks)

    # -----------------------------------------------------------------------
    # Phase 4: Prefix Banks & Common-Prefix Distribution Alignment (Step 5)
    # -----------------------------------------------------------------------
    if run_all or "banks" in phases:
        # 1. Exact speculative consistency bank using original draft at gamma=4
        spec_bank_path = args.output_dir / "banks" / "exact_spec_original_gamma4.jsonl.gz"
        if not (resume and spec_bank_path.exists()):
            print("Generating exact_spec_original_gamma4 prefix bank (120 prompts)...", flush=True)
            spec_bank = generate_student_or_spec_prefix_bank(
                "exact_spec_original_gamma4", test_prompts, t_tok, orig_draft, target_model=target_model,
                gamma=4, use_speculative=True, seed_offset=0, device=device,
            )
            save_jsonl_gz(spec_bank_path, spec_bank)

        # 2. Own-student frozen rollout banks for each trained draft
        own_banks: list[dict[str, Any]] = []
        for dname in draft_names[1:]:
            bpath = args.output_dir / "banks" / f"own_{dname}.jsonl.gz"
            if resume and bpath.exists():
                own_banks.extend(load_prefix_bank_jsonl_gz(bpath))
            elif dname in merged_drafts:
                print(f"Generating own-student rollout bank for {dname}...", flush=True)
                recs = generate_student_or_spec_prefix_bank(
                    f"own_{dname}", test_prompts, t_tok, merged_drafts[dname],
                    use_speculative=False, seed_offset=555, device=device,
                )
                save_jsonl_gz(bpath, recs)
                own_banks.extend(recs)

        # 3. Evaluate KL, TV, greedy, 1-TV on supplied teacher/original_draft banks + exact_spec + own banks
        align_path = args.output_dir / "prefix_alignment_metrics.json"
        if not (resume and align_path.exists()):
            print("Evaluating common-prefix KL/TV/greedy/expected-acceptance across all banks...", flush=True)
            supplied_banks = load_prefix_bank_jsonl_gz(args.dataset_root / "data" / "prefix_banks" / "prefix_banks_locked.jsonl.gz")
            spec_bank_loaded = load_prefix_bank_jsonl_gz(spec_bank_path)
            all_bank_recs = supplied_banks + spec_bank_loaded + own_banks
            align_res = evaluate_distribution_alignment_on_banks(
                all_bank_recs, prompt_subsplit_map, target_model, merged_drafts, t_tok, device
            )
            save_json(align_path, align_res)

    # -----------------------------------------------------------------------
    # Phase 5: Speculative Decoding Runtime Benchmark + Diagnostics (Steps 2 & 6)
    # -----------------------------------------------------------------------
    if run_all or "benchmark" in phases:
        # Isolated c and v_gamma step costs
        iso_path = args.output_dir / "spec_bench" / "isolated_step_costs.json"
        if not (resume and iso_path.exists()):
            print("Measuring isolated draft/target/verify step costs...", flush=True)
            iso = benchmark_isolated_step_costs(target_model, orig_draft, prompt_lengths=(64, 128, 256), gammas=(2, 4), device=device, repeats=10)
            save_json(iso_path, iso)

        # Primary 120-prompt EOS-aware benchmark for original and all 4 merged drafts at gamma=2,4
        for dname in draft_names:
            if dname not in merged_drafts:
                continue
            b_path = args.output_dir / "spec_bench" / f"spec120_{dname}.jsonl.gz"
            if resume and b_path.exists():
                continue
            print(f"\nBenchmarking 120-prompt speculative decoding for {dname} (gamma=2,4 x 2 reps)...", flush=True)
            rows = benchmark_speculative_120_prompts(
                dname, merged_drafts[dname], target_model, t_tok, test_prompts, device=device, gammas=(2, 4), repeats=2, max_new_tokens=64
            )
            save_jsonl_gz(b_path, rows)

        # Fixed-64-token ignore-EOS throughput diagnostic on 24 math prompts
        noeos_path = args.output_dir / "spec_bench" / "diag24_ignore_eos.jsonl.gz"
        if not (resume and noeos_path.exists()):
            print("Running 24-prompt ignore-EOS fixed-64 throughput diagnostic...", flush=True)
            math24 = [p for p in test_prompts if p["domain"] == "math"][:24]
            diag_rows = []
            for dname in draft_names:
                if dname not in merged_drafts:
                    continue
                for gamma in (2, 4):
                    for j, item in enumerate(math24):
                        p_ids = torch.tensor([format_chat_prompt_ids(t_tok, item["prompt"])], dtype=torch.long, device=device)
                        seed_j = 6304 + 10007 * j + 31 * gamma
                        t_res = target_only_generate(target_model, p_ids, 64, t_tok.eos_token_id, ignore_eos=True, seed=seed_j)
                        s_res = speculative_generate(target_model, merged_drafts[dname], p_ids, 64, gamma, t_tok.eos_token_id, ignore_eos=True, seed=seed_j)
                        diag_rows.append({
                            "draft": dname, "gamma": gamma, "prompt_id": item["prompt_id"],
                            "target_total_ms": t_res["total_ms"], "target_decode_ms": t_res["decode_ms"],
                            "spec_total_ms": s_res["total_ms"], "spec_decode_ms": s_res["decode_ms"],
                            "accepted_over_proposed": s_res["accepted_over_proposed"],
                            "speedup_total": t_res["total_ms"] / max(s_res["total_ms"], 1e-6),
                        })
            save_jsonl_gz(noeos_path, diag_rows)

        # Unmerged wrapper latency overhead check (1 adapter on 24 prompts)
        unm_path = args.output_dir / "spec_bench" / "unmerged_wrapper_overhead.json"
        adir_chk = args.output_dir / "adapters" / "offpolicy_s6304"
        if adir_chk.exists() and not (resume and unm_path.exists()):
            print("Timing unmerged PEFT adapter wrapper overhead vs merged FP16...", flush=True)
            base_tmp = AutoModelForCausalLM.from_pretrained(DRAFT_ID, revision=d_rev, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
            unmerged = PeftModel.from_pretrained(base_tmp, str(adir_chk)).to(device=device, dtype=torch.float16).eval()
            sub24 = test_prompts[:24]
            r_unm = benchmark_speculative_120_prompts("offpolicy_s6304_unmerged", unmerged, target_model, t_tok, sub24, device=device, gammas=(4,), repeats=1, max_new_tokens=64)
            r_mrg = benchmark_speculative_120_prompts("offpolicy_s6304_merged_24", merged_drafts["offpolicy_s6304"], target_model, t_tok, sub24, device=device, gammas=(4,), repeats=1, max_new_tokens=64)
            save_json(unm_path, {
                "unmerged_mean_draft_ms": float(sum(x["speculative"]["round_timing_ms"]["draft_proposal_ms"] for x in r_unm) / len(r_unm)),
                "merged_mean_draft_ms": float(sum(x["speculative"]["round_timing_ms"]["draft_proposal_ms"] for x in r_mrg) / len(r_mrg)),
                "unmerged_mean_total_ms": float(sum(x["speculative"]["total_ms"] for x in r_unm) / len(r_unm)),
                "merged_mean_total_ms": float(sum(x["speculative"]["total_ms"] for x in r_mrg) / len(r_mrg)),
            })
            del unmerged, base_tmp

    # -----------------------------------------------------------------------
    # Phase 6: Standalone Model Answer Accuracy (512 math / 32 short)
    # -----------------------------------------------------------------------
    if run_all or "standalone" in phases:
        models_to_eval = [("target_1.5B", target_model), *[(d, merged_drafts[d]) for d in draft_names if d in merged_drafts]]
        for mname, mod in models_to_eval:
            out_s = args.output_dir / "standalone" / f"{mname}.jsonl.gz"
            if resume and out_s.exists():
                continue
            print(f"Evaluating standalone accuracy (512 math / 32 short) for {mname}...", flush=True)
            rows = evaluate_standalone_accuracy(mname, mod, t_tok, test_prompts, device=device)
            save_jsonl_gz(out_s, rows)

    print("\nALL REQUESTED TASK 3 PHASES COMPLETE.", flush=True)


if __name__ == "__main__":
    main()

