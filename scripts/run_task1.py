"""End-to-end Task 1 runner for Colab GPU execution with automatic resumability.

Runs:
1. Baseline evaluation + model storage accounting + calibration probes (C4 128x512).
2. Uniform pruning (P1 Magnitude, P2 Wanda at 25/50/75%, 2:4 Wanda at 50%).
3. P3 Nonuniform dev-set selection (C4 dev 32x512 only) -> P3 test runs (50%, 75%).
4. Uniform PTQ (Q1) and AWQ-style PTQ (Q2) at INT8/INT4/INT3 (group size 128).
5. Q3 Mixed-precision dev-set selection (C4 dev 32x512 only) -> Q3 test runs (INT4 & INT3 budgets).
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.env import COMMON_SEED, get_env_metadata, set_seed
from src.task1.benchmark import benchmark_inference
from src.task1.eval_hellaswag import evaluate_hellaswag, load_hellaswag_examples
from src.task1.eval_ppl import evaluate_perplexity_windows, load_locked_windows
from src.task1.model_utils import (
    compute_pruned_model_storage,
    compute_unique_parameter_accounting,
    enumerate_eligible_projections,
    load_qwen_base_fp16,
)
from src.task1.prune import (
    allocate_marginal_distortion_sparsities,
    allocate_owl_sparsities,
    allocate_shortgpt_bi_sparsities,
    apply_pruning_to_model,
    collect_calibration_stats,
    collect_output_fisher_scales,
    load_sequence_manifest,
    measure_layer_output_errors,
)
from src.task1.quant import (
    allocate_mixed_precision_milp,
    apply_q1_uniform,
    apply_q2_awq,
    clear_awq_hooks_and_buffers,
    measure_quantization_distortion_matrix,
)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_json_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def restore_base_weights(model: torch.nn.Module, base_state_cpu: dict[str, torch.Tensor]) -> None:
    clear_awq_hooks_and_buffers(model)
    model.load_state_dict(base_state_cpu, strict=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_and_save_condition(
    cond_id: str,
    model: torch.nn.Module,
    ref_model_cpu: torch.nn.Module | None,
    tokenizer,
    datasets: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    hellaswag_full: bool,
    storage_info: dict[str, Any],
    compression_time_sec: float,
    extra_meta: dict[str, Any] | None = None,
    eval_batch_size: int = 8,
    hs_batch_size: int = 32,
    skip_bench: bool = False,
    measure_errors: bool = True,
) -> dict[str, Any]:
    cond_path = out_dir / "conditions" / f"{cond_id}.json"
    print(f"\n=== Evaluating condition: {cond_id} (hellaswag_full={hellaswag_full}) ===", flush=True)

    # 1. Perplexity suites
    wt2_res = evaluate_perplexity_windows(
        model, datasets["wt2"], device=device, batch_size=eval_batch_size, expected_total_tokens=50000
    )
    c4_val_res = evaluate_perplexity_windows(
        model, datasets["c4_val"], device=device, batch_size=eval_batch_size, expected_total_tokens=50000
    )
    c4_dev_res = evaluate_perplexity_windows(
        model, datasets["c4_dev_windows"], device=device, batch_size=eval_batch_size, expected_total_tokens=None
    )

    # 2. HellaSwag
    hs_examples = datasets["hellaswag_2000"] if hellaswag_full else datasets["hellaswag_512"]
    hs_res = evaluate_hellaswag(
        model, tokenizer, hs_examples, device=device, batch_size_candidates=hs_batch_size
    )
    hs_diag_512 = hs_res["per_example"][:512]
    hs_diag_acc = sum(x["correct_raw"] for x in hs_diag_512) / 512.0
    hs_diag_acc_norm = sum(x["correct_norm"] for x in hs_diag_512) / 512.0

    # 3. Layer-output error on calibration subset (move ref_model to device temporarily, then back to CPU)
    layer_errs = None
    if measure_errors and ref_model_cpu is not None:
        ref_model_cpu.to(device)
        layer_errs = measure_layer_output_errors(
            ref_model_cpu, model, datasets["c4_cal_seqs"], device=device, batch_size=4, max_sequences=32
        )
        ref_model_cpu.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Latency & Peak VRAM benchmark (alone on GPU)
    bench = None
    if not skip_bench:
        bench = benchmark_inference(
            model,
            vocab_size=model.config.vocab_size,
            device=device,
            warmup=3,
            repeats=5,
            decode_prompt_len=512,
            decode_steps=64,
            seed=COMMON_SEED,
        )

    # Save per-example outputs cleanly
    save_json_gz(
        out_dir / "per_example" / f"{cond_id}_ppl_wt2.json.gz",
        {
            "per_window_nll_sum": wt2_res["per_window_nll_sum"],
            "per_window_scored_tokens": wt2_res["per_window_scored_tokens"],
        },
    )
    save_json_gz(
        out_dir / "per_example" / f"{cond_id}_ppl_c4.json.gz",
        {
            "per_window_nll_sum": c4_val_res["per_window_nll_sum"],
            "per_window_scored_tokens": c4_val_res["per_window_scored_tokens"],
        },
    )
    save_json_gz(
        out_dir / "per_example" / f"{cond_id}_hellaswag.json.gz",
        hs_res["per_example"],
    )

    summary = {
        "condition_id": cond_id,
        "compression_time_sec": compression_time_sec,
        "wikitext2": {
            "n_windows": wt2_res["n_windows"],
            "scored_tokens": wt2_res["scored_tokens"],
            "total_nll": wt2_res["total_nll"],
            "mean_nll": wt2_res["mean_nll"],
            "perplexity": wt2_res["perplexity"],
        },
        "c4_val": {
            "n_windows": c4_val_res["n_windows"],
            "scored_tokens": c4_val_res["scored_tokens"],
            "total_nll": c4_val_res["total_nll"],
            "mean_nll": c4_val_res["mean_nll"],
            "perplexity": c4_val_res["perplexity"],
        },
        "c4_dev": {
            "scored_tokens": c4_dev_res["scored_tokens"],
            "mean_nll": c4_dev_res["mean_nll"],
            "perplexity": c4_dev_res["perplexity"],
        },
        "hellaswag": {
            "prompt_format": hs_res["prompt_format"],
            "n_evaluated": hs_res["n_examples"],
            "acc": hs_res["acc"],
            "acc_norm": hs_res["acc_norm"],
            "diag_512_acc": hs_diag_acc,
            "diag_512_acc_norm": hs_diag_acc_norm,
        },
        "storage": storage_info,
        "layer_output_errors": layer_errs,
        "benchmark": bench,
        "meta": extra_meta or {},
    }
    save_json(cond_path, summary)
    print(
        f"  [{cond_id}] WT2 PPL={summary['wikitext2']['perplexity']:.3f} | "
        f"C4 PPL={summary['c4_val']['perplexity']:.3f} | "
        f"HS({hs_res['n_examples']}) acc_norm={hs_res['acc_norm']:.4f} (512={hs_diag_acc_norm:.4f})",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task1")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--hs-batch-size", type=int, default=32)
    parser.add_argument("--cal-batch-size", type=int, default=4)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "baseline",
            "prune_uniform",
            "prune_dev",
            "prune_p3",
            "quant_uniform",
            "quant_dev",
            "quant_q3",
        ],
    )
    args = parser.parse_args()
    resume = not args.no_resume
    run_all = "all" in args.phases
    phases = set(args.phases)

    set_seed(COMMON_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conditions").mkdir(exist_ok=True)
    (args.output_dir / "per_example").mkdir(exist_ok=True)
    (args.output_dir / "dev_selection").mkdir(exist_ok=True)

    env_meta = get_env_metadata(args.dataset_root)
    rev = env_meta.get("source_revisions", {}).get("revisions", {}).get("model_base")
    print(f"Device: {env_meta['device_name']} | Qwen2.5-1.5B rev: {rev}", flush=True)

    # Load all locked Task 1 datasets
    pub = args.dataset_root / "data" / "public_locked"
    c4_cal_seqs = load_sequence_manifest(pub / "c4_calibration_locked.jsonl.gz")
    c4_dev_seqs = load_sequence_manifest(pub / "c4_development_locked.jsonl.gz")
    # Convert 32x512 dev sequences into PPL-window format matching setup_data.py
    c4_dev_windows = [
        {
            "token_ids": seq,
            "score_target_offsets": [1, len(seq)],
            "scored_tokens": len(seq) - 1,
        }
        for seq in c4_dev_seqs
    ]
    datasets = {
        "c4_cal_seqs": c4_cal_seqs,
        "c4_dev_seqs": c4_dev_seqs,
        "c4_dev_windows": c4_dev_windows,
        "wt2": load_locked_windows(pub / "wikitext2_test_windows_locked.jsonl.gz"),
        "c4_val": load_locked_windows(pub / "c4_validation_windows_locked.jsonl.gz"),
        "hellaswag_2000": load_hellaswag_examples(pub / "hellaswag_val_locked.jsonl.gz"),
    }
    datasets["hellaswag_512"] = datasets["hellaswag_2000"][:512]

    # Load base FP16 model once and snapshot clean CPU state_dict
    model, tok = load_qwen_base_fp16("Qwen/Qwen2.5-1.5B", revision=rev, device=device)
    acct = compute_unique_parameter_accounting(model)
    save_json(args.output_dir / "env_and_accounting.json", {"env": env_meta, "accounting": acct})
    print(f"Eligible matrices: {acct['num_eligible_matrices']}, eligible params: {acct['eligible_params']:,}, "
          f"tied duplicates: {acct['tied_duplicate_tensors']}, baseline FP16 bytes: {acct['baseline_fp16_bytes']:,}", flush=True)

    base_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ref_model_cpu = copy.deepcopy(model).cpu().eval()

    # Cache / load calibration probes (input_l2, input_mean_abs, block_bi, fisher, quant_distortion)
    probes_path = args.output_dir / "calibration_probes.pt"
    if probes_path.exists() and resume:
        print(f"Loading cached calibration probes from {probes_path}", flush=True)
        probes = torch.load(probes_path, map_location="cpu", weights_only=False)
    else:
        print("Collecting calibration stats (Wanda L2 norms, ShortGPT BI, Output Fisher)...", flush=True)
        t0 = time.perf_counter()
        cal_stats = collect_calibration_stats(model, c4_cal_seqs, device=device, batch_size=args.cal_batch_size)
        t_cal = time.perf_counter() - t0

        t1 = time.perf_counter()
        fisher_stats = collect_output_fisher_scales(model, c4_cal_seqs, device=device)
        t_fisher = time.perf_counter() - t1

        t2 = time.perf_counter()
        q_dist = measure_quantization_distortion_matrix(
            model, c4_cal_seqs, candidate_bits=(2, 3, 4, 8), device=device, batch_size=args.cal_batch_size, max_sequences=64
        )
        t_qdist = time.perf_counter() - t2

        probes = {
            **cal_stats,
            **fisher_stats,
            "quant_distortion_matrix": q_dist,
            "timing_sec": {"cal_stats": t_cal, "fisher": t_fisher, "quant_distortion": t_qdist},
        }
        torch.save(probes, probes_path)
        print(f"Saved calibration probes (cal={t_cal:.1f}s, fisher={t_fisher:.1f}s, qdist={t_qdist:.1f}s)", flush=True)

    # -----------------------------------------------------------------------
    # Phase 1: Baseline
    # -----------------------------------------------------------------------
    if run_all or "baseline" in phases:
        cid = "fp16_baseline"
        if not (resume and (args.output_dir / "conditions" / f"{cid}.json").exists()):
            restore_base_weights(model, base_state_cpu)
            storage = compute_pruned_model_storage(model)
            storage.pop("per_projection", None)
            evaluate_and_save_condition(
                cid, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                hellaswag_full=True, storage_info=storage, compression_time_sec=0.0,
                eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                skip_bench=args.skip_bench, measure_errors=False,
            )

    # -----------------------------------------------------------------------
    # Phase 2: P1 (Magnitude), P2 (Wanda), and 2:4 Wanda Diagnostic
    # -----------------------------------------------------------------------
    if run_all or "prune_uniform" in phases:
        uniform_runs = [
            ("p1_mag_s25", "magnitude", 0.25, False),
            ("p1_mag_s50", "magnitude", 0.50, True),
            ("p1_mag_s75", "magnitude", 0.75, False),
            ("p2_wanda_s25", "wanda", 0.25, False),
            ("p2_wanda_s50", "wanda", 0.50, True),
            ("p2_wanda_s75", "wanda", 0.75, False),
            ("p2_wanda_2_4_s50", "wanda_2_4", 0.50, False),
        ]
        for cid, method, target_s, hs_full in uniform_runs:
            if resume and (args.output_dir / "conditions" / f"{cid}.json").exists():
                continue
            restore_base_weights(model, base_state_cpu)
            t0 = time.perf_counter()
            prune_meta = apply_pruning_to_model(
                model, method=method, sparsity_map=target_s,
                input_l2=None if method == "magnitude" else probes["input_l2"],
            )
            t_comp = time.perf_counter() - t0
            storage = compute_pruned_model_storage(model)
            per_proj = storage.pop("per_projection")
            evaluate_and_save_condition(
                cid, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                hellaswag_full=hs_full, storage_info=storage, compression_time_sec=t_comp,
                extra_meta={"method": method, "target_sparsity": target_s, "per_projection": per_proj},
                eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                skip_bench=args.skip_bench,
            )

    # -----------------------------------------------------------------------
    # Phase 3: P3 Nonuniform Pruning (Dev selection on C4 dev 32x512 -> Test)
    # -----------------------------------------------------------------------
    p3_dev_path = args.output_dir / "dev_selection" / "p3_candidates.json"
    if run_all or "prune_dev" in phases:
        if not (resume and p3_dev_path.exists()):
            restore_base_weights(model, base_state_cpu)
            projs = enumerate_eligible_projections(model)
            candidates = []
            for target_s in (0.50, 0.75):
                # Candidate 1 & 2: OWL block-level (Yin et al. 2024)
                for lam in (0.05, 0.08, 0.12):
                    s_map, sens = allocate_owl_sparsities(projs, probes["input_l2"], target_s, outlier_m=5.0, lam=lam, unit="block")
                    candidates.append((f"owl_block_lam{lam}", target_s, s_map, sens, "block"))
                # Candidate 3: OWL projection-level
                s_map, sens = allocate_owl_sparsities(projs, probes["input_l2"], target_s, outlier_m=5.0, lam=0.08, unit="projection")
                candidates.append(("owl_proj_lam0.08", target_s, s_map, sens, "projection"))
                # Candidate 4: ShortGPT BI block-level (Men et al. 2025)
                s_map, sens = allocate_shortgpt_bi_sparsities(projs, probes["block_bi"], target_s, lam=0.15)
                candidates.append(("shortgpt_bi_lam0.15", target_s, s_map, sens, "block"))
                # Candidate 5: Unweighted Wanda local energy water-filling (projection-level)
                ones_sens = {p.name: 1.0 for p in projs}
                s_map, sens = allocate_marginal_distortion_sparsities(
                    projs, probes["input_l2"], ones_sens, target_s, normalize_by_tot_sq=True
                )
                candidates.append(("marginal_local_wanda", target_s, s_map, sens, "projection"))
                # Candidate 6: Output-Fisher-weighted Wanda water-filling (projection-level)
                s_map, sens = allocate_marginal_distortion_sparsities(
                    projs, probes["input_l2"], probes["output_grad_msq"], target_s, normalize_by_tot_sq=False
                )
                candidates.append(("marginal_fisher_wanda", target_s, s_map, sens, "projection"))
                # Candidate 7: Weight-Taylor-weighted Wanda water-filling (projection-level)
                s_map, sens = allocate_marginal_distortion_sparsities(
                    projs, probes["input_l2"], probes["weight_taylor_sum"], target_s, normalize_by_tot_sq=True
                )
                candidates.append(("marginal_taylor_wanda", target_s, s_map, sens, "projection"))

            dev_records = []
            for cname, target_s, s_map, sens, unit_kind in candidates:
                restore_base_weights(model, base_state_cpu)
                apply_pruning_to_model(model, method="wanda", sparsity_map=s_map, input_l2=probes["input_l2"])
                st = compute_pruned_model_storage(model)
                dev_res = evaluate_perplexity_windows(
                    model, datasets["c4_dev_windows"], device=device, batch_size=args.eval_batch_size, expected_total_tokens=None
                )
                dev_records.append({
                    "candidate": cname,
                    "target_sparsity": target_s,
                    "achieved_sparsity": st["achieved_eligible_sparsity"],
                    "unit": unit_kind,
                    "c4_dev_nll": dev_res["mean_nll"],
                    "c4_dev_ppl": dev_res["perplexity"],
                    "sparsity_map": s_map,
                    "sensitivity": sens,
                })
                print(f"  [P3 dev] s={target_s:.2f} {cname:24s} -> C4 dev PPL={dev_res['perplexity']:.4f} "
                      f"(achieved s={st['achieved_eligible_sparsity']:.6f})", flush=True)

            # Pick best rule per average dev NLL across 50% and 75% (so one consistent rule is chosen)
            by_name: dict[str, list[dict[str, Any]]] = {}
            for r in dev_records:
                by_name.setdefault(r["candidate"], []).append(r)
            winner = min(
                by_name.keys(),
                key=lambda k: sum(
                    x["c4_dev_nll"] if math.isfinite(x["c4_dev_nll"]) else 1e9 for x in by_name[k]
                ),
            )
            save_json(p3_dev_path, {"selected_candidate": winner, "candidates": dev_records})
            print(f"  >>> Selected P3 rule from C4 dev: {winner}", flush=True)

    if run_all or "prune_p3" in phases:
        p3_dev = load_json(p3_dev_path)
        winner = p3_dev["selected_candidate"]
        for target_s, cid, hs_full in [(0.50, "p3_nonuniform_s50", True), (0.75, "p3_nonuniform_s75", False)]:
            if resume and (args.output_dir / "conditions" / f"{cid}.json").exists():
                continue
            rec = next(x for x in p3_dev["candidates"] if x["candidate"] == winner and abs(x["target_sparsity"] - target_s) < 1e-6)
            restore_base_weights(model, base_state_cpu)
            t0 = time.perf_counter()
            apply_pruning_to_model(model, method="wanda", sparsity_map=rec["sparsity_map"], input_l2=probes["input_l2"])
            t_comp = time.perf_counter() - t0
            storage = compute_pruned_model_storage(model)
            per_proj = storage.pop("per_projection")
            evaluate_and_save_condition(
                cid, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                hellaswag_full=hs_full, storage_info=storage, compression_time_sec=t_comp,
                extra_meta={
                    "selected_candidate": winner,
                    "unit": rec["unit"],
                    "target_sparsity": target_s,
                    "sparsity_map": rec["sparsity_map"],
                    "sensitivity": rec["sensitivity"],
                    "per_projection": per_proj,
                },
                eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                skip_bench=args.skip_bench,
            )

    # -----------------------------------------------------------------------
    # Phase 4: Q1 (Uniform PTQ) & Q2 (AWQ-style PTQ) at INT8, INT4, INT3
    # -----------------------------------------------------------------------
    if run_all or "quant_uniform" in phases:
        for bits, hs_full in [(8, False), (4, True), (3, True)]:
            cid_q1 = f"q1_int{bits}"
            if not (resume and (args.output_dir / "conditions" / f"{cid_q1}.json").exists()):
                restore_base_weights(model, base_state_cpu)
                t0 = time.perf_counter()
                q1_sum = apply_q1_uniform(model, bits=bits, group_size=128)
                t_comp = time.perf_counter() - t0
                per_proj = q1_sum.pop("per_projection")
                evaluate_and_save_condition(
                    cid_q1, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                    hellaswag_full=hs_full, storage_info=q1_sum, compression_time_sec=t_comp,
                    extra_meta={"per_projection": per_proj},
                    eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                    skip_bench=args.skip_bench,
                )

            cid_q2 = f"q2_awq_int{bits}"
            if not (resume and (args.output_dir / "conditions" / f"{cid_q2}.json").exists()):
                restore_base_weights(model, base_state_cpu)
                t0 = time.perf_counter()
                q2_sum = apply_q2_awq(
                    model, c4_cal_seqs, bits_map=bits, device=device, group_size=128, batch_size=args.cal_batch_size
                )
                t_comp = time.perf_counter() - t0
                per_proj = q2_sum.pop("per_projection")
                alphas = q2_sum.pop("awq_search_alphas")
                evaluate_and_save_condition(
                    cid_q2, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                    hellaswag_full=hs_full, storage_info=q2_sum, compression_time_sec=t_comp,
                    extra_meta={"awq_search_alphas": alphas, "per_projection": per_proj},
                    eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                    skip_bench=args.skip_bench,
                )

    # -----------------------------------------------------------------------
    # Phase 5: Q3 Mixed-Precision Dev Selection -> Test
    # -----------------------------------------------------------------------
    q3_dev_path = args.output_dir / "dev_selection" / "q3_candidates.json"
    if run_all or "quant_dev" in phases:
        if not (resume and q3_dev_path.exists()):
            q2_int4_bytes = load_json(args.output_dir / "conditions" / "q2_awq_int4.json")["storage"]["total_packed_bytes"]
            q2_int3_bytes = load_json(args.output_dir / "conditions" / "q2_awq_int3.json")["storage"]["total_packed_bytes"]

            restore_base_weights(model, base_state_cpu)
            projs = enumerate_eligible_projections(model)
            dist = probes["quant_distortion_matrix"]

            # Build normalized distortion option for milp_rel_mse
            rel_dist = {}
            for p in projs:
                d8 = max(dist[p.name][8], 1e-12)
                norm_base = max(dist[p.name][2], d8)
                rel_dist[p.name] = {b: dist[p.name][b] / norm_base for b in (2, 3, 4, 8)}

            sens_designs = {
                "milp_unweighted_mse": (dist, {p.name: 1.0 for p in projs}),
                "milp_rel_mse": (rel_dist, {p.name: 1.0 for p in projs}),
                "milp_output_fisher": (dist, probes["output_grad_msq"]),
                "milp_taylor": (rel_dist, probes["weight_taylor_sum"]),
            }

            q3_dev_records = []
            for label, budget in [("int4_budget", q2_int4_bytes), ("int3_budget", q2_int3_bytes)]:
                for sname, (dmat, sw) in sens_designs.items():
                    bmap, milp_info = allocate_mixed_precision_milp(
                        projs, dmat, sw,
                        packed_byte_budget=budget,
                        uncompressed_fp16_bytes=acct["uncompressed_fp16_bytes"],
                        candidate_bits=(2, 3, 4, 8),
                        group_size=128,
                        unfolded_proj_types=("o_proj",),
                    )
                    restore_base_weights(model, base_state_cpu)
                    q3_sum = apply_q2_awq(
                        model, c4_cal_seqs, bits_map=bmap, device=device, group_size=128, batch_size=args.cal_batch_size
                    )
                    if q3_sum["total_packed_bytes"] > budget:
                        raise RuntimeError(f"Q3 exceeded budget: {q3_sum['total_packed_bytes']} > {budget}")
                    dev_res = evaluate_perplexity_windows(
                        model, datasets["c4_dev_windows"], device=device, batch_size=args.eval_batch_size, expected_total_tokens=None
                    )
                    q3_dev_records.append({
                        "candidate": sname,
                        "budget_label": label,
                        "budget_bytes": budget,
                        "allocated_bytes": q3_sum["total_packed_bytes"],
                        "c4_dev_nll": dev_res["mean_nll"],
                        "c4_dev_ppl": dev_res["perplexity"],
                        "bit_histogram": milp_info["bit_histogram"],
                        "bit_map": bmap,
                        "sensitivity": sw,
                    })
                    print(
                        f"  [Q3 dev] {label} {sname:20s} -> C4 dev PPL={dev_res['perplexity']:.4f} "
                        f"| bits={milp_info['bit_histogram']} | bytes={q3_sum['total_packed_bytes']:,}/{budget:,}",
                        flush=True,
                    )

            by_name_q: dict[str, list[dict[str, Any]]] = {}
            for r in q3_dev_records:
                by_name_q.setdefault(r["candidate"], []).append(r)
            winner_q = min(
                by_name_q.keys(),
                key=lambda k: sum(
                    x["c4_dev_nll"] if math.isfinite(x["c4_dev_nll"]) else 1e9 for x in by_name_q[k]
                ),
            )
            save_json(q3_dev_path, {"selected_candidate": winner_q, "candidates": q3_dev_records})
            print(f"  >>> Selected Q3 rule from C4 dev: {winner_q}", flush=True)

    if run_all or "quant_q3" in phases:
        q3_dev = load_json(q3_dev_path)
        winner_q = q3_dev["selected_candidate"]
        for label, cid in [("int4_budget", "q3_mixed_int4"), ("int3_budget", "q3_mixed_int3")]:
            if resume and (args.output_dir / "conditions" / f"{cid}.json").exists():
                continue
            rec = next(x for x in q3_dev["candidates"] if x["candidate"] == winner_q and x["budget_label"] == label)
            restore_base_weights(model, base_state_cpu)
            t0 = time.perf_counter()
            q3_sum = apply_q2_awq(
                model, c4_cal_seqs, bits_map=rec["bit_map"], device=device, group_size=128, batch_size=args.cal_batch_size
            )
            t_comp = time.perf_counter() - t0
            if q3_sum["total_packed_bytes"] > rec["budget_bytes"]:
                raise RuntimeError(f"Q3 budget violation: {q3_sum['total_packed_bytes']} > {rec['budget_bytes']}")
            per_proj = q3_sum.pop("per_projection")
            alphas = q3_sum.pop("awq_search_alphas")
            evaluate_and_save_condition(
                cid, model, ref_model_cpu, tok, datasets, args.output_dir, device,
                hellaswag_full=True, storage_info=q3_sum, compression_time_sec=t_comp,
                extra_meta={
                    "selected_candidate": winner_q,
                    "budget_label": label,
                    "budget_bytes": rec["budget_bytes"],
                    "bit_histogram": rec["bit_histogram"],
                    "bit_map": rec["bit_map"],
                    "sensitivity": rec["sensitivity"],
                    "awq_search_alphas": alphas,
                    "per_projection": per_proj,
                },
                eval_batch_size=args.eval_batch_size, hs_batch_size=args.hs_batch_size,
                skip_bench=args.skip_bench,
            )

    print("\nALL REQUESTED TASK 1 PHASES COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
