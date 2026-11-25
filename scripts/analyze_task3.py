"""Analyze Task 3 results: distillation costs, prefix alignment, speculative speedups, and bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.bootstrap import paired_bootstrap_mean_diff
from src.task3.speculative import compute_analytical_speculative_model


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task3-dir", type=Path, default=ROOT / "results" / "task3")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "task3" / "analysis")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots = args.out_dir / "plots"
    plots.mkdir(exist_ok=True)

    # 1. Distillation summary & learning curves
    dist_rows = []
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for dname in ("offpolicy_s6304", "offpolicy_s6305", "onpolicy_s6304", "onpolicy_s6305"):
        p = args.task3_dir / "distill" / f"{dname}.json"
        if not p.exists():
            continue
        d = load_json(p)
        t = d["timing_sec"]
        tot_wall = t["total_train_loop_sec"] + t.get("teacher_pool_prep_sec", 0.0)
        dist_rows.append({
            "run": dname,
            "mode": d["mode"],
            "seed": d["seed"],
            "trainable_params": d["param_counts"]["trainable_lora_params"],
            "math_scored": d["math_scored_tokens"],
            "short_scored": d["short_scored_tokens"],
            "optimizer_steps": d["optimizer_steps"],
            "skipped_steps": d["skipped_overflow_steps"],
            "teacher_pool_prep_s": t.get("teacher_pool_prep_sec", 0.0),
            "student_rollout_s": t["student_rollout_sec"],
            "teacher_scoring_s": t["teacher_scoring_sec"],
            "student_fwd_bwd_opt_s": t["student_fwd_bwd_opt_sec"],
            "total_cost_s": tot_wall,
            "peak_vram_mb": d.get("peak_allocated_bytes", 0) / 1024**2,
            "final_kl": d["learning_curve"][-1]["mean_kl_loss"],
        })
        xs = [pt["teacher_scored_tokens"] for pt in d["learning_curve"]]
        ys = [pt["mean_kl_loss"] for pt in d["learning_curve"]]
        ax.plot(xs, ys, label=dname, linewidth=1.8)
    if dist_rows:
        with open(args.out_dir / "distillation_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()))
            w.writeheader()
            w.writerows(dist_rows)
        ax.set_xlabel("Teacher-Scored Response Tokens")
        ax.set_ylabel("Mean Forward-KL Loss")
        ax.set_title("Off-Policy vs On-Policy Distillation Curves (40k Token Quota)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "fig_task3_distill_curves.png", dpi=200)
        fig.savefig(plots / "fig_task3_distill_curves.pdf")
    plt.close(fig)

    # 2. Common-prefix distribution alignment summary (token-weighted across bins)
    align_path = args.task3_dir / "prefix_alignment_metrics.json"
    if align_path.exists():
        align = load_json(align_path)["metrics_by_cell"]
        agg: dict[tuple[str, str, str], list[float]] = {}
        for r in align:
            for sub in (r["subsplit"], "ALL"):
                k = (r["draft"], r["bank"], sub)
                c = r["count"]
                a = agg.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                a[0] += r["forward_kl"] * c
                a[1] += r["reverse_kl"] * c
                a[2] += r["tv"] * c
                a[3] += r["expected_acceptance"] * c
                a[4] += r["greedy_agreement"] * c
                a[5] += c
        arows = [
            {
                "draft": d, "bank": b, "subsplit": s, "n_positions": int(v[5]),
                "forward_kl": v[0] / v[5], "reverse_kl": v[1] / v[5], "tv": v[2] / v[5],
                "expected_acceptance": v[3] / v[5], "greedy_agreement": v[4] / v[5],
            }
            for (d, b, s), v in sorted(agg.items())
        ]
        with open(args.out_dir / "prefix_alignment_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(arows[0].keys()))
            w.writeheader()
            w.writerows(arows)

    # 3. Speculative decoding benchmark summary + analytical S_hat + bootstrap CIs + amortization
    iso_path = args.task3_dir / "spec_bench" / "isolated_step_costs.json"
    iso = load_json(iso_path)["by_prompt_len"] if iso_path.exists() else []
    iso_mid = iso[1] if len(iso) > 1 else (iso[0] if iso else None)

    drafts = ("original", "offpolicy_s6304", "offpolicy_s6305", "onpolicy_s6304", "onpolicy_s6305")
    spec_rows = []
    loaded_spec: dict[tuple[str, int], list[dict[str, Any]]] = {}
    dist_cost_map = {r["run"]: r["total_cost_s"] for r in dist_rows}
    merge_path = args.task3_dir / "merge_verification.json"
    merge_info = load_json(merge_path) if merge_path.exists() else {}

    for dname in drafts:
        bp = args.task3_dir / "spec_bench" / f"spec120_{dname}.jsonl.gz"
        if not bp.exists():
            continue
        all_r = load_jsonl_gz(bp)
        for gamma in (2, 4):
            sub = [x for x in all_r if x["gamma"] == gamma]
            loaded_spec[(dname, gamma)] = sub
            # Aggregate over prompts (mean of the 2 reps per prompt for paired comparisons)
            tot_ex = sum(x["speculative"]["examined_tokens"] for x in sub)
            tot_prop = sum(x["speculative"]["proposed_tokens"] for x in sub)
            tot_acc = sum(x["speculative"]["accepted_proposed"] for x in sub)
            tot_toks_spec = sum(x["speculative"]["n_generated"] for x in sub)
            tot_ver_passes = sum(x["speculative"]["target_verify_passes"] for x in sub)
            tot_draft_passes = sum(x["speculative"]["draft_passes"] for x in sub)
            tot_t_ms = sum(x["target_only"]["total_ms"] for x in sub)
            tot_s_ms = sum(x["speculative"]["total_ms"] for x in sub)
            dec_t_ms = sum(x["target_only"]["decode_ms"] for x in sub)
            dec_s_ms = sum(x["speculative"]["decode_ms"] for x in sub)

            emp_acc = tot_acc / max(tot_ex, 1)
            c_ratio = iso_mid["c_ratio"] if iso_mid else 0.0
            v_gamma = iso_mid["verify_passes"][str(gamma)]["v_gamma"] if iso_mid else 1.0
            ana = compute_analytical_speculative_model(emp_acc, c_ratio, v_gamma, gamma)

            # Upfront distillation + merge cost amortization
            upfront_s = dist_cost_map.get(dname, 0.0) + merge_info.get(dname, {}).get("merge_time_sec", 0.0)
            saved_per_req_s = (tot_t_ms - tot_s_ms) / max(len(sub), 1) / 1000.0
            breakeven_reqs = (upfront_s / saved_per_req_s) if saved_per_req_s > 0 and upfront_s > 0 else None

            spec_rows.append({
                "draft": dname,
                "gamma": gamma,
                "n_evals": len(sub),
                "acc_over_examined": emp_acc,
                "acc_over_proposed": tot_acc / max(tot_prop, 1),
                "mean_accepted_prefix": float(np.mean([x["speculative"]["mean_accepted_prefix_per_verify"] for x in sub])),
                "target_passes_per_tok": tot_ver_passes / max(tot_toks_spec, 1),
                "draft_passes_per_tok": tot_draft_passes / max(tot_toks_spec, 1),
                "target_only_tok_s": sum(x["target_only"]["n_generated"] for x in sub) / max(tot_t_ms / 1000.0, 1e-9),
                "spec_tok_s": tot_toks_spec / max(tot_s_ms / 1000.0, 1e-9),
                "measured_speedup_total": tot_t_ms / max(tot_s_ms, 1e-6),
                "measured_speedup_decode": dec_t_ms / max(dec_s_ms, 1e-6),
                "predicted_S_hat": ana["predicted_speedup_S_hat"],
                "break_even_alpha": ana["break_even_alpha"],
                "eos_rate": float(np.mean([x["speculative"]["terminated_by_eos"] for x in sub])),
                "truncation_rate": float(np.mean([x["speculative"]["truncated_by_cap"] for x in sub])),
                "draft_proposal_ms_mean": float(np.mean([x["speculative"]["round_timing_ms"]["draft_proposal_ms"] for x in sub])),
                "target_verify_ms_mean": float(np.mean([x["speculative"]["round_timing_ms"]["target_verify_ms"] for x in sub])),
                "accept_residual_ms_mean": float(np.mean([x["speculative"]["round_timing_ms"]["accept_residual_ms"] for x in sub])),
                "cache_update_ms_mean": float(np.mean([x["speculative"]["round_timing_ms"]["cache_update_ms"] for x in sub])),
                "upfront_distill_merge_s": upfront_s,
                "saved_sec_per_req": saved_per_req_s,
                "amortize_breakeven_requests": breakeven_reqs,
            })

    if spec_rows:
        with open(args.out_dir / "speculative_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(spec_rows[0].keys()))
            w.writeheader()
            w.writerows(spec_rows)

    # Paired bootstrap CIs over the 120 prompts (averaging 2 reps per prompt)
    def _prompt_means(sub: list[dict[str, Any]], key: str) -> list[float]:
        by_p: dict[str, list[float]] = {}
        for x in sub:
            val = x["speculative"][key] if key != "speedup_total" else x["speedup_total"]
            by_p.setdefault(x["prompt_id"], []).append(float(val))
        return [float(np.mean(by_p[pid])) for pid in sorted(by_p)]

    boot_pairs = []
    for gamma in (2, 4):
        for da, db in (
            ("offpolicy_s6304", "original"),
            ("onpolicy_s6304", "original"),
            ("onpolicy_s6304", "offpolicy_s6304"),
            ("offpolicy_s6305", "original"),
            ("onpolicy_s6305", "original"),
            ("onpolicy_s6305", "offpolicy_s6305"),
        ):
            if (da, gamma) in loaded_spec and (db, gamma) in loaded_spec:
                boot_pairs.append({
                    "comparison": f"{da}_vs_{db}_g{gamma}",
                    "accepted_over_examined": paired_bootstrap_mean_diff(
                        _prompt_means(loaded_spec[(da, gamma)], "accepted_over_examined"),
                        _prompt_means(loaded_spec[(db, gamma)], "accepted_over_examined"),
                        n_resamples=args.n_resamples,
                    ),
                    "total_tokens_per_sec": paired_bootstrap_mean_diff(
                        _prompt_means(loaded_spec[(da, gamma)], "total_tokens_per_sec"),
                        _prompt_means(loaded_spec[(db, gamma)], "total_tokens_per_sec"),
                        n_resamples=args.n_resamples,
                    ),
                })
    (args.out_dir / "task3_bootstrap_ci.json").write_text(json.dumps(boot_pairs, indent=2), encoding="utf-8")

    # 4. Standalone student accuracy summary
    st_rows = []
    for mname in ("target_1.5B", *drafts):
        sp = args.task3_dir / "standalone" / f"{mname}.jsonl.gz"
        if not sp.exists():
            continue
        items = load_jsonl_gz(sp)
        for subsplit in ("math", "short_familiar", "short_shifted"):
            sub = [x for x in items if x["subsplit"] == subsplit]
            if not sub:
                continue
            st_rows.append({
                "model": mname,
                "subsplit": subsplit,
                "n": len(sub),
                "accuracy": float(np.mean([x["correct"] for x in sub])),
                "format_compliance": float(np.mean([x["format_compliant"] for x in sub])),
                "truncation_rate": float(np.mean([x["truncated"] for x in sub])),
                "unextractable_rate": float(np.mean([x["unextractable"] for x in sub])),
            })
    if st_rows:
        with open(args.out_dir / "standalone_accuracy_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(st_rows[0].keys()))
            w.writeheader()
            w.writerows(st_rows)

    print(f"Task 3 analysis written to {args.out_dir}")


if __name__ == "__main__":
    main()
