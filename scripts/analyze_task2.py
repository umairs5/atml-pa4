"""Analyze Task 2 outputs (Parts A, B, C) into summary CSVs, paired bootstrap CIs, and plots."""

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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Part A Analysis
# ---------------------------------------------------------------------------

def analyze_part_a(task2_dir: Path, out_dir: Path) -> None:
    p = task2_dir / "part_a" / "part_a_results.json"
    if not p.exists():
        return
    data = load_json(p)
    rows = data["benchmarks"]

    csv_rows = []
    for r in rows:
        probe = (r.get("sdpa_probe") or {})
        csv_rows.append({
            "L": r["L"],
            "dtype": r["dtype"],
            "operator": r["operator"],
            "fwd_status": r["fwd_status"],
            "fwdbwd_status": r["fwdbwd_status"],
            "out_max_abs_err": r.get("out_max_abs_err"),
            "dq_max_abs_err": r.get("dq_max_abs_err"),
            "dk_max_abs_err": r.get("dk_max_abs_err"),
            "dv_max_abs_err": r.get("dv_max_abs_err"),
            "fwd_ms": (r.get("fwd_latency") or {}).get("mean_ms"),
            "fwdbwd_ms": (r.get("fwdbwd_latency") or {}).get("mean_ms"),
            "fwd_peak_alloc_mb": ((r.get("fwd_memory") or {}).get("peak_allocated_bytes", 0) / 1024**2) if r.get("fwd_memory") else None,
            "fwdbwd_peak_alloc_mb": ((r.get("fwdbwd_memory") or {}).get("peak_allocated_bytes", 0) / 1024**2) if r.get("fwdbwd_memory") else None,
            "sdpa_backend": probe.get("inferred_default_backend"),
        })
    if csv_rows:
        with open(out_dir / "part_a_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    # Plot latency and peak allocated memory vs sequence length (FP16 and FP32)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, dt in enumerate(("fp16", "fp32")):
        for op, fmt in (("dense", "o--"), ("tiled_64", "s-"), ("sdpa", "^-")):
            sub = [r for r in csv_rows if r["dtype"] == dt and r["operator"] == op]
            sub_f = [r for r in sub if r["fwd_ms"] is not None]
            sub_b = [r for r in sub if r["fwdbwd_ms"] is not None]
            if sub_f:
                axes[0, col].plot([r["L"] for r in sub_f], [r["fwd_ms"] for r in sub_f], fmt, label=f"{op} fwd")
                axes[1, col].plot([r["L"] for r in sub_f], [r["fwd_peak_alloc_mb"] for r in sub_f], fmt, label=f"{op} fwd")
            if sub_b:
                axes[0, col].plot([r["L"] for r in sub_b], [r["fwdbwd_ms"] for r in sub_b], fmt, alpha=0.65, label=f"{op} fwd+bwd")
                axes[1, col].plot([r["L"] for r in sub_b], [r["fwdbwd_peak_alloc_mb"] for r in sub_b], fmt, alpha=0.65, label=f"{op} fwd+bwd")
        axes[0, col].set_title(f"Latency vs Sequence Length ({dt.upper()})")
        axes[0, col].set_xscale("log", base=2)
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("Latency (ms)")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_title(f"Peak Allocated GPU Memory ({dt.upper()})")
        axes[1, col].set_xscale("log", base=2)
        axes[1, col].set_ylabel("Allocated Memory (MiB)")
        axes[1, col].set_xlabel("Sequence Length L")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "fig_task2a_scaling.png", dpi=200)
    fig.savefig(plots / "fig_task2a_scaling.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part B Analysis
# ---------------------------------------------------------------------------

def analyze_part_b(task2_dir: Path, out_dir: Path) -> None:
    runs_dir = task2_dir / "part_b" / "runs"
    if not runs_dir.exists():
        return
    all_runs = [load_json(p) for p in sorted(runs_dir.glob("*.json"))]
    if not all_runs:
        return

    # Group across the 3 seeds by (group, mechanism, train_k, train_regime)
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for r in all_runs:
        key = (r["group"], r["mechanism"], r["train_k"], r["train_regime"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    bins = ("[0,32)", "[32,64)", "[64,128)", "[128,256)")
    for (grp, mech, k, reg), items in sorted(grouped.items()):
        near_accs = [x["test_eval"]["by_cell"][f"K{k}_near"]["acc"] for x in items]
        far_accs = [x["test_eval"]["by_cell"][f"K{k}_far"]["acc"] for x in items]
        mix_accs = [x["test_eval"]["by_cell"][f"K{k}_mixed"]["acc"] for x in items]
        row = {
            "group": grp,
            "mechanism": mech,
            "train_k": k,
            "train_regime": reg,
            "n_seeds": len(items),
            "trainable_params": items[0]["trainable_params"],
            "near_acc_mean": float(np.mean(near_accs)),
            "near_acc_std": float(np.std(near_accs, ddof=1)) if len(items) > 1 else 0.0,
            "far_acc_mean": float(np.mean(far_accs)),
            "far_acc_std": float(np.std(far_accs, ddof=1)) if len(items) > 1 else 0.0,
            "mixed_acc_mean": float(np.mean(mix_accs)),
            "mixed_acc_std": float(np.std(mix_accs, ddof=1)) if len(items) > 1 else 0.0,
            "chance_1_over_64": 1.0 / 64.0,
            "chance_1_over_K": 1.0 / k,
        }
        for b in bins:
            # Compute bin accuracy on K-matched test evaluation? Note test_eval aggregates all K;
            # compute across K=train_k if desired or overall test_eval bin
            vals = [x["test_eval"]["by_distance_bin"][b]["acc"] for x in items if x["test_eval"]["by_distance_bin"][b]["acc"] is not None]
            row[f"bin_{b}_mean"] = float(np.mean(vals)) if vals else None

        if items[0].get("test_eval_w64_mask_control") is not None:
            c_near = [x["test_eval_w64_mask_control"]["by_cell"][f"K{k}_near"]["acc"] for x in items]
            c_far = [x["test_eval_w64_mask_control"]["by_cell"][f"K{k}_far"]["acc"] for x in items]
            row["w64_ctrl_near_mean"] = float(np.mean(c_near))
            row["w64_ctrl_far_mean"] = float(np.mean(c_far))
        else:
            row["w64_ctrl_near_mean"] = None
            row["w64_ctrl_far_mean"] = None
        summary_rows.append(row)

    with open(out_dir / "part_b_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Learning curves at K=16 mixed across architectures
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for mech, label in (
        ("dense", "Dense Causal"),
        ("sliding_w64", "Sliding Window W=64"),
        ("local_dilated", "Mixed Local/Dilated"),
        ("linear_elu1", "Linear (ELU+1)"),
        ("topk_64", "B3 Content Top-k (k=64)"),
    ):
        items = grouped.get(("B1", mech, 16, "mixed")) or grouped.get(("B2", mech, 16, "mixed")) or grouped.get(("B3", mech, 16, "mixed"))
        if not items:
            continue
        steps = [pt["step"] for pt in items[0]["learning_curve"]]
        curves = np.array([[pt["dev_acc"] * 100 for pt in it["learning_curve"]] for it in items])
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0)
        ax.plot(steps, mean_c, marker="o", label=label, linewidth=2)
        ax.fill_between(steps, mean_c - std_c, mean_c + std_c, alpha=0.15)
    ax.set_title("MQAR Learning Curves (K=16, Mixed Training, Dev Set)")
    ax.set_xlabel("Optimizer Update Step")
    ax.set_ylabel("Dev Query Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "fig_task2b_learning_curves_k16.png", dpi=200)
    fig.savefig(plots / "fig_task2b_learning_curves_k16.pdf")
    plt.close(fig)

    # Distance x Load Heatmap for trained-on-mixed models at K=4,8,16
    mechs_order = [
        ("dense", 4), ("dense", 8), ("dense", 16),
        ("sliding_w64", 16),
        ("local_dilated", 4), ("local_dilated", 16),
        ("linear_elu1", 4), ("linear_elu1", 8), ("linear_elu1", 16),
        ("topk_64", 16),
    ]
    mat = []
    ylabels = []
    for mech, k in mechs_order:
        items = [r for r in all_runs if r["mechanism"] == mech and r["train_k"] == k and r["train_regime"] == "mixed"]
        if not items:
            continue
        ylabels.append(f"{mech} (K={k})")
        mat.append([
            np.mean([x["test_eval"]["by_cell"][f"K{k}_{reg}"]["acc"] * 100 for x in items])
            for reg in ("near", "mixed", "far")
        ])
    if mat:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(np.array(mat), aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Near (<64)", "Mixed", "Far (>=128)"])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_title("MQAR Held-Out Accuracy (%) by Model & Distance Regime")
        for i in range(len(mat)):
            for j in range(3):
                ax.text(j, i, f"{mat[i][j]:.1f}", ha="center", va="center", color="black" if mat[i][j] < 60 else "white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(plots / "fig_task2b_distance_load_heatmap.png", dpi=200)
        fig.savefig(plots / "fig_task2b_distance_load_heatmap.pdf")
        plt.close(fig)

    # Inference sweep plot
    sw_path = task2_dir / "part_b" / "inference_sweep.json"
    if sw_path.exists():
        sw = load_json(sw_path)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
        for mech in ("dense", "sliding_w64", "local_dilated", "linear_elu1", "topk_64"):
            pts = [x for x in sw if x["mechanism"] == mech and x["status"] == "ok"]
            if not pts:
                continue
            axes[0].plot([x["T"] for x in pts], [x["fwd_latency"]["mean_ms"] for x in pts], marker="o", label=mech)
            axes[1].plot([x["T"] for x in pts], [x["fwd_memory"]["peak_allocated_bytes"] / 1024**2 for x in pts], marker="o", label=mech)
        axes[0].set_xscale("log", base=2); axes[0].set_yscale("log"); axes[0].set_title("Inference Forward Latency"); axes[0].set_ylabel("Latency (ms)"); axes[0].set_xlabel("T"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
        axes[1].set_xscale("log", base=2); axes[1].set_title("Inference Peak Allocated GPU Memory"); axes[1].set_ylabel("MiB"); axes[1].set_xlabel("T"); axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "fig_task2b_inference_sweep.png", dpi=200)
        fig.savefig(plots / "fig_task2b_inference_sweep.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Part C Analysis
# ---------------------------------------------------------------------------

def analyze_part_c(task2_dir: Path, out_dir: Path, n_resamples: int = 2000) -> None:
    pol_dir = task2_dir / "part_c" / "policies"
    if not pol_dir.exists():
        return
    policies = (
        "full_fp16",
        "evict_sink_recent",
        "evict_question_aware",
        "quant_int4_tok",
        "quant_int4_kivi",
        "hybrid_fp16_int2",
    )
    loaded = {p: load_jsonl_gz(pol_dir / f"{p}.jsonl.gz") for p in policies if (pol_dir / f"{p}.jsonl.gz").exists()}
    if "full_fp16" not in loaded:
        return

    ref_rows = loaded["full_fp16"]
    ref_correct = [r["exact_match"] for r in ref_rows]

    summary_rows = []
    breakdown_rows = []
    for p, rows in loaded.items():
        acc = float(np.mean([r["exact_match"] for r in rows]))
        # Accuracy conditional on FP16 reference correctness (always keep wrong FP16 cases in unconditional!)
        cond_indices = [i for i, ok in enumerate(ref_correct) if ok == 1]
        cond_acc = float(np.mean([rows[i]["exact_match"] for i in cond_indices])) if cond_indices else None
        ev_means = [r["retained_evidence_mean"] for r in rows if r["retained_evidence_mean"] is not None]
        kls = [r["post_prefill_kl_fp16"] for r in rows if r["post_prefill_kl_fp16"] is not None]
        summary_rows.append({
            "policy": p,
            "n_prompts": len(rows),
            "unconditional_acc": acc,
            "conditional_on_fp16_acc": cond_acc,
            "mean_answer_nll": float(np.mean([r["reference_answer_nll"] for r in rows if r["reference_answer_nll"] is not None])),
            "mean_post_prefill_kl": float(np.mean(kls)) if kls else 0.0,
            "mean_retained_evidence": float(np.mean(ev_means)) if ev_means else None,
            "mean_ctx_packed_mb": float(np.mean([r["ctx_packed_bytes"] for r in rows])) / (1024**2),
            "mean_realized_fp16_frac": float(np.mean([r["realized_ctx_fraction_fp16"] for r in rows])),
            "mean_shared_decode_kb": float(np.mean([r["shared_decode_fp16_bytes"] for r in rows])) / 1024.0,
            "mean_prefill_ms": float(np.mean([r["prefill_sec"] for r in rows])) * 1000.0,
            "mean_score_compress_ms": float(np.mean([r["score_compress_sec"] for r in rows])) * 1000.0,
            "mean_decode_ms": float(np.mean([r["decode_sec"] for r in rows])) * 1000.0,
            "mean_peak_alloc_mb": float(np.mean([r["peak_allocated_bytes"] for r in rows])) / (1024**2),
        })

        # Breakdowns by target_tokens, position, distractors
        for dim_name in ("target_tokens", "position", "distractors"):
            vals = sorted({r[dim_name] for r in rows})
            for v in vals:
                sub = [r for r in rows if r[dim_name] == v]
                breakdown_rows.append({
                    "policy": p,
                    "factor": dim_name,
                    "level": v,
                    "n": len(sub),
                    "exact_match": float(np.mean([r["exact_match"] for r in sub])),
                    "mean_answer_nll": float(np.mean([r["reference_answer_nll"] for r in sub])),
                })

    with open(out_dir / "part_c_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with open(out_dir / "part_c_breakdowns.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
        w.writeheader()
        w.writerows(breakdown_rows)

    # Paired 95% bootstrap CIs (2,000 resamples over prompts)
    boot_pairs = [
        ("question_aware_vs_sink_recent", "evict_question_aware", "evict_sink_recent"),
        ("kivi_vs_tok_int4", "quant_int4_kivi", "quant_int4_tok"),
        ("hybrid_vs_question_aware", "hybrid_fp16_int2", "evict_question_aware"),
        ("hybrid_vs_kivi_int4", "hybrid_fp16_int2", "quant_int4_kivi"),
        ("kivi_vs_full_fp16", "quant_int4_kivi", "full_fp16"),
        ("hybrid_vs_full_fp16", "hybrid_fp16_int2", "full_fp16"),
    ]
    boot_out = {}
    for label, pa, pb in boot_pairs:
        if pa in loaded and pb in loaded:
            boot_out[label] = {
                "exact_match": paired_bootstrap_mean_diff(
                    [r["exact_match"] for r in loaded[pa]],
                    [r["exact_match"] for r in loaded[pb]],
                    n_resamples=n_resamples,
                ),
                "answer_nll": paired_bootstrap_mean_diff(
                    [r["reference_answer_nll"] for r in loaded[pa]],
                    [r["reference_answer_nll"] for r in loaded[pb]],
                    n_resamples=n_resamples,
                ),
            }
    (out_dir / "part_c_bootstrap_ci.json").write_text(json.dumps(boot_out, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-dir", type=Path, default=ROOT / "results" / "task2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "task2" / "analysis")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    analyze_part_a(args.task2_dir, args.out_dir)
    analyze_part_b(args.task2_dir, args.out_dir)
    analyze_part_c(args.task2_dir, args.out_dir)
    print(f"Task 2 analysis saved to {args.out_dir}")


if __name__ == "__main__":
    main()

