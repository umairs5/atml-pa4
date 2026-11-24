"""Generate Task 1 summary tables, paired 95% bootstrap CIs, plots, and qualitative cases."""

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

from src.common.bootstrap import paired_bootstrap_mean_diff, paired_bootstrap_ppl_diff
from src.task1.eval_hellaswag import load_hellaswag_examples
from src.task1.model_utils import ELIGIBLE_PROJ_NAMES

ORDERED_CONDITIONS = [
    "fp16_baseline",
    "p1_mag_s25",
    "p1_mag_s50",
    "p1_mag_s75",
    "p2_wanda_s25",
    "p2_wanda_s50",
    "p2_wanda_s75",
    "p2_wanda_2_4_s50",
    "p3_nonuniform_s50",
    "p3_nonuniform_s75",
    "q1_int8",
    "q1_int4",
    "q1_int3",
    "q2_awq_int8",
    "q2_awq_int4",
    "q2_awq_int3",
    "q3_mixed_int4",
    "q3_mixed_int3",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_summary_rows(results_dir: Path) -> list[dict[str, Any]]:
    cond_dir = results_dir / "conditions"
    base = load_json(cond_dir / "fp16_baseline.json")
    base_wt2 = base["wikitext2"]["perplexity"]
    base_c4 = base["c4_val"]["perplexity"]
    base_hs_512 = base["hellaswag"]["diag_512_acc_norm"]
    base_hs_2k = base["hellaswag"]["acc_norm"]

    rows = []
    for cid in ORDERED_CONDITIONS:
        p = cond_dir / f"{cid}.json"
        if not p.exists():
            continue
        d = load_json(p)
        st = d["storage"]
        is_quant = cid.startswith("q")
        packed_bytes = st["total_packed_bytes"] if is_quant else st.get("sparse_total_bytes", st["dense_fp16_bytes"])
        if cid == "fp16_baseline":
            packed_bytes = st["dense_fp16_bytes"]
        nom_bits = st.get("nominal_bits_per_eligible_weight", 16.0)
        eff_bits = st.get("effective_bits_per_eligible_weight", nom_bits)
        sparsity = st.get("achieved_eligible_sparsity", 0.0)
        bench = d.get("benchmark") or {}
        errs = d.get("layer_output_errors") or {}
        rows.append({
            "condition_id": cid,
            "wt2_ppl": d["wikitext2"]["perplexity"],
            "wt2_delta_ppl": d["wikitext2"]["perplexity"] - base_wt2,
            "wt2_nll": d["wikitext2"]["mean_nll"],
            "c4_ppl": d["c4_val"]["perplexity"],
            "c4_delta_ppl": d["c4_val"]["perplexity"] - base_c4,
            "c4_nll": d["c4_val"]["mean_nll"],
            "c4_dev_ppl": d["c4_dev"]["perplexity"],
            "hs_n": d["hellaswag"]["n_evaluated"],
            "hs_acc": d["hellaswag"]["acc"],
            "hs_acc_norm": d["hellaswag"]["acc_norm"],
            "hs_512_acc": d["hellaswag"]["diag_512_acc"],
            "hs_512_acc_norm": d["hellaswag"]["diag_512_acc_norm"],
            "hs_512_delta_norm_pp": (d["hellaswag"]["diag_512_acc_norm"] - base_hs_512) * 100.0,
            "hs_2k_delta_norm_pp": (d["hellaswag"]["acc_norm"] - base_hs_2k) * 100.0 if d["hellaswag"]["n_evaluated"] == 2000 else None,
            "achieved_sparsity": sparsity,
            "nominal_bits_elig": nom_bits,
            "effective_bits_elig": eff_bits,
            "packed_bytes": packed_bytes,
            "packed_mb": packed_bytes / (1024 ** 2),
            "compression_time_s": d["compression_time_sec"],
            "mean_proj_rel_l2": errs.get("mean_proj_rel_l2"),
            "final_block_cum_rel_l2": errs.get("final_block_cum_rel_l2"),
            "prefill_512_ms": bench.get("prefill_512", {}).get("mean_ms"),
            "prefill_2048_ms": bench.get("prefill_2048", {}).get("mean_ms"),
            "decode_64_ms": bench.get("decode_64", {}).get("mean_ms"),
            "peak_vram_mb": (bench.get("peak_allocated_bytes", 0) / (1024 ** 2)) if bench else None,
        })
    return rows


def compute_all_bootstraps(results_dir: Path, n_resamples: int = 2000) -> dict[str, Any]:
    pe_dir = results_dir / "per_example"
    pairs = [
        ("p3_vs_p2_s50", "p3_nonuniform_s50", "p2_wanda_s50"),
        ("p3_vs_p2_s75", "p3_nonuniform_s75", "p2_wanda_s75"),
        ("p2_vs_p1_s50", "p2_wanda_s50", "p1_mag_s50"),
        ("wanda_2_4_vs_p2_s50", "p2_wanda_2_4_s50", "p2_wanda_s50"),
        ("q3_vs_q2_int4", "q3_mixed_int4", "q2_awq_int4"),
        ("q3_vs_q2_int3", "q3_mixed_int3", "q2_awq_int3"),
        ("q2_vs_q1_int4", "q2_awq_int4", "q1_int4"),
        ("q2_vs_q1_int3", "q2_awq_int3", "q1_int3"),
    ]
    out: dict[str, Any] = {}
    for label, a_id, b_id in pairs:
        wt_a = pe_dir / f"{a_id}_ppl_wt2.json.gz"
        wt_b = pe_dir / f"{b_id}_ppl_wt2.json.gz"
        if not (wt_a.exists() and wt_b.exists()):
            continue
        wa, wb = load_json_gz(wt_a), load_json_gz(wt_b)
        ca, cb = load_json_gz(pe_dir / f"{a_id}_ppl_c4.json.gz"), load_json_gz(pe_dir / f"{b_id}_ppl_c4.json.gz")
        hsa, hsb = load_json_gz(pe_dir / f"{a_id}_hellaswag.json.gz"), load_json_gz(pe_dir / f"{b_id}_hellaswag.json.gz")

        entry: dict[str, Any] = {
            "wikitext2": paired_bootstrap_ppl_diff(
                wa["per_window_nll_sum"], wb["per_window_nll_sum"], wa["per_window_scored_tokens"], n_resamples=n_resamples
            ),
            "c4_val": paired_bootstrap_ppl_diff(
                ca["per_window_nll_sum"], cb["per_window_nll_sum"], ca["per_window_scored_tokens"], n_resamples=n_resamples
            ),
            "hellaswag_512_acc": paired_bootstrap_mean_diff(
                [x["correct_raw"] for x in hsa[:512]], [x["correct_raw"] for x in hsb[:512]], n_resamples=n_resamples
            ),
            "hellaswag_512_acc_norm": paired_bootstrap_mean_diff(
                [x["correct_norm"] for x in hsa[:512]], [x["correct_norm"] for x in hsb[:512]], n_resamples=n_resamples
            ),
        }
        if len(hsa) == 2000 and len(hsb) == 2000:
            entry["hellaswag_2000_acc"] = paired_bootstrap_mean_diff(
                [x["correct_raw"] for x in hsa], [x["correct_raw"] for x in hsb], n_resamples=n_resamples
            )
            entry["hellaswag_2000_acc_norm"] = paired_bootstrap_mean_diff(
                [x["correct_norm"] for x in hsa], [x["correct_norm"] for x in hsb], n_resamples=n_resamples
            )
        out[label] = entry
    return out


def plot_task1_figures(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["condition_id"]: r for r in rows}

    # 1. Pruning trade-off (WT2 PPL & HellaSwag 512 acc_norm vs Sparsity)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    groups = [
        ("P1 Magnitude", ["fp16_baseline", "p1_mag_s25", "p1_mag_s50", "p1_mag_s75"], "o--"),
        ("P2 Wanda", ["fp16_baseline", "p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75"], "s-"),
        ("P3 Nonuniform", ["fp16_baseline", "p3_nonuniform_s50", "p3_nonuniform_s75"], "^-"),
    ]
    for label, ids, fmt in groups:
        pts = [by_id[i] for i in ids if i in by_id]
        if not pts:
            continue
        xs = [p["achieved_sparsity"] * 100 for p in pts]
        axes[0].plot(xs, [p["wt2_ppl"] for p in pts], fmt, label=label, linewidth=2)
        axes[1].plot(xs, [p["hs_512_acc_norm"] * 100 for p in pts], fmt, label=label, linewidth=2)
    if "p2_wanda_2_4_s50" in by_id:
        p = by_id["p2_wanda_2_4_s50"]
        axes[0].scatter([50.0], [p["wt2_ppl"]], marker="D", s=70, zorder=5, label="Wanda 2:4 (50%)")
        axes[1].scatter([50.0], [p["hs_512_acc_norm"] * 100], marker="D", s=70, zorder=5, label="Wanda 2:4 (50%)")
    axes[0].set_title("WikiText-2 Perplexity vs Sparsity")
    axes[0].set_xlabel("Eligible-Weight Sparsity (%)")
    axes[0].set_ylabel("Perplexity (lower is better)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("HellaSwag (512 diag) acc_norm vs Sparsity")
    axes[1].set_xlabel("Eligible-Weight Sparsity (%)")
    axes[1].set_ylabel("acc_norm (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig1_pruning_tradeoff.png", dpi=200)
    fig.savefig(plots_dir / "fig1_pruning_tradeoff.pdf")
    plt.close(fig)

    # 2. Quantization trade-off (WT2 PPL & HellaSwag vs Packed MB)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    q_groups = [
        ("Q1 Uniform", ["fp16_baseline", "q1_int8", "q1_int4", "q1_int3"], "o--"),
        ("Q2 AWQ", ["fp16_baseline", "q2_awq_int8", "q2_awq_int4", "q2_awq_int3"], "s-"),
        ("Q3 Mixed", ["q3_mixed_int4", "q3_mixed_int3"], "^-"),
    ]
    for label, ids, fmt in q_groups:
        pts = [by_id[i] for i in ids if i in by_id]
        if not pts:
            continue
        xs = [p["packed_mb"] for p in pts]
        axes[0].plot(xs, [p["wt2_ppl"] for p in pts], fmt, label=label, linewidth=2)
        axes[1].plot(xs, [p["hs_512_acc_norm"] * 100 for p in pts], fmt, label=label, linewidth=2)
    axes[0].set_title("WikiText-2 Perplexity vs Total Packed Storage")
    axes[0].set_xlabel("Total Packed Size (MiB, incl. metadata & FP16)")
    axes[0].set_ylabel("Perplexity (lower is better)")
    axes[0].invert_xaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("HellaSwag (512 diag) acc_norm vs Packed Storage")
    axes[1].set_xlabel("Total Packed Size (MiB, incl. metadata & FP16)")
    axes[1].set_ylabel("acc_norm (%)")
    axes[1].invert_xaxis()
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig2_quant_tradeoff.png", dpi=200)
    fig.savefig(plots_dir / "fig2_quant_tradeoff.pdf")
    plt.close(fig)

    # 3. P3 Sparsity Map & Sensitivity Heatmap
    for cid, fname in [("p3_nonuniform_s50", "fig3_p3_s50_map"), ("p3_nonuniform_s75", "fig3_p3_s75_map")]:
        cpath = results_dir / "conditions" / f"{cid}.json"
        if not cpath.exists():
            continue
        d = load_json(cpath)
        smap = d["meta"]["sparsity_map"]
        sens = d["meta"]["sensitivity"]
        n_layers = 28
        mat_s = np.zeros((n_layers, len(ELIGIBLE_PROJ_NAMES)))
        for l in range(n_layers):
            for j, ptype in enumerate(ELIGIBLE_PROJ_NAMES):
                sub = "self_attn" if ptype in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                mat_s[l, j] = smap[f"model.layers.{l}.{sub}.{ptype}"] * 100.0
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        im = ax.imshow(mat_s, aspect="auto", cmap="YlOrRd", vmin=10, vmax=85)
        ax.set_xticks(range(len(ELIGIBLE_PROJ_NAMES)))
        ax.set_xticklabels(ELIGIBLE_PROJ_NAMES, rotation=30, ha="right")
        ax.set_ylabel("Transformer Block Index")
        ax.set_title(f"{cid} Final Sparsity Allocation (%) [{d['meta']['selected_candidate']}]")
        fig.colorbar(im, ax=ax, label="Sparsity (%)")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{fname}.png", dpi=200)
        fig.savefig(plots_dir / f"{fname}.pdf")
        plt.close(fig)

    # 4. Q3 Bit-Width Allocation Heatmaps (INT4 and INT3 budgets)
    q_mats = []
    q_titles = []
    for cid in ("q3_mixed_int4", "q3_mixed_int3"):
        cpath = results_dir / "conditions" / f"{cid}.json"
        if not cpath.exists():
            continue
        d = load_json(cpath)
        bmap = d["meta"]["bit_map"]
        mat_b = np.zeros((28, len(ELIGIBLE_PROJ_NAMES)))
        for l in range(28):
            for j, ptype in enumerate(ELIGIBLE_PROJ_NAMES):
                sub = "self_attn" if ptype in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                mat_b[l, j] = bmap[f"model.layers.{l}.{sub}.{ptype}"]
        q_mats.append(mat_b)
        q_titles.append(f"{cid} ({d['meta']['selected_candidate']})")
    if q_mats:
        fig, axes = plt.subplots(1, len(q_mats), figsize=(6.2 * len(q_mats), 5.8))
        if len(q_mats) == 1:
            axes = [axes]
        for ax, mat_b, title in zip(axes, q_mats, q_titles):
            im = ax.imshow(mat_b, aspect="auto", cmap="viridis", vmin=2, vmax=8)
            ax.set_xticks(range(len(ELIGIBLE_PROJ_NAMES)))
            ax.set_xticklabels(ELIGIBLE_PROJ_NAMES, rotation=30, ha="right")
            ax.set_ylabel("Transformer Block Index")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="Assigned Bits")
        fig.tight_layout()
        fig.savefig(plots_dir / "fig4_q3_bit_allocations.png", dpi=200)
        fig.savefig(plots_dir / "fig4_q3_bit_allocations.pdf")
        plt.close(fig)

    # 5. Calibration layer error vs held-out WT2 PPL
    pts_err = [r for r in rows if r["final_block_cum_rel_l2"] is not None]
    if pts_err:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for r in pts_err:
            ax.scatter(r["final_block_cum_rel_l2"], r["wt2_ppl"], s=45)
            ax.annotate(r["condition_id"], (r["final_block_cum_rel_l2"], r["wt2_ppl"]), fontsize=6.5, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("Final-Block Cumulative Relative L2 Error (Calibration)")
        ax.set_ylabel("WikiText-2 Perplexity")
        ax.set_title("Calibration Reconstruction Error vs Held-Out Perplexity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "fig5_cal_error_vs_heldout.png", dpi=200)
        fig.savefig(plots_dir / "fig5_cal_error_vs_heldout.pdf")
        plt.close(fig)


def extract_qualitative_cases(results_dir: Path, dataset_root: Path) -> list[dict[str, Any]]:
    pe_dir = results_dir / "per_example"
    key_cids = [
        "fp16_baseline",
        "p1_mag_s50",
        "p2_wanda_s50",
        "p3_nonuniform_s50",
        "q1_int4",
        "q2_awq_int4",
        "q3_mixed_int4",
        "q2_awq_int3",
        "q3_mixed_int3",
    ]
    loaded = {}
    for cid in key_cids:
        p = pe_dir / f"{cid}_hellaswag.json.gz"
        if p.exists():
            loaded[cid] = load_json_gz(p)
    if "fp16_baseline" not in loaded:
        return []

    hs_raw = load_hellaswag_examples(dataset_root / "data" / "public_locked" / "hellaswag_val_locked.jsonl.gz", limit=512)
    base = loaded["fp16_baseline"]

    # Pick informative matched cases deterministically (from first 512 diagnostic IDs):
    # (1) Baseline right, P1@50 wrong, P2/P3 right; (2) Baseline right, Q2@INT3 wrong, Q3@INT3 right;
    # (3) Baseline right, both compressed fail; (4) Baseline wrong, Q2/P3 flips right.
    chosen: list[tuple[str, int]] = []
    for idx in range(512):
        b_ok = base[idx]["correct_norm"]
        p1_ok = loaded.get("p1_mag_s50", base)[idx]["correct_norm"]
        p2_ok = loaded.get("p2_wanda_s50", base)[idx]["correct_norm"]
        p3_ok = loaded.get("p3_nonuniform_s50", base)[idx]["correct_norm"]
        q2_3_ok = loaded.get("q2_awq_int3", base)[idx]["correct_norm"]
        q3_3_ok = loaded.get("q3_mixed_int3", base)[idx]["correct_norm"]

        tags = {t for t, _ in chosen}
        if "prune_recovery" not in tags and b_ok == 1 and p1_ok == 0 and p3_ok == 1:
            chosen.append(("prune_recovery", idx))
        elif "quant_recovery" not in tags and b_ok == 1 and q2_3_ok == 0 and q3_3_ok == 1:
            chosen.append(("quant_recovery", idx))
        elif "shared_degradation" not in tags and b_ok == 1 and p2_ok == 0 and q3_3_ok == 0:
            chosen.append(("shared_degradation", idx))
        elif "unexpected_flip" not in tags and b_ok == 0 and (p3_ok == 1 or q3_3_ok == 1):
            chosen.append(("unexpected_flip", idx))
        if len(chosen) >= 4:
            break

    cases = []
    for reason, idx in chosen:
        ex = hs_raw[idx]
        per_model = {}
        for cid, arr in loaded.items():
            rec = arr[idx]
            per_model[cid] = {
                "pred_raw": rec["pred_raw"],
                "pred_norm": rec["pred_norm"],
                "correct_raw": rec["correct_raw"],
                "correct_norm": rec["correct_norm"],
                "raw_lls": rec["raw_lls"],
                "norm_lls": rec["norm_lls"],
            }
        cases.append({
            "selection_criterion": reason,
            "order": ex["order"],
            "source_row_index": ex["source_row_index"],
            "activity_label": ex["activity_label"],
            "context": ex["context"],
            "endings": ex["endings"],
            "gold": ex["gold"],
            "models": per_model,
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "task1")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()

    rows = build_summary_rows(args.results_dir)
    csv_path = args.results_dir / "summary_task1.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    boot = compute_all_bootstraps(args.results_dir, n_resamples=args.n_resamples)
    (args.results_dir / "bootstrap_ci_task1.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    plot_task1_figures(args.results_dir, rows)
    cases = extract_qualitative_cases(args.results_dir, args.dataset_root)
    (args.results_dir / "qualitative_hellaswag_cases.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )
    print(f"Wrote {csv_path}, bootstrap CIs, plots/, and {len(cases)} qualitative cases.")


if __name__ == "__main__":
    main()

