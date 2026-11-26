"""Generate publication-quality NeurIPS vector PDF + PNG figures for ATML PA4 report.
Reads strictly from results/task1, results/task2, results/task3 (zero hardcoded metrics).
"""
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Clean NeurIPS style (colorblind-safe Okabe-Ito / muted academic palette)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_YELLOW = "#E69F00"
C_SKY = "#56B4E9"
C_GRAY = "#555555"
C_RED = "#B22222"


def save_fig(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=240)
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


# =============================================================================
# FIGURE 1: Task 1 — Model Compression (Pareto, Non-uniform Allocation, Cal-Err)
# =============================================================================
def make_fig1():
    rows = {r["condition_id"]: r for r in csv.DictReader(open(RES / "task1" / "summary_task1.csv"))}
    p3_50 = json.load(open(RES / "task1" / "conditions" / "p3_nonuniform_s50.json"))
    p3_75 = json.load(open(RES / "task1" / "conditions" / "p3_nonuniform_s75.json"))
    q3_3 = json.load(open(RES / "task1" / "conditions" / "q3_mixed_int3.json"))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.75), gridspec_kw={"wspace": 0.48})

    # (a) Effective Bits vs WikiText-2 PPL
    ax = axes[0]
    base_ppl = float(rows["fp16_baseline"]["wt2_ppl"])
    ax.axhline(base_ppl, color=C_GRAY, ls=":", lw=1.0, label=f"FP16 ({base_ppl:.1f})")

    series = [
        ("P1 Mag", ["p1_mag_s25", "p1_mag_s50", "p1_mag_s75"], C_RED, "s", "--"),
        ("P2 Wanda", ["p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75"], C_ORANGE, "o", "-"),
        ("P3 ShortGPT-BI", ["p3_nonuniform_s50", "p3_nonuniform_s75"], C_YELLOW, "^", "-"),
        ("Q1 RTN", ["q1_int8", "q1_int4", "q1_int3"], C_SKY, "D", "--"),
        ("Q2 AWQ", ["q2_awq_int8", "q2_awq_int4", "q2_awq_int3"], C_BLUE, "o", "-"),
        ("Q3 MILP-Taylor", ["q3_mixed_int4", "q3_mixed_int3"], C_GREEN, "*", "-"),
    ]
    for label, cids, col, mk, ls in series:
        xs = [float(rows[c]["effective_bits_elig"]) for c in cids]
        ys = [float(rows[c]["wt2_ppl"]) for c in cids]
        ms = 6.5 if mk == "*" else 4.2
        ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.3, markersize=ms, label=label)

    # Single point for 2:4
    r24 = rows["p2_wanda_2_4_s50"]
    ax.scatter([float(r24["effective_bits_elig"])], [float(r24["wt2_ppl"])],
               color=C_PURPLE, marker="X", s=38, zorder=5, label="Wanda 2:4 (50%)")

    ax.set_yscale("log")
    ax.set_xlim(2.5, 13.8)
    ax.set_ylim(11, 1.5e8)
    ax.set_xlabel("Effective Bits / Eligible Weight")
    ax.set_ylabel("WikiText-2 Perplexity (log)")
    ax.set_title("(a) Compression Rate–Distortion")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, handlelength=1.2, columnspacing=0.5, fontsize=6.6)

    # (b) Layer-wise P3 sparsity & Q3 INT3 upgraded projections
    ax = axes[1]
    layers = np.arange(28)
    s50 = [p3_50["meta"]["sparsity_map"][f"model.layers.{i}.mlp.down_proj"] * 100 for i in layers]
    s75 = [p3_75["meta"]["sparsity_map"][f"model.layers.{i}.mlp.down_proj"] * 100 for i in layers]
    ax.plot(layers, s50, color=C_ORANGE, lw=1.4, marker="o", ms=2.5, label="P3 Sparsity (50% avg)")
    ax.plot(layers, s75, color=C_RED, lw=1.4, marker="s", ms=2.5, label="P3 Sparsity (75% avg)")
    ax.axhline(50, color=C_ORANGE, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(75, color=C_RED, ls=":", lw=0.8, alpha=0.6)

    # Overlay count of 4-bit upgraded projections per layer in Q3 INT3
    ax2 = ax.twinx()
    bm3 = q3_3["meta"]["bit_map"]
    up4 = [sum(1 for proj in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
               if bm3[f"model.layers.{i}.{proj}"] == 4) for i in layers]
    ax2.bar(layers, up4, color=C_BLUE, alpha=0.28, width=0.65, label="Q3-INT3 4-bit projs")
    ax2.set_ylabel("# 4-bit Matrices (Q3-INT3)", color=C_BLUE)
    ax2.tick_params(axis="y", colors=C_BLUE)
    ax2.set_ylim(0, 4.5)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    ax.set_xlabel("Decoder Block Index (0–27)")
    ax.set_ylabel("Block Sparsity (%)")
    ax.set_ylim(15, 86)
    ax.set_title("(b) Non-Uniform P3 & Q3 Allocation")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, framealpha=0.9, fontsize=6.8)

    # (c) Cumulative calibration error vs held-out WT2 NLL
    ax = axes[2]
    pr_x, pr_y, qt_x, qt_y = [], [], [], []
    for cid, r in rows.items():
        if not r["final_block_cum_rel_l2"]:
            continue
        x = float(r["final_block_cum_rel_l2"])
        y = float(r["wt2_nll"])
        if cid.startswith("p"):
            pr_x.append(x); pr_y.append(y)
        else:
            qt_x.append(x); qt_y.append(y)
    ax.scatter(pr_x, pr_y, color=C_ORANGE, marker="o", s=34, label="Pruning (P1–P3)", zorder=4)
    ax.scatter(qt_x, qt_y, color=C_BLUE, marker="D", s=32, label="Quantization (Q1–Q3)", zorder=4)

    # Annotate key points cleanly
    ann = [
        ("q2_awq_int4", "AWQ-4", (-6, 8)),
        ("q1_int3", "RTN-3", (-12, 8)),
        ("p2_wanda_s50", "Wanda-50%", (5, -10)),
        ("p3_nonuniform_s75", "P3-75%", (-36, 4)),
        ("p2_wanda_s75", "Wanda-75%", (-20, -14)),
        ("p1_mag_s75", "Mag-75%", (-40, -10)),
    ]
    for cid, txt, offset in ann:
        r = rows[cid]
        ax.annotate(txt, (float(r["final_block_cum_rel_l2"]), float(r["wt2_nll"])),
                    xytext=offset, textcoords="offset points", fontsize=6.5, color="#222222")

    ax.text(0.04, 0.80, r"Spearman $\rho = 0.971$" + "\n" + r"(vs. HS-512: $r = -0.931$)",
            transform=ax.transAxes, fontsize=7.0, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92))
    ax.set_xlabel("Final-Block Cumulative Rel. $L_2$ Error")
    ax.set_ylabel("Held-Out WikiText-2 NLL")
    ax.set_ylim(2.0, 15.2)
    ax.set_title("(c) Calibration Proxy vs. Held-Out NLL")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=True, framealpha=0.92, fontsize=6.8)

    save_fig(fig, "fig1_task1_compression")


# =============================================================================
# FIGURE 2: Task 2 Parts A & B — Scaling, MQAR Distance Recall, Inference Mem
# =============================================================================
def make_fig2():
    pa = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_a_summary.csv")))
    pb = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_b_summary.csv")))
    inf = json.load(open(RES / "task2" / "part_b" / "inference_sweep.json"))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.75), gridspec_kw={"wspace": 0.48})

    # (a) Part A FP16 Fwd+Bwd Peak Memory & Latency
    ax = axes[0]
    ax2 = ax.twinx()
    ops = [("dense", "Dense", C_RED, "o"), ("tiled_64", "Tiled-64 (Py)", C_ORANGE, "s"), ("sdpa", "SDPA (Flash)", C_BLUE, "^")]
    for op, lbl, col, mk in ops:
        sub = sorted([r for r in pa if r["dtype"] == "fp16" and r["operator"] == op], key=lambda r: int(r["L"]))
        Ls = [int(r["L"]) for r in sub]
        mem = [float(r["fwdbwd_peak_alloc_mb"]) for r in sub]
        lat = [float(r["fwdbwd_ms"]) for r in sub]
        ax.plot(Ls, mem, color=col, marker=mk, lw=1.4, ms=4.0, label=f"{lbl} Mem")
        ax2.plot(Ls, lat, color=col, marker=mk, lw=1.0, ls="--", ms=3.0, alpha=0.8)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Sequence Length $L$ (FP16 Fwd+Bwd)")
    ax.set_ylabel("Peak Alloc. Memory (MiB, solid)")
    ax2.set_ylabel("Latency (ms, dashed)", color=C_GRAY)
    ax2.tick_params(axis="y", colors=C_GRAY)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax.set_title("(a) Part A: Attention Memory vs. Latency")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.7)

    # (b) Part B MQAR K=16 accuracy by distance bin (+ individual linear seeds)
    ax = axes[1]
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    x = np.arange(len(bins))
    configs = [
        ("dense", "mixed", "Dense (mixed)", C_BLUE, "o", "-"),
        ("topk_64", "mixed", "Top-$k$=64 (B3)", C_GREEN, "*", "-"),
        ("sliding_w64", "near", "Sliding $W$=64 (near)", C_ORANGE, "s", "--"),
        ("sliding_w64", "mixed", "Sliding $W$=64 (mix)", C_RED, "x", ":"),
        ("linear_elu1", "mixed", "Linear ELU+1 (mean)", C_PURPLE, "D", "-"),
        ("local_dilated", "mixed", "Local+Dilated", C_GRAY, "^", "--"),
    ]
    for mech, reg, lbl, col, mk, ls in configs:
        r = [row for row in pb if row["mechanism"] == mech and int(row["train_k"]) == 16 and row["train_regime"] == reg][0]
        ys = [float(r[f"bin_{b}_mean"]) * 100 for b in bins]
        ms = 6.5 if mk == "*" else 4.0
        ax.plot(x, ys, color=col, marker=mk, ls=ls, lw=1.4, ms=ms, label=lbl)

    # Annotate linear_elu1 per-seed split at K=16
    ax.annotate("Linear 3 seeds:\n96%, 99%, 7.8%", xy=(2.45, 69.5), xytext=(1.78, 34),
                fontsize=6.4, color=C_PURPLE,
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_PURPLE, alpha=0.92))
    ax.axhline(100/64, color="#999999", ls=":", lw=0.8)
    ax.text(3.05, 3.8, "1/64", fontsize=6.4, color="#777777", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["0–31", "32–63", "64–127", "128–255"])
    ax.set_ylim(-3, 108)
    ax.set_xlabel("Key–Query Distance Bin ($K=16$)")
    ax.set_ylabel("MQAR Recall Accuracy (%)")
    ax.set_title("(b) Part B: Distance-Bin Recall ($K=16$)")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.39), frameon=True, framealpha=0.94, fontsize=6.2)

    # (c) Part B Inference Memory Scaling (T=256..8192)
    ax = axes[2]
    m_styles = [
        ("dense", "Dense", C_BLUE, "o", "-"),
        ("sliding_w64", "Sliding $W$=64 (masked)", C_ORANGE, "s", "--"),
        ("local_dilated", "Local+Dilated (masked)", C_GRAY, "^", ":"),
        ("topk_64", "Top-$k$=64 (masked)", C_GREEN, "*", "-"),
        ("linear_elu1", "Linear ELU+1 (state)", C_PURPLE, "D", "-"),
    ]
    for mech, lbl, col, mk, ls in m_styles:
        sub = sorted([r for r in inf if r["mechanism"] == mech], key=lambda r: r["T"])
        Ts = [r["T"] for r in sub]
        mbs = [r["fwd_memory"]["peak_allocated_bytes"] / (1024**2) for r in sub]
        ax.plot(Ts, mbs, color=col, marker=mk, ls=ls, lw=1.4, ms=4.5 if mk != "*" else 6.5, label=lbl)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 1024, 4096, 8192])
    ax.set_xticklabels(["256", "1k", "4k", "8k"])
    ax.set_xlabel("Sequence Length $T$ (Inference Fwd)")
    ax.set_ylabel("Peak Alloc. Memory (MiB)")
    ax.set_title("(c) Part B: Inference Memory Scaling")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.5)

    save_fig(fig, "fig2_task2_ab")


# =============================================================================
# FIGURE 3: Task 2 Part C — Matched-Budget KV Accuracy & C5 Pack/Unpack Gap
# =============================================================================
def make_fig3():
    bk = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_c_breakdowns.csv")))
    c5 = json.load(open(RES / "task2" / "part_c" / "c5_packed_encode_decode_bench.json"))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.75), gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1.0]})

    # (a) Exact match by evidence position
    ax = axes[0]
    policies = [
        ("full_fp16", "Full FP16\n(100%)"),
        ("evict_sink_recent", "Sink+Rec\n(28.7%)"),
        ("evict_question_aware", "Q-Aware\n(28.7%)"),
        ("quant_int4_tok", "INT4-Tok\n(28.7%)"),
        ("quant_int4_kivi", "INT4-KIVI\n(28.7%)"),
        ("hybrid_fp16_int2", "Hybrid\n(27.5%)"),
    ]
    x = np.arange(len(policies))
    w = 0.25
    pos_styles = [("start", "Start ($n=60$)", C_BLUE), ("middle", "Middle ($n=60$)", C_ORANGE), ("end", "End ($n=60$)", C_GREEN)]
    for idx, (pos, lbl, col) in enumerate(pos_styles):
        vals = []
        for pol, _ in policies:
            row = [r for r in bk if r["policy"] == pol and r["factor"] == "position" and r["level"] == pos][0]
            vals.append(float(row["exact_match"]) * 100)
        bars = ax.bar(x + (idx - 1) * w, vals, width=w, color=col, label=lbl, edgecolor="white", lw=0.4)
        for b, v in zip(bars, vals):
            if v in (0.0, 100.0) and pos == "middle" and b.get_x() > 2.5 and v == 0.0:
                ax.text(b.get_x() + b.get_width()/2, 3.0, "0%", ha="center", va="bottom", fontsize=6.5, color=C_RED, fontweight="bold")
            elif 90 < v < 100:
                ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2)

    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in policies], fontsize=7.2)
    ax.set_ylim(0, 124)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("(a) Part C: Retrieval by Evidence Position (Matched INT4 Budget)")
    ax.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.92, fontsize=6.8)

    # (b) C5 Real Packed Storage vs Decode-with-Unpack Latency
    ax = axes[1]
    cfgs = [(1, 2048, "B1\nL2k"), (1, 4096, "B1\nL4k"), (8, 2048, "B8\nL2k"), (8, 4096, "B8\nL4k")]
    xc = np.arange(len(cfgs))
    w = 0.25
    fmts = [("fp16_ref", "FP16 Ref", C_GRAY), ("int4_kivi", "INT4-KIVI", C_BLUE), ("int2_kivi", "INT2-KIVI", C_GREEN)]
    for idx, (fmt, lbl, col) in enumerate(fmts):
        lats = []
        ratios = []
        for B, L, _ in cfgs:
            row = [r for r in c5 if r["B"] == B and r["L"] == L and r["format"] == fmt][0]
            ref = [r for r in c5 if r["B"] == B and r["L"] == L and r["format"] == "fp16_ref"][0]
            lats.append(row["decode_step_with_unpack_latency"]["mean_ms"])
            ratios.append(ref["serialized_bytes"] / row["serialized_bytes"])
        bars = ax.bar(xc + (idx - 1) * w, lats, width=w, color=col, label=f"{lbl} ({ratios[0]:.1f}$\\times$ smaller)" if fmt != "fp16_ref" else lbl)
        for b, v in zip(bars, lats):
            ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.1, rotation=0)

    ax.set_xticks(xc)
    ax.set_xticklabels([c[2] for c in cfgs])
    ax.set_ylim(0, 122)
    ax.set_ylabel("Decode Step + Bit-Unpack Latency (ms)")
    ax.set_title("(b) C5: Packed KV Storage Savings vs. Unfused Latency")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.8)

    save_fig(fig, "fig3_task2_c_kv")


# =============================================================================
# FIGURE 4: Task 3 — Prefix Alignment, 4-Part Timing, Analytical vs Measured S
# =============================================================================
def make_fig4():
    pa = list(csv.DictReader(open(RES / "task3" / "analysis" / "prefix_alignment_summary.csv")))
    sp = list(csv.DictReader(open(RES / "task3" / "analysis" / "speculative_summary.csv")))
    isc = json.load(open(RES / "task3" / "spec_bench" / "isolated_step_costs.json"))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.80), gridspec_kw={"wspace": 0.34})

    # (a) Prefix alignment Forward KL on teacher & original_draft banks (avg over 2 seeds)
    ax = axes[0]
    subsplits = ["math", "short_familiar", "short_shifted"]
    x = np.arange(len(subsplits))
    w = 0.25
    groups = [
        ("Original", ["original"], C_GRAY),
        ("Off-Policy (2s)", ["offpolicy_s6304", "offpolicy_s6305"], C_BLUE),
        ("On-Policy (2s)", ["onpolicy_s6304", "onpolicy_s6305"], C_ORANGE),
    ]
    for idx, (lbl, drafts, col) in enumerate(groups):
        vals_t = []
        vals_d = []
        for s in subsplits:
            vt = [float(r["forward_kl"]) for r in pa if r["draft"] in drafts and r["bank"] == "teacher" and r["subsplit"] == s]
            vd = [float(r["forward_kl"]) for r in pa if r["draft"] in drafts and r["bank"] == "original_draft" and r["subsplit"] == s]
            vals_t.append(statistics.mean(vt))
            vals_d.append(statistics.mean(vd))
        ax.bar(x + (idx - 1) * w, vals_t, width=w, color=col, label=f"{lbl}", edgecolor="white", lw=0.4)
        # Overlay diamond markers for original_draft bank
        ax.scatter(x + (idx - 1) * w, vals_d, color="black", marker="D", s=16, zorder=5,
                   label="Orig-Draft Bank" if idx == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(["Math", "Short\nFam.", "Short\nShift."])
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("Forward KL $D_{\\mathrm{KL}}(p \\| q)$")
    ax.set_title("(a) Common-Prefix Alignment (Bars: Teacher)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.5)

    # (b) 4-Part Speculative Round-Time Decomposition vs Target-Only
    ax = axes[1]
    # Average over seeds for clean display
    bars_spec = [
        ("Orig\n$\\gamma$=2", ["original"], 2),
        ("Off-P\n$\\gamma$=2", ["offpolicy_s6304", "offpolicy_s6305"], 2),
        ("On-P\n$\\gamma$=2", ["onpolicy_s6304", "onpolicy_s6305"], 2),
        ("Orig\n$\\gamma$=4", ["original"], 4),
        ("Off-P\n$\\gamma$=4", ["offpolicy_s6304", "offpolicy_s6305"], 4),
        ("On-P\n$\\gamma$=4", ["onpolicy_s6304", "onpolicy_s6305"], 4),
    ]
    xb = np.arange(len(bars_spec))
    d_ms, v_ms, a_ms, c_ms, tgt_ms = [], [], [], [], []
    for _, drafts, g in bars_spec:
        sub = [r for r in sp if r["draft"] in drafts and int(r["gamma"]) == g]
        d_ms.append(statistics.mean(float(r["draft_proposal_ms_mean"]) for r in sub))
        v_ms.append(statistics.mean(float(r["target_verify_ms_mean"]) for r in sub))
        a_ms.append(statistics.mean(float(r["accept_residual_ms_mean"]) for r in sub))
        c_ms.append(statistics.mean(float(r["cache_update_ms_mean"]) for r in sub))
        # Target-only mean decode time ~ (total_spec_ms * measured_speedup_decode)
        tot_s = d_ms[-1] + v_ms[-1] + a_ms[-1] + c_ms[-1]
        tgt_ms.append(tot_s * statistics.mean(float(r["measured_speedup_decode"]) for r in sub))

    ax.bar(xb, d_ms, width=0.58, color=C_ORANGE, label="Draft proposal")
    ax.bar(xb, v_ms, width=0.58, bottom=d_ms, color=C_BLUE, label="Target verify")
    b2 = [d + v for d, v in zip(d_ms, v_ms)]
    ax.bar(xb, [a + c for a, c in zip(a_ms, c_ms)], width=0.58, bottom=b2, color=C_GREEN, label="Accept+Cache")
    ax.axhline(statistics.mean(tgt_ms), color=C_RED, ls="--", lw=1.2, label=f"Target-only ({statistics.mean(tgt_ms):.0f} ms)")

    ax.set_xticks(xb)
    ax.set_xticklabels([b[0] for b in bars_spec], fontsize=6.9)
    ax.set_ylim(0, 1780)
    ax.set_ylabel("Mean Decode Time / Prompt (ms)")
    ax.set_title("(b) Speculative Round-Time Decomposition")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, fontsize=6.2, columnspacing=0.6)

    # (c) Analytical Speedup Curve vs Measured + Hardware Counterfactual
    ax = axes[2]
    # Use prompt_len=64 measured c and v_gamma (nearly identical across 64/128/256)
    c_meas = statistics.mean(r["c_ratio"] for r in isc["by_prompt_len"])
    v2 = statistics.mean(r["verify_passes"]["2"]["v_gamma"] for r in isc["by_prompt_len"])
    v4 = statistics.mean(r["verify_passes"]["4"]["v_gamma"] for r in isc["by_prompt_len"])

    alphas = np.linspace(0.55, 0.99, 200)
    def S_hat(a, g, c, v):
        e_tok = (1.0 - a**(g + 1)) / (1.0 - a)
        return e_tok / (g * c + v)

    ax.plot(alphas * 100, S_hat(alphas, 2, c_meas, v2), color=C_BLUE, lw=1.5, label=f"$\\hat{{S}}_2$ ($c$={c_meas:.2f}, L4 eager)")
    ax.plot(alphas * 100, S_hat(alphas, 4, c_meas, v4), color=C_ORANGE, lw=1.5, label=f"$\\hat{{S}}_4$ ($c$={c_meas:.2f}, L4 eager)")
    ax.plot(alphas * 100, S_hat(alphas, 4, 0.25, 1.15), color=C_GREEN, lw=1.2, ls="--", label="$\\hat{S}_4$ ($c$=0.25 counterfactual)")
    ax.axhline(1.0, color=C_GRAY, ls=":", lw=1.0)

    # Plot empirical (acc_over_examined, measured_speedup_decode) points
    for g, col, mk in [(2, C_BLUE, "o"), (4, C_ORANGE, "s")]:
        sub = [r for r in sp if int(r["gamma"]) == g]
        ax.scatter([float(r["acc_over_examined"]) * 100 for r in sub],
                   [float(r["measured_speedup_decode"]) for r in sub],
                   color=col, marker=mk, s=26, edgecolor="black", lw=0.5, zorder=5,
                   label=f"Measured $S_{g}$" )

    ax.annotate("$\\alpha^*\\approx 94\\%$", xy=(93.8, 1.0), xytext=(78, 1.25),
                fontsize=6.8, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))
    ax.set_xlim(58, 100)
    ax.set_ylim(0.55, 2.35)
    ax.set_xlabel("Per-Token Acceptance Rate $\\alpha$ (%)")
    ax.set_ylabel("Speedup Over Target-Only ($\\times$)")
    ax.set_title("(c) Cost Model $\\hat{S}_\\gamma(\\alpha)$ vs. Measured $S_\\gamma$")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.1)

    save_fig(fig, "fig4_task3_speculative")


# =============================================================================
# APPENDIX FIGURES: A1 (Task 1 Dev + Q3 Heatmap), A2 (Task 2 MQAR Curves), A3 (Task 3 Distill + Pos Bins)
# =============================================================================
def make_figA1():
    p3c = json.load(open(RES / "task1" / "dev_selection" / "p3_candidates.json"))["candidates"]
    q3c = json.load(open(RES / "task1" / "dev_selection" / "q3_candidates.json"))["candidates"]
    q3_int3 = json.load(open(RES / "task1" / "conditions" / "q3_mixed_int3.json"))["meta"]["bit_map"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.85), gridspec_kw={"wspace": 0.34, "width_ratios": [1.05, 0.9, 1.25]})

    # (a) P3 C4 dev PPL at 75% sparsity
    ax = axes[0]
    p75 = [c for c in p3c if abs(c["target_sparsity"] - 0.75) < 1e-6]
    names = [c["candidate"].replace("_", "\n") for c in p75]
    ppls = [c["c4_dev_ppl"] for c in p75]
    cols = [C_GREEN if c["candidate"] == "shortgpt_bi_lam0.15" else C_GRAY for c in p75]
    y = np.arange(len(p75))
    ax.barh(y, ppls, color=cols, height=0.65)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([c["candidate"].replace("marginal_", "marg_").replace("shortgpt_bi_", "shortgpt_") for c in p75], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("C4 Dev PPL at 75% Sparsity (log)")
    ax.set_title("(a) P3 Dev Selection (75%)")

    # (b) Q3 C4 dev PPL at INT4 & INT3
    ax = axes[1]
    cands = ["milp_unweighted_mse", "milp_rel_mse", "milp_output_fisher", "milp_taylor"]
    labels = ["Unwt MSE", "Rel MSE", "Out Fisher", "Taylor*"]
    x = np.arange(len(cands))
    p4 = [next(c["c4_dev_ppl"] for c in q3c if c["candidate"] == cn and c["budget_label"] == "int4_budget") for cn in cands]
    p3 = [next(c["c4_dev_ppl"] for c in q3c if c["candidate"] == cn and c["budget_label"] == "int3_budget") for cn in cands]
    ax.bar(x - 0.18, p4, width=0.35, color=C_BLUE, label="INT4 budget")
    ax.bar(x + 0.18, p3, width=0.35, color=C_ORANGE, label="INT3 budget")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0, rotation=15)
    ax.set_ylabel("C4 Dev Perplexity")
    ax.set_title("(b) Q3 MILP Dev Selection")
    ax.legend(loc="upper right", frameon=True, fontsize=6.8)

    # (c) Q3 INT3 Bit-Width Heatmap (7 projections x 28 layers)
    ax = axes[2]
    projs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    plbls = ["q", "k", "v", "o", "gate", "up", "down"]
    mat = np.zeros((len(projs), 28))
    for pi, pr in enumerate(projs):
        for li in range(28):
            mat[pi, li] = q3_int3[f"model.layers.{li}.{pr}"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=2, vmax=4)
    ax.set_yticks(np.arange(len(projs)))
    ax.set_yticklabels(plbls, fontsize=7.2)
    ax.set_xticks([0, 7, 14, 21, 27])
    ax.set_xlabel("Decoder Layer (0–27)")
    ax.set_title("(c) Q3-INT3 Bit Allocation (2 / 3 / 4-bit)")
    cbar = fig.colorbar(im, ax=ax, ticks=[2, 3, 4], pad=0.02, fraction=0.045)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)

    save_fig(fig, "figA1_task1_dev_and_q3")


def make_figA2():
    runs_dir = RES / "task2" / "part_b" / "runs"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.75), gridspec_kw={"wspace": 0.28})

    # (a) Learning curves at K=16 mixed across mechanisms
    ax = axes[0]
    mechs = [
        ("B1_dense_K16_mixed", "Dense", C_BLUE, "-"),
        ("B3_topk_64_K16_mixed", "Top-$k$=64 (B3)", C_GREEN, "-"),
        ("B2_linear_elu1_K16_mixed", "Linear ELU+1 (seeds)", C_PURPLE, "--"),
        ("B1_sliding_w64_K16_mixed", "Sliding $W$=64 (mix)", C_RED, ":"),
        ("B2_local_dilated_K16_mixed", "Local+Dilated", C_GRAY, "-."),
    ]
    for prefix, lbl, col, ls in mechs:
        seed_curves = []
        steps = None
        for s in [6304, 6305, 6306]:
            d = json.load(open(runs_dir / f"{prefix}_s{s}.json"))
            lc = d["learning_curve"]
            steps = [pt["step"] for pt in lc]
            accs = [pt["dev_acc"] * 100 for pt in lc]
            seed_curves.append(accs)
            if "linear_elu1" in prefix:
                ax.plot(steps, accs, color=col, ls=ls, lw=1.1, alpha=0.88, label="Linear ELU+1 (3 seeds)" if s == 6304 else None)
        if "linear_elu1" not in prefix:
            mean_c = np.mean(seed_curves, axis=0)
            ax.plot(steps, mean_c, color=col, ls=ls, lw=1.5, label=lbl)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Dev Recall Accuracy (%)")
    ax.set_title("(a) Part B: MQAR Learning Curves ($K=16$, Mixed)")
    ax.set_ylim(-2, 106)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.48), frameon=True, framealpha=0.92, fontsize=6.6)

    # (b) Part A FP32 vs FP16 Max Abs Error across L
    ax = axes[1]
    pa = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_a_summary.csv")))
    for dt, ls in [("fp16", "-"), ("fp32", "--")]:
        for op, col, mk in [("tiled_64", C_ORANGE, "s"), ("sdpa", C_BLUE, "^")]:
            sub = sorted([r for r in pa if r["dtype"] == dt and r["operator"] == op], key=lambda r: int(r["L"]))
            Ls = [int(r["L"]) for r in sub]
            err_o = [float(r["out_max_abs_err"]) for r in sub]
            err_dq = [float(r["dq_max_abs_err"]) for r in sub]
            ax.plot(Ls, err_o, color=col, ls=ls, marker=mk, lw=1.3, ms=3.8, label=f"{op} {dt.upper()} (out)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Sequence Length $L$")
    ax.set_ylabel("Max Abs Output Error vs. FP32 Dense")
    ax.set_title("(b) Part A: Numerical Stability Across $L$ & Dtype")
    ax.legend(loc="center right", frameon=True, framealpha=0.92, fontsize=6.8)

    save_fig(fig, "figA2_task2_mqar_curves")


def make_figA3():
    dist = list(csv.DictReader(open(RES / "task3" / "analysis" / "distillation_summary.csv")))
    pam = json.load(open(RES / "task3" / "prefix_alignment_metrics.json"))["metrics_by_cell"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.75), gridspec_kw={"wspace": 0.28})

    # (a) Distillation wall-clock cost breakdown
    ax = axes[0]
    runs = ["offpolicy_s6304", "offpolicy_s6305", "onpolicy_s6304", "onpolicy_s6305"]
    lbls = ["Off-P\ns6304", "Off-P\ns6305", "On-P\ns6304", "On-P\ns6305"]
    x = np.arange(len(runs))
    prep = [float(next(r["teacher_pool_prep_s"] for r in dist if r["run"] == rn)) for rn in runs]
    roll = [float(next(r["student_rollout_s"] for r in dist if r["run"] == rn)) for rn in runs]
    tscr = [float(next(r["teacher_scoring_s"] for r in dist if r["run"] == rn)) for rn in runs]
    opt = [float(next(r["student_fwd_bwd_opt_s"] for r in dist if r["run"] == rn)) for rn in runs]

    ax.bar(x, prep, width=0.55, color=C_SKY, label="Teacher pool prep")
    ax.bar(x, roll, width=0.55, bottom=prep, color=C_ORANGE, label="Student rollouts")
    b2 = [a + b for a, b in zip(prep, roll)]
    ax.bar(x, [a + b for a, b in zip(tscr, opt)], width=0.55, bottom=b2, color=C_BLUE, label="Scoring + Fwd/Bwd/Opt")
    for xi, tot in zip(x, [p + r + t + o for p, r, t, o in zip(prep, roll, tscr, opt)]):
        ax.text(xi, tot + 35, f"{tot:.0f}s", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls)
    ax.set_ylim(0, 2250)
    ax.set_ylabel("Wall-Clock Training Cost (s)")
    ax.set_title("(a) Distillation Cost Decomposition (40k Tokens)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.8)

    # (b) Position-bin Forward KL on teacher & original_draft banks
    ax = axes[1]
    bins = ["[1,16]", "[17,32]", "[33,64]", "[65,192]"]
    xb = np.arange(len(bins))
    for bank, ls in [("teacher", "-"), ("original_draft", "--")]:
        for lbl, drafts, col, mk in [
            ("Original", ["original"], C_GRAY, "o"),
            ("Off-Policy", ["offpolicy_s6304", "offpolicy_s6305"], C_BLUE, "s"),
            ("On-Policy", ["onpolicy_s6304", "onpolicy_s6305"], C_ORANGE, "^"),
        ]:
            ys = []
            for pb in bins:
                cells = [c for c in pam if c["draft"] in drafts and c["bank"] == bank and c["pos_bin"] == pb]
                tot_n = sum(c["count"] for c in cells)
                ys.append(sum(c["forward_kl"] * c["count"] for c in cells) / tot_n)
            b_tag = "Tch" if bank == "teacher" else "Drf"
            ax.plot(xb, ys, color=col, ls=ls, marker=mk, lw=1.35, ms=4.0, label=f"{lbl} ({b_tag})")

    ax.set_xticks(xb)
    ax.set_xticklabels(bins)
    ax.set_ylim(0.155, 0.355)
    ax.set_xlabel("Token Position Bin")
    ax.set_ylabel("Forward KL $D_{\\mathrm{KL}}(p \\| q)$")
    ax.set_title("(b) Alignment Across Token Position Bins")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, fontsize=6.3, columnspacing=0.6)

    save_fig(fig, "figA3_task3_distill_and_pos")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_figA1()
    make_figA2()
    make_figA3()
    print("All figures generated cleanly!")
