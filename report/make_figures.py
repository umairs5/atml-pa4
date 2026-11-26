"""Generate clean, simple matplotlib figures for ATML PA4 (matching PA1/PA2/PA3 style).
Reads strictly from results/task1, results/task2, results/task3.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Clean default matplotlib look matching PA1/PA2/PA3
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"Saved {name}.pdf/.png")


# =============================================================================
# MAIN FIGURE 1 (Task 1): 2 simple panels — Packed MB vs PPL + P3 layer sparsity
# =============================================================================
def make_fig1():
    rows = {r["condition_id"]: r for r in csv.DictReader(open(RES / "task1" / "summary_task1.csv"))}
    p3_50 = json.load(open(RES / "task1" / "conditions" / "p3_nonuniform_s50.json"))
    p3_75 = json.load(open(RES / "task1" / "conditions" / "p3_nonuniform_s75.json"))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.75), gridspec_kw={"wspace": 0.28})

    # Left: Packed size (MiB) vs WikiText-2 PPL (log)
    ax = axes[0]
    base_ppl = float(rows["fp16_baseline"]["wt2_ppl"])
    ax.axhline(base_ppl, color="black", ls="--", lw=0.9, label=f"FP16 ({base_ppl:.1f})")

    curves = [
        ("P1 Magnitude", ["p1_mag_s25", "p1_mag_s50", "p1_mag_s75"], "C3", "s", "--"),
        ("P2 Wanda", ["p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75"], "C1", "o", "-"),
        ("P3 ShortGPT-BI", ["p3_nonuniform_s50", "p3_nonuniform_s75"], "C2", "^", "-"),
        ("Q1 RTN", ["q1_int8", "q1_int4", "q1_int3"], "C0", "D", "--"),
        ("Q2 AWQ", ["q2_awq_int8", "q2_awq_int4", "q2_awq_int3"], "C0", "o", "-"),
        ("Q3 MILP-Taylor", ["q3_mixed_int4", "q3_mixed_int3"], "C4", "*", "-"),
    ]
    for lbl, cids, col, mk, ls in curves:
        xs = [float(rows[c]["packed_mb"]) for c in cids]
        ys = [float(rows[c]["wt2_ppl"]) for c in cids]
        ax.plot(xs, ys, color=col, marker=mk, ls=ls, lw=1.3, ms=5.5 if mk == "*" else 4.2, label=lbl)

    r24 = rows["p2_wanda_2_4_s50"]
    ax.scatter([float(r24["packed_mb"])], [float(r24["wt2_ppl"])], color="C5", marker="X", s=40, zorder=5, label="Wanda 2:4")

    ax.set_yscale("log")
    ax.set_ylim(10, 1e8)
    ax.set_xlabel("Packed model storage (MiB)")
    ax.set_ylabel("WikiText-2 perplexity (log)")
    ax.set_title("Compression size vs. perplexity")
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.9, fontsize=6.6, columnspacing=0.6)

    # Right: P3 block sparsity across layers 0-27
    ax = axes[1]
    layers = np.arange(28)
    s50 = [p3_50["meta"]["sparsity_map"][f"model.layers.{i}.mlp.down_proj"] * 100 for i in layers]
    s75 = [p3_75["meta"]["sparsity_map"][f"model.layers.{i}.mlp.down_proj"] * 100 for i in layers]
    ax.plot(layers, s50, color="C0", marker="o", ms=3, lw=1.3, label="P3 at 50% global")
    ax.plot(layers, s75, color="C1", marker="s", ms=3, lw=1.3, label="P3 at 75% global")
    ax.axhline(50, color="C0", ls=":", lw=0.9)
    ax.axhline(75, color="C1", ls=":", lw=0.9)
    ax.set_xlabel("Transformer block index")
    ax.set_ylabel("Assigned block sparsity (%)")
    ax.set_ylim(15, 85)
    ax.set_title("P3 ShortGPT-BI sparsity per block")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    save(fig, "fig1_task1")


# =============================================================================
# MAIN FIGURE 2 (Task 2): 2 simple panels — Part A memory + Part B distance bins
# =============================================================================
def make_fig2():
    pa = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_a_summary.csv")))
    pb = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_b_summary.csv")))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.75), gridspec_kw={"wspace": 0.28})

    # Left: Part A FP16 fwd+bwd peak allocated memory
    ax = axes[0]
    for op, lbl, col, mk in [("dense", "Dense", "C3", "o"), ("tiled_64", "Tiled-64 (Python)", "C1", "s"), ("sdpa", "SDPA (Flash)", "C0", "^")]:
        sub = sorted([r for r in pa if r["dtype"] == "fp16" and r["operator"] == op], key=lambda r: int(r["L"]))
        Ls = [int(r["L"]) for r in sub]
        mem = [float(r["fwdbwd_peak_alloc_mb"]) for r in sub]
        ax.plot(Ls, mem, color=col, marker=mk, lw=1.4, ms=4.5, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k"])
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("Part A: FP16 forward+backward memory")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Right: Part B MQAR K=16 accuracy by distance bin
    ax = axes[1]
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    x = np.arange(len(bins))
    configs = [
        ("dense", "mixed", "Dense (mixed)", "C0", "o", "-"),
        ("topk_64", "mixed", "Top-k=64 (B3)", "C2", "*", "-"),
        ("sliding_w64", "near", "Sliding W=64 (near)", "C1", "s", "--"),
        ("sliding_w64", "mixed", "Sliding W=64 (mixed)", "C3", "x", ":"),
        ("linear_elu1", "mixed", "Linear ELU+1 (mean)", "C4", "D", "-"),
        ("local_dilated", "mixed", "Local+dilated", "0.45", "^", "--"),
    ]
    for mech, reg, lbl, col, mk, ls in configs:
        r = [row for row in pb if row["mechanism"] == mech and int(row["train_k"]) == 16 and row["train_regime"] == reg][0]
        ys = [float(r[f"bin_{b}_mean"]) * 100 for b in bins]
        ax.plot(x, ys, color=col, marker=mk, ls=ls, lw=1.35, ms=6.0 if mk == "*" else 4.2, label=lbl)

    ax.axhline(100 / 64, color="0.5", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["0-31", "32-63", "64-127", "128-255"])
    ax.set_ylim(-3, 108)
    ax.set_xlabel("Key-query distance bin (K = 16)")
    ax.set_ylabel("Test recall accuracy (%)")
    ax.set_title("Part B: MQAR accuracy by distance")
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.37), frameon=True, framealpha=0.92, fontsize=6.6)

    save(fig, "fig2_task2")


# =============================================================================
# MAIN FIGURE 3 (Task 3): 2 simple panels — Prefix KL + Round time breakdown
# =============================================================================
def make_fig3():
    pa = list(csv.DictReader(open(RES / "task3" / "analysis" / "prefix_alignment_summary.csv")))
    sp = list(csv.DictReader(open(RES / "task3" / "analysis" / "speculative_summary.csv")))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.75), gridspec_kw={"wspace": 0.28})

    # Left: Common-prefix Forward KL on teacher bank by workload
    ax = axes[0]
    subs = ["math", "short_familiar", "short_shifted"]
    x = np.arange(len(subs))
    w = 0.25
    groups = [
        ("Original", ["original"], "0.55"),
        ("Off-policy (avg 2 seeds)", ["offpolicy_s6304", "offpolicy_s6305"], "C0"),
        ("On-policy (avg 2 seeds)", ["onpolicy_s6304", "onpolicy_s6305"], "C1"),
    ]
    for idx, (lbl, drafts, col) in enumerate(groups):
        vals = [statistics.mean(float(r["forward_kl"]) for r in pa if r["draft"] in drafts and r["bank"] == "teacher" and r["subsplit"] == s) for s in subs]
        ax.bar(x + (idx - 1) * w, vals, width=w, color=col, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(["Math", "Short familiar", "Short shifted"])
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("Forward KL on teacher bank")
    ax.set_title("Prefix alignment by workload")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=6.8)

    # Right: Decode time breakdown vs target-only (all 4 timing components separate)
    ax = axes[1]
    bars = [
        ("Orig\ng=2", ["original"], 2),
        ("Off-P\ng=2", ["offpolicy_s6304", "offpolicy_s6305"], 2),
        ("On-P\ng=2", ["onpolicy_s6304", "onpolicy_s6305"], 2),
        ("Orig\ng=4", ["original"], 4),
        ("Off-P\ng=4", ["offpolicy_s6304", "offpolicy_s6305"], 4),
        ("On-P\ng=4", ["onpolicy_s6304", "onpolicy_s6305"], 4),
    ]
    xb = np.arange(len(bars))
    d_ms, v_ms, a_ms, c_ms, tgt_ms = [], [], [], [], []
    for _, drafts, g in bars:
        sub = [r for r in sp if r["draft"] in drafts and int(r["gamma"]) == g]
        d = statistics.mean(float(r["draft_proposal_ms_mean"]) for r in sub)
        v = statistics.mean(float(r["target_verify_ms_mean"]) for r in sub)
        a = statistics.mean(float(r["accept_residual_ms_mean"]) for r in sub)
        c = statistics.mean(float(r["cache_update_ms_mean"]) for r in sub)
        d_ms.append(d); v_ms.append(v); a_ms.append(a); c_ms.append(c)
        tgt_ms.append((d + v + a + c) * statistics.mean(float(r["measured_speedup_decode"]) for r in sub))

    b2 = [d + v for d, v in zip(d_ms, v_ms)]
    b3 = [x + a for x, a in zip(b2, a_ms)]
    ax.bar(xb, d_ms, width=0.55, color="C1", label="Draft proposal")
    ax.bar(xb, v_ms, width=0.55, bottom=d_ms, color="C0", label="Target verify")
    ax.bar(xb, a_ms, width=0.55, bottom=b2, color="C2", label="Accept/residual")
    ax.bar(xb, c_ms, width=0.55, bottom=b3, color="C4", label="Cache update")
    ax.axhline(statistics.mean(tgt_ms), color="C3", ls="--", lw=1.2, label=f"Target-only ({statistics.mean(tgt_ms):.0f} ms)")
    ax.set_xticks(xb)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.2)
    ax.set_ylim(0, 1950)
    ax.set_ylabel("Mean decode time per prompt (ms)")
    ax.set_title("Speculative round-time breakdown (4 components)")
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, fontsize=6.2, columnspacing=0.5)

    save(fig, "fig3_task3")


# =============================================================================
# APPENDIX FIGURE 4 (Task 1): P3/Q3 sensitivities + Q3 bit map + Drift vs NLL
# =============================================================================
def make_fig4_app():
    rows = list(csv.DictReader(open(RES / "task1" / "summary_task1.csv")))
    p3_meta = json.load(open(RES / "task1" / "conditions" / "p3_nonuniform_s50.json"))["meta"]
    q3_meta = json.load(open(RES / "task1" / "conditions" / "q3_mixed_int3.json"))["meta"]
    p3_sens = p3_meta["sensitivity"]
    q3_sens = q3_meta["sensitivity"]
    q3_3 = q3_meta["bit_map"]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.1), gridspec_kw={"wspace": 0.28, "hspace": 0.38})

    # Top-left: P3 ShortGPT BI sensitivity scores across 28 blocks
    ax = axes[0, 0]
    layers = list(range(28))
    bi_vals = [p3_sens[f"block_{li}"] for li in layers]
    ax.bar(layers, bi_vals, width=0.72, color="C0")
    ax.set_xlabel("Decoder block")
    ax.set_ylabel("ShortGPT Block Influence (BI)")
    ax.set_title("(a) P3 sensitivity: ShortGPT BI per block")

    # Top-right: Q3 Taylor sensitivity by projection type (raw vs per-Mparam)
    ax = axes[0, 1]
    projs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    plbls = ["q", "k", "v", "o", "gate", "up", "down"]
    # Exact parameter counts per projection in Qwen2.5-1.5B (from p2 meta)
    p2_pp = json.load(open(RES / "task1" / "conditions" / "p2_wanda_s50.json"))["meta"]["per_projection"]
    numels = {item["proj_type"]: item["numel"] for item in p2_pp}
    pt_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    mean_per_mp = []
    for pr, pk in zip(projs, pt_keys):
        vals = [q3_sens[f"model.layers.{li}.{pr}"] / (numels[pk] / 1e6) for li in range(28)]
        mean_per_mp.append(statistics.mean(vals))
    ax.bar(np.arange(len(plbls)), mean_per_mp, width=0.62, color="C1")
    ax.set_xticks(np.arange(len(plbls)))
    ax.set_xticklabels(plbls)
    ax.set_xlabel("Projection type")
    ax.set_ylabel("Mean Taylor sensitivity / Mparams")
    ax.set_title("(b) Q3 sensitivity density (explains v/k upgrades)")

    # Bottom-left: Q3 INT3 bit map
    ax = axes[1, 0]
    mat = np.array([[q3_3[f"model.layers.{li}.{pr}"] for li in range(28)] for pr in projs])
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=2, vmax=4)
    ax.set_yticks(np.arange(len(projs)))
    ax.set_yticklabels(pt_keys, fontsize=7.0)
    ax.set_xticks([0, 7, 14, 21, 27])
    ax.set_xlabel("Decoder layer")
    ax.set_title("(c) Q3-INT3 bit allocation (2 / 3 / 4-bit)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, ticks=[2, 3, 4], pad=0.03, fraction=0.05)
    cbar.ax.tick_params(labelsize=7)

    # Bottom-right: Calibration drift vs WT2 NLL
    ax = axes[1, 1]
    px, py, qx, qy = [], [], [], []
    for r in rows:
        if not r["final_block_cum_rel_l2"]:
            continue
        x = float(r["final_block_cum_rel_l2"])
        y = float(r["wt2_nll"])
        if r["condition_id"].startswith("p"):
            px.append(x); py.append(y)
        else:
            qx.append(x); qy.append(y)
    ax.scatter(px, py, color="C1", marker="o", s=34, label="Pruning (rho = 0.933)")
    ax.scatter(qx, qy, color="C0", marker="D", s=30, label="Quantization (rho = 0.976)")
    ax.set_xlabel("Final-block cumulative relative L2 error")
    ax.set_ylabel("Held-out WikiText-2 NLL")
    ax.set_title("(d) Calibration drift vs. WT2 NLL (all rho = 0.971)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.0)

    save(fig, "fig4_app_task1")


# =============================================================================
# APPENDIX FIGURE 5 (Task 2): MQAR curves (K=4,8,16) + Memory + Part C Pos & Dist
# =============================================================================
def make_fig5_app():
    runs_dir = RES / "task2" / "part_b" / "runs"
    inf = json.load(open(RES / "task2" / "part_b" / "inference_sweep.json"))
    bk = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_c_breakdowns.csv")))

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.1), gridspec_kw={"wspace": 0.28, "hspace": 0.38})

    # Top-left: MQAR learning curves across K=4, 8, 16
    ax = axes[0, 0]
    for prefix, lbl, col, ls in [
        ("B1_dense_K16_mixed", "Dense (K=16)", "C0", "-"),
        ("B3_topk_64_K16_mixed", "Top-k=64 (K=16)", "C2", "-"),
        ("B2_linear_elu1_K4_mixed", "Linear (K=4, mean)", "C4", ":"),
        ("B2_linear_elu1_K8_mixed", "Linear (K=8, 3s)", "C1", "--"),
        ("B2_linear_elu1_K16_mixed", "Linear (K=16, 3s)", "C4", "--"),
        ("B1_sliding_w64_K16_mixed", "Sliding (K=16 mixed)", "C3", "-."),
        ("B2_local_dilated_K16_mixed", "Local+dil (K=16)", "0.45", ":"),
    ]:
        curves = []
        steps = None
        for s in [6304, 6305, 6306]:
            lc = json.load(open(runs_dir / f"{prefix}_s{s}.json"))["learning_curve"]
            steps = [pt["step"] for pt in lc]
            accs = [pt["dev_acc"] * 100 for pt in lc]
            curves.append(accs)
            if prefix in ("B2_linear_elu1_K8_mixed", "B2_linear_elu1_K16_mixed"):
                ax.plot(steps, accs, color=col, ls=ls, lw=0.9, alpha=0.85, label=lbl if s == 6304 else None)
        if prefix not in ("B2_linear_elu1_K8_mixed", "B2_linear_elu1_K16_mixed"):
            ax.plot(steps, np.mean(curves, axis=0), color=col, ls=ls, lw=1.35, label=lbl)
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Dev recall accuracy (%)")
    ax.set_title("(a) MQAR learning curves (K=4, 8, 16)")
    ax.set_ylim(-2, 108)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=5.6, borderpad=0.3, handlelength=1.8)

    # Top-right: Part B inference memory
    ax = axes[0, 1]
    for mech, lbl, col, mk, ls in [
        ("dense", "Dense", "C0", "o", "-"),
        ("sliding_w64", "Sliding (masked)", "C1", "s", "--"),
        ("topk_64", "Top-k=64 (masked)", "C2", "*", "-"),
        ("linear_elu1", "Linear (state)", "C4", "D", "-"),
    ]:
        sub = sorted([r for r in inf if r["mechanism"] == mech], key=lambda r: r["T"])
        ax.plot([r["T"] for r in sub], [r["fwd_memory"]["peak_allocated_bytes"] / (1024**2) for r in sub],
                color=col, marker=mk, ls=ls, lw=1.3, ms=4.5, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([256, 1024, 4096, 8192])
    ax.set_xticklabels(["256", "1k", "4k", "8k"])
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("(b) Part B: Forward inference memory")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.5)

    # Bottom-left: Part C exact match by evidence position
    ax = axes[1, 0]
    pols = [("full_fp16", "FP16"), ("evict_sink_recent", "Sink"), ("evict_question_aware", "Q-Aw"),
            ("quant_int4_tok", "Tok4"), ("quant_int4_kivi", "KIVI4"), ("hybrid_fp16_int2", "Hyb")]
    x = np.arange(len(pols))
    w = 0.25
    for idx, (pos, lbl, col) in enumerate([("start", "Start", "C0"), ("middle", "Middle", "C1"), ("end", "End", "C2")]):
        vals = [float([r for r in bk if r["policy"] == p and r["factor"] == "position" and r["level"] == pos][0]["exact_match"]) * 100 for p, _ in pols]
        ax.bar(x + (idx - 1) * w, vals, width=w, color=col, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pols], fontsize=7.2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Exact match (%)")
    ax.set_title("(c) Part C: Accuracy by evidence position")
    ax.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.9, fontsize=6.5, columnspacing=0.4)

    # Bottom-right: Part C exact match by distractor count
    ax = axes[1, 1]
    for idx, (lvl, lbl, col) in enumerate([("0", "0 dist", "C0"), ("2", "2 dist", "C1"), ("8", "8 dist", "C3")]):
        vals = [float([r for r in bk if r["policy"] == p and r["factor"] == "distractors" and r["level"] == lvl][0]["exact_match"]) * 100 for p, _ in pols]
        ax.bar(x + (idx - 1) * w, vals, width=w, color=col, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pols], fontsize=7.2)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Exact match (%)")
    ax.set_title("(d) Part C: Accuracy by distractor count")
    ax.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.9, fontsize=6.5, columnspacing=0.4)

    save(fig, "fig5_app_task2")


# =============================================================================
# APPENDIX FIGURE 6 (Task 3): Distillation KL curves + Analytical speedup curve
# =============================================================================
def make_fig6_app():
    sp = list(csv.DictReader(open(RES / "task3" / "analysis" / "speculative_summary.csv")))
    isc = json.load(open(RES / "task3" / "spec_bench" / "isolated_step_costs.json"))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.75), gridspec_kw={"wspace": 0.28})

    # Left: Distillation KL over teacher-scored tokens
    ax = axes[0]
    for rn, lbl, col, ls in [
        ("offpolicy_s6304", "Off-policy s6304", "C0", "-"),
        ("offpolicy_s6305", "Off-policy s6305", "C0", "--"),
        ("onpolicy_s6304", "On-policy s6304", "C1", "-"),
        ("onpolicy_s6305", "On-policy s6305", "C1", "--"),
    ]:
        d = json.load(open(RES / "task3" / "distill" / f"{rn}.json"))
        steps = d.get("learning_curve", [])
        if steps:
            xs = [s["teacher_scored_tokens"] / 1000.0 for s in steps]
            ys = [s["mean_kl_loss"] for s in steps]
            k = 5
            ys_s = np.convolve(ys, np.ones(k) / k, mode="valid")
            xs_s = xs[k - 1:]
            ax.plot(xs_s, ys_s, color=col, ls=ls, lw=1.2, label=lbl)
    ax.set_xlabel("Teacher-scored tokens (thousands)")
    ax.set_ylabel("Training forward KL (5-step mean)")
    ax.set_title("Distillation training curves")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=6.8)

    # Right: Analytical speedup S_hat vs empirical
    ax = axes[1]
    c_meas = statistics.mean(r["c_ratio"] for r in isc["by_prompt_len"])
    v2 = statistics.mean(r["verify_passes"]["2"]["v_gamma"] for r in isc["by_prompt_len"])
    v4 = statistics.mean(r["verify_passes"]["4"]["v_gamma"] for r in isc["by_prompt_len"])
    alphas = np.linspace(0.60, 0.99, 200)
    def shat(a, g, c, v):
        return ((1.0 - a**(g + 1)) / (1.0 - a)) / (g * c + v)
    ax.plot(alphas * 100, shat(alphas, 2, c_meas, v2), color="C0", lw=1.4, label=f"Model g=2 (c={c_meas:.2f})")
    ax.plot(alphas * 100, shat(alphas, 4, c_meas, v4), color="C1", lw=1.4, label=f"Model g=4 (c={c_meas:.2f})")
    ax.plot(alphas * 100, shat(alphas, 4, 0.25, v4), color="C2", ls="--", lw=1.2, label=f"Counterfactual g=4 (c=0.25, v={v4:.2f})")
    ax.axhline(1.0, color="0.4", ls=":", lw=0.9)
    for g, col, mk in [(2, "C0", "o"), (4, "C1", "s")]:
        sub = [r for r in sp if int(r["gamma"]) == g]
        ax.scatter([float(r["acc_over_examined"]) * 100 for r in sub],
                   [float(r["measured_speedup_decode"]) for r in sub],
                   color=col, marker=mk, s=26, edgecolor="black", lw=0.5, label=f"Measured g={g}")
    ax.set_xlim(60, 100)
    ax.set_ylim(0.55, 2.30)
    ax.set_xlabel("Per-token acceptance rate (%)")
    ax.set_ylabel("Speedup over target-only (x)")
    ax.set_title("Analytical vs. measured speedup")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.6)

    save(fig, "fig6_app_task3")


# =============================================================================
# APPENDIX FIGURE 5b (Task 2 Part B): Distance-by-load MQAR heatmap
# =============================================================================
def make_fig5b_app():
    rows = list(csv.DictReader(open(RES / "task2" / "analysis" / "part_b_summary.csv")))
    order = [
        ("dense", "4", "mixed"), ("dense", "8", "mixed"), ("dense", "16", "mixed"), ("dense", "16", "near"),
        ("sliding_w64", "4", "near"), ("sliding_w64", "8", "near"), ("sliding_w64", "16", "near"), ("sliding_w64", "16", "mixed"),
        ("local_dilated", "4", "mixed"), ("local_dilated", "16", "mixed"),
        ("linear_elu1", "4", "mixed"), ("linear_elu1", "8", "mixed"), ("linear_elu1", "16", "mixed"),
        ("topk_64", "16", "mixed"),
    ]
    rmap = {(r["mechanism"], r["train_k"], r["train_regime"]): r for r in rows}
    labels, mat = [], []
    for mech, k, reg in order:
        r = rmap[(mech, k, reg)]
        labels.append(f"{mech} (K={k}, {reg})")
        mat.append([float(r["near_acc_mean"]) * 100, float(r["mixed_acc_mean"]) * 100, float(r["far_acc_mean"]) * 100])
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Near (<64)", "Mixed", "Far (>=128)"], fontsize=8.0)
    ax.set_title("MQAR held-out accuracy (%) by model, training K, and distance regime")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=7.2)
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.045)
    cbar.set_label("Accuracy (%)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    save(fig, "fig5b_app_mqar_heatmap")


if __name__ == "__main__":
    # Remove old over-styled figures so report/figures/ stays completely clean
    for old in OUT.glob("*"):
        old.unlink()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4_app()
    make_fig5_app()
    make_fig5b_app()
    make_fig6_app()
    print("All clean simple figures generated!")
