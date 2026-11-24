"""Task 2 Part C runner: Qwen2.5-1.5B-Instruct KV validation (C1), 5 compressed policies (C2-C4), and C5 benchmarks."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.env import COMMON_SEED, get_env_metadata, set_seed
from src.task2.part_c_kv import (
    benchmark_c5_packed_encode_decode,
    evaluate_single_kv_case,
    load_kv_retrieval_cases,
    tokenize_kv_case_split,
    validate_c1_full_cache_equivalence,
)

POLICIES = (
    "full_fp16",
    "evict_sink_recent",
    "evict_question_aware",
    "quant_int4_tok",
    "quant_int4_kivi",
    "hybrid_fp16_int2",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task2" / "part_c")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    resume = not args.no_resume

    set_seed(COMMON_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies_dir = args.output_dir / "policies"
    policies_dir.mkdir(exist_ok=True)

    env_meta = get_env_metadata(args.dataset_root)
    rev = env_meta.get("source_revisions", {}).get("revisions", {}).get("model_target")
    (args.output_dir / "env.json").write_text(json.dumps(env_meta, indent=2), encoding="utf-8")

    # 1. C5 Standalone Packed Encode/Decode Benchmark (independent of full model)
    c5_bench_path = args.output_dir / "c5_packed_encode_decode_bench.json"
    if not (resume and c5_bench_path.exists()):
        print(f"Running C5 standalone packed encode/decode benchmark on {env_meta['device_name']}...", flush=True)
        c5_bench = benchmark_c5_packed_encode_decode(
            batch_sizes=(1, 8),
            lengths=(2048, 4096),
            repeats=10,
            device=device,
        )
        c5_bench_path.write_text(json.dumps(c5_bench, indent=2), encoding="utf-8")

    # 2. Load Qwen2.5-1.5B-Instruct (FP16, SDPA)
    print(f"Loading Qwen/Qwen2.5-1.5B-Instruct (rev={rev})...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision=rev, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision=rev,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()

    cases = load_kv_retrieval_cases(args.dataset_root / "data" / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz")
    splits = [tokenize_kv_case_split(tok, c) for c in cases]

    # Save exact encoded lengths audit
    length_audit = [
        {
            "id": c["id"],
            "target_tokens": c["target_tokens"],
            "position": c["position"],
            "distractors": c["distractors"],
            "L_ctx": s["L_ctx"],
            "L_q": s["L_q"],
            "L_total": s["L_total"],
            "n_evidence_tokens": len(s["evidence_token_indices"]),
        }
        for c, s in zip(cases, splits)
    ]
    (args.output_dir / "token_length_audit.json").write_text(json.dumps(length_audit, indent=2), encoding="utf-8")

    # 3. C1 Full-forward vs split-prefill equivalence checks (one prompt per factorial cell = 18 cells)
    c1_val_path = args.output_dir / "c1_equivalence_checks.json"
    if not (resume and c1_val_path.exists()):
        seen_cells = set()
        c1_checks = []
        for c, s in zip(cases, splits):
            cell = (c["target_tokens"], c["position"], c["distractors"])
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            c1_checks.append(validate_c1_full_cache_equivalence(model, tok, c, s))
        c1_val_path.write_text(json.dumps(c1_checks, indent=2), encoding="utf-8")
        max_diff = max(x["q_logits_max_abs_diff"] for x in c1_checks)
        print(f"C1 split-prefill equivalence checked on {len(c1_checks)} cells: max_abs_diff={max_diff:.5f}", flush=True)

    # 4. Run full_fp16 first and cache pre-answer logits for post-prefill KL
    ref_logits_path = args.output_dir / "fp16_ref_logits_last.pt"
    fp16_out_path = policies_dir / "full_fp16.jsonl.gz"
    ref_logits_map: dict[str, torch.Tensor] = {}

    if resume and fp16_out_path.exists() and ref_logits_path.exists():
        print("Loading cached full_fp16 reference logits...", flush=True)
        ref_logits_map = torch.load(ref_logits_path, map_location="cpu", weights_only=False)
    else:
        print("Evaluating full_fp16 across 180 retrieval prompts...", flush=True)
        rows = []
        for idx, (c, s) in enumerate(zip(cases, splits)):
            res = evaluate_single_kv_case(model, tok, c, s, "full_fp16", max_new_tokens=args.max_new_tokens)
            ref_logits_map[c["id"]] = res.pop("logits_last_cpu")
            res.pop("selection_mask_sample", None)
            rows.append(res)
            if (idx + 1) % 30 == 0:
                acc_so_far = sum(r["exact_match"] for r in rows) / len(rows)
                print(f"  [full_fp16] {idx + 1}/180 | acc={acc_so_far:.4f}", flush=True)
        torch.save(ref_logits_map, ref_logits_path)
        with gzip.open(fp16_out_path, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Full FP16 overall accuracy: {sum(r['exact_match'] for r in rows) / len(rows):.4f}", flush=True)

    # Save aligned qualitative selection masks on 6 representative prompts (3 lengths/positions)
    sample_qual_ids = {
        "KV-2048-start-8-00",
        "KV-2048-middle-8-00",
        "KV-2048-end-8-00",
        "KV-4096-start-8-00",
        "KV-4096-middle-8-00",
        "KV-4096-end-8-00",
    }
    qual_masks_dir = args.output_dir / "qualitative_masks"
    qual_masks_dir.mkdir(exist_ok=True)

    # 5. Evaluate the 5 compressed policies at matched INT4 budget
    for policy in POLICIES[1:]:
        p_out = policies_dir / f"{policy}.jsonl.gz"
        if resume and p_out.exists():
            print(f"Skipping completed policy {policy}", flush=True)
            continue
        print(f"\nEvaluating policy: {policy} across 180 prompts...", flush=True)
        rows = []
        masks_saved = []
        for idx, (c, s) in enumerate(zip(cases, splits)):
            ref_l = ref_logits_map[c["id"]].to(device)
            res = evaluate_single_kv_case(
                model, tok, c, s, policy, fp16_ref_logits_last=ref_l, max_new_tokens=args.max_new_tokens
            )
            res.pop("logits_last_cpu", None)
            mask_sample = res.pop("selection_mask_sample", None)
            if c["id"] in sample_qual_ids:
                masks_saved.append({
                    "id": c["id"],
                    "policy": policy,
                    "gold_code": c["code"],
                    "pred_code": res["pred_code"],
                    "generated_text": res["generated_text"],
                    "exact_match": res["exact_match"],
                    "evidence_token_indices": s["evidence_token_indices"],
                    "selection_mask_sample": mask_sample,
                })
            rows.append(res)
            if (idx + 1) % 30 == 0:
                acc_so_far = sum(r["exact_match"] for r in rows) / len(rows)
                print(f"  [{policy}] {idx + 1}/180 | acc={acc_so_far:.4f}", flush=True)

        with gzip.open(p_out, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        if masks_saved:
            (qual_masks_dir / f"{policy}_masks.json").write_text(json.dumps(masks_saved, indent=2), encoding="utf-8")
        acc = sum(r["exact_match"] for r in rows) / len(rows)
        print(f"Finished {policy}: exact_match={acc:.4f}", flush=True)

    print("Part C complete.", flush=True)


if __name__ == "__main__":
    main()
