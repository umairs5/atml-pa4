"""Task 2 Part B runner: MQAR training (B1/B2/B3 across seeds 6304..6306), controls, and sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.env import get_env_metadata, set_seed
from src.task2.part_b_mqar import (
    MQAR_NUM_CLASSES,
    MQAR_T,
    MQAR_VALUE_OFFSET,
    benchmark_part_b_inference_sweep,
    evaluate_mqar_records,
    load_mqar_jsonl_gz,
    theoretical_cost_at_T256,
    train_mqar_run,
)

SEEDS = (6304, 6305, 6306)

# (group, mechanism, train_k, train_regime, run_w64_test_control)
RUN_SPECS = [
    # B1: dense mixed K=4,8,16 + dense near K=16 control
    ("B1", "dense", 4, "mixed", True),
    ("B1", "dense", 8, "mixed", True),
    ("B1", "dense", 16, "mixed", True),
    ("B1", "dense", 16, "near", True),
    # B1: sliding window W=64 near K=4,8,16 + mixed K=16
    ("B1", "sliding_w64", 4, "near", False),
    ("B1", "sliding_w64", 8, "near", False),
    ("B1", "sliding_w64", 16, "near", False),
    ("B1", "sliding_w64", 16, "mixed", False),
    # B2: ELU+1 linear attention mixed K=4,8,16
    ("B2", "linear_elu1", 4, "mixed", False),
    ("B2", "linear_elu1", 8, "mixed", False),
    ("B2", "linear_elu1", 16, "mixed", False),
    # B2: Mixed local/dilated sparse mixed K=4,16
    ("B2", "local_dilated", 4, "mixed", False),
    ("B2", "local_dilated", 16, "mixed", False),
    # B3: Controlled intervention (content-selected topk_64) at K=16 mixed
    ("B3", "topk_64", 16, "mixed", False),
]


def _worker_gen_chunk(args: tuple[str, int, str, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scripts_dir, k, regime, start_idx, count = args
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from mqar_generator import make_example

    toks = np.empty((count, MQAR_T), dtype=np.int64)
    qpos = np.empty((count, k), dtype=np.int64)
    ans = np.empty((count, k), dtype=np.int64)
    for i in range(count):
        ex = make_example("train", k, regime, start_idx + i)
        toks[i] = ex["tokens"]
        qpos[i] = ex["query_positions"]
        ans[i] = np.asarray(ex["answer_ids"], dtype=np.int64) - MQAR_VALUE_OFFSET
    return toks, qpos, ans


def build_training_pool_parallel(
    dataset_root: Path,
    k: int,
    regime: str,
    total_examples: int = 4500 * 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scripts_dir = str(dataset_root / "scripts")
    n_workers = 4
    chunk = total_examples // n_workers
    tasks = [
        (scripts_dir, k, regime, w * chunk, chunk if w < n_workers - 1 else (total_examples - w * chunk))
        for w in range(n_workers)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_worker_gen_chunk, tasks))
    toks = torch.from_numpy(np.concatenate([p[0] for p in parts], axis=0))
    qpos = torch.from_numpy(np.concatenate([p[1] for p in parts], axis=0))
    ans = torch.from_numpy(np.concatenate([p[2] for p in parts], axis=0))
    return toks, qpos, ans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task2" / "part_b")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    resume = not args.no_resume

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    ckpts_dir = args.output_dir / "checkpoints"
    runs_dir.mkdir(exist_ok=True)
    ckpts_dir.mkdir(exist_ok=True)

    env_meta = get_env_metadata(args.dataset_root)
    (args.output_dir / "env_and_theory.json").write_text(
        json.dumps({"env": env_meta, "theoretical_T256": theoretical_cost_at_T256()}, indent=2),
        encoding="utf-8",
    )

    # Verify generator invariant checks first
    scripts_dir = str(args.dataset_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from mqar_generator import make_example

    for reg in ("near", "far", "mixed"):
        for k in (4, 8, 16):
            make_example("train", k, reg, 0)
            make_example("development", k, reg, 0)

    dev_records = load_mqar_jsonl_gz(args.dataset_root / "data" / "mqar" / "mqar_development.jsonl.gz")
    test_records = load_mqar_jsonl_gz(args.dataset_root / "data" / "mqar" / "mqar_test.jsonl.gz")
    print(f"Loaded MQAR dev={len(dev_records)}, test={len(test_records)} on {env_meta['device_name']}", flush=True)

    # Inference sweep at T in {256, 1024, 4096, 8192}
    sweep_path = args.output_dir / "inference_sweep.json"
    if not (resume and sweep_path.exists()):
        print("Running Part B forward-only inference sweep T in {256, 1024, 4096, 8192}...", flush=True)
        sweep = benchmark_part_b_inference_sweep(lengths=(256, 1024, 4096, 8192), device=device, repeats=10)
        sweep_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    # Pre-generate training streams lazily per (K, regime)
    stream_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for group, mech, k, regime, run_w64_ctrl in RUN_SPECS:
        for seed in SEEDS:
            run_id = f"{group}_{mech}_K{k}_{regime}_s{seed}"
            out_json = runs_dir / f"{run_id}.json"
            if resume and out_json.exists():
                continue

            if (k, regime) not in stream_cache:
                t0 = time.perf_counter()
                print(f"Generating 144,000 MQAR training stream for K={k}, regime={regime}...", flush=True)
                stream_cache[(k, regime)] = build_training_pool_parallel(args.dataset_root, k, regime)
                print(f"  Generated in {time.perf_counter() - t0:.1f}s", flush=True)

            tok_pool, qpos_pool, ans_pool = stream_cache[(k, regime)]
            perm_gen = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(tok_pool), generator=perm_gen)
            shuffled_tensors = (tok_pool[perm], qpos_pool[perm], ans_pool[perm])

            model, train_info = train_mqar_run(
                mechanism=mech,
                train_k=k,
                train_regime=regime,
                seed=seed,
                train_tensors=shuffled_tensors,
                dev_records=dev_records,
                device=device,
            )

            # Evaluate final checkpoint on ALL 9,216 held-out test records
            test_eval = evaluate_mqar_records(model, test_records, device=device, batch_size=128)
            test_ctrl = None
            if run_w64_ctrl:
                test_ctrl = evaluate_mqar_records(
                    model, test_records, device=device, batch_size=128, override_mechanism="sliding_w64"
                )

            torch.save(model.state_dict(), ckpts_dir / f"{run_id}.pt")
            record = {
                "run_id": run_id,
                "group": group,
                **train_info,
                "test_eval": test_eval,
                "test_eval_w64_mask_control": test_ctrl,
            }
            out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
            near_acc = test_eval["by_cell"][f"K{k}_near"]["acc"]
            far_acc = test_eval["by_cell"][f"K{k}_far"]["acc"]
            print(
                f"  [{run_id}] train={train_info['train_time_sec']:.1f}s | "
                f"Test(K={k}) near={near_acc:.3f}, far={far_acc:.3f}, overall={test_eval['accuracy']:.3f}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Part B complete.", flush=True)


if __name__ == "__main__":
    main()

