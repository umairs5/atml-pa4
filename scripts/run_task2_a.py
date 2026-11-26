"""Task 2 Part A runner: dense causal, online-softmax tiled, and PyTorch SDPA benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.env import COMMON_SEED, get_env_metadata, set_seed
from src.task2.part_a_attention import benchmark_attention_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task2" / "part_a")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    set_seed(COMMON_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "part_a_results.json"

    env_meta = get_env_metadata(args.dataset_root)
    print(f"Running Task 2 Part A on {env_meta['device_name']} (repeats={args.repeats})", flush=True)
    suite = benchmark_attention_suite(
        lengths=(256, 512, 1024, 2048, 4096, 8192),
        dtypes=("fp16", "fp32"),
        B=2,
        H=4,
        d_h=64,
        repeats=args.repeats,
        warmup=3,
        device=device,
    )
    payload = {"env": env_meta, **suite}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Part A results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

