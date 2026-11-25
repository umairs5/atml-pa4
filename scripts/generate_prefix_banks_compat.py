"""Compatible prefix-bank generator for transformers==4.57.1.

Course script `atml-pa4-datasets/scripts/setup_data.py --prefix-banks` passes `generator=gen`
into `model.generate()`, which `transformers==4.57.1` rejects via `_validate_model_kwargs`.
This script reproduces the exact prompt order, seeds, chat template, and position selection
using `torch.random.fork_rng` + `torch.manual_seed`, saving `prefix_banks_locked.jsonl.gz`.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "atml-pa4-datasets")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run prefix-bank generation on a CUDA GPU")

    droot = args.dataset_root
    lock = json.loads((droot / "data" / "local_source_revisions.json").read_text(encoding="utf8"))
    rev = lock["revisions"]
    repos = lock["repos"]

    dest = droot / "data" / "prefix_banks" / "prefix_banks_locked.jsonl.gz"
    if dest.exists() and not args.force:
        print(f"Prefix banks already exist at {dest}; skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(repos["model_target"], revision=rev["model_target"])
    draft_tok = AutoTokenizer.from_pretrained(repos["model_draft"], revision=rev["model_draft"])
    for s in ["Example 123", "Hello there", "A complicated example with \U0001d70b"]:
        if tok.encode(s) != draft_tok.encode(s):
            raise RuntimeError("Target/draft tokenizers differ")

    gsm = load_dataset(repos["gsm8k"], "main", split="test", revision=rev["gsm8k"])
    ms = read_jsonl(droot / "data" / "public_manifests" / "gsm8k_test_ids.jsonl")
    prompts = [(f"GSM8K-{r['order']:04d}", gsm[r["row_index"]]["question"], "math") for r in ms]
    prompts.extend((x["id"], x["input"], "short") for x in read_jsonl(droot / "data" / "short_form" / "short_test.jsonl"))

    device = torch.device("cuda")
    output = []
    for which, key in [("teacher", "model_target"), ("original_draft", "model_draft")]:
        print("Generating prefix bank:", which, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            repos[key], revision=rev[key], torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(device).eval()

        for j, (pid, prompt, domain) in enumerate(prompts):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            seed_val = 6304 + 10007 * j + (0 if which == "teacher" else 123)
            with torch.inference_mode():
                with torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(seed_val)
                    seq = model.generate(
                        ids,
                        max_new_tokens=192,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        pad_token_id=tok.eos_token_id,
                    )[0, ids.shape[1] :].tolist()

            rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}")
            positions = []
            for lo, hi, n in [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]:
                candidates = list(range(lo, min(hi, len(seq)) + 1))
                positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
            for pos in positions:
                output.append(
                    dict(
                        prompt_id=pid,
                        domain=domain,
                        bank=which,
                        position=pos,
                        prompt=prompt,
                        response_prefix_token_ids=seq[: pos - 1],
                        source_length=len(seq),
                        source_generation_seed=seed_val,
                    )
                )
            if (j + 1) % 20 == 0:
                print(which, j + 1, "/", len(prompts), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with gzip.open(dest, "wt", encoding="utf8") as f:
        for x in output:
            f.write(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"Wrote {dest} ({dest.stat().st_size:,} bytes, {len(output)} records)", flush=True)


if __name__ == "__main__":
    main()
