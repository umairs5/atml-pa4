# ATML PA4 — Efficient Inference & Compression for Language Models

**Course:** EE-5102 / CS-6304 *Advanced Topics in Machine Learning* (Fall 2026), LUMS  
**Author:** Muhammad Umair (`25280013@lums.edu.pk`)  
**Repository:** https://github.com/umairs5/atml-pa4

This repository contains the complete, reproducible implementation, Colab notebooks, unit tests, lightweight machine-readable summaries, and NeurIPS-style report for Programming Assignment 4 covering:
1. **Task 1 — Post-Training Model Compression** (`Qwen/Qwen2.5-1.5B` FP16): Uniform Magnitude & Wanda pruning (25/50/75\% + 2:4), dev-selected non-uniform pruning (`ShortGPT-BI`), uniform group-128 asymmetric RTN & AWQ quantization (INT8/4/3), and dev-selected Taylor-weighted mixed-precision MILP (`Q3`) at exact byte budgets.
2. **Task 2 — Efficient Attention & Context Memory**:
   - *Part A:* Exact online-softmax tiled attention vs. dense attention and PyTorch SDPA (`FLASH_ATTENTION` / `EFFICIENT_ATTENTION`) across $L \in \{256,\dots,8192\}$ (FP16/FP32).
   - *Part B:* Matched-parameter (436,736 params) Multi-Query Associative Recall (MQAR) across 5 attention mechanisms (`dense`, `sliding_w64`, `linear_elu1`, `local_dilated`, `topk_64`), $K \in \{4,8,16\}$, 3 seeds (42 runs).
   - *Part C:* KV-cache compression on `Qwen/Qwen2.5-1.5B-Instruct` ($n=180$ long-context prompts) at matched INT4 KIVI budget (`full_fp16`, `evict_sink_recent`, `evict_question_aware`, `quant_int4_tok`, `quant_int4_kivi`, `hybrid_fp16_int2`) + real bit-packed encode/decode benchmark (C5).
3. **Task 3 — Policy Distillation & Speculative Decoding** (`Qwen/Qwen2.5-1.5B-Instruct` target, `Qwen/Qwen2.5-0.5B-Instruct` + LoRA `r=8, α=16` draft): Exactness validation (100k-trial 3-symbol Markov $\chi^2$, broken residual control, self-speculative & KV rollback checks), off-policy vs. on-policy KL distillation (40k tokens, 2 seeds), common-prefix & own-rollout alignment, live speculative decoding (`120 × 2` evals + 24-prompt ignore-EOS diagnostic), analytical cost model & amortization, and standalone accuracy.

---

## Hardware & Software Environment

All GPU experiments were executed on a single **NVIDIA L4 (24 GiB, Ada Lovelace `sm_89`)** via Google Colab Pro:
- Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`
- `transformers==4.57.1`, `huggingface_hub==0.36.2`, `peft`, `datasets`, `scipy`, `matplotlib`
- Common base seed: `6304` (multi-seed runs: `6304, 6305, 6306` in Task 2B; `6304, 6305` in Task 3)
- Exact Hugging Face dataset and model commit SHAs are locked in `configs/local_source_revisions.json`.

---

## Repository Structure

```text
atml-pa4/
├── configs/                    # Locked HF source revisions & configs
├── notebooks/                  # Turnkey Google Colab L4 execution notebooks (Tasks 1–3)
├── report/                     # NeurIPS 2019 LaTeX report, figures/, make_figures.py, references.bib
├── results/                    # Lightweight machine-readable CSV/JSON summaries & bootstrap CIs
├── scripts/                    # CLI entry points for running & analyzing Tasks 1–3
├── src/
│   ├── common/                 # Environment capture, deterministic seeding, paired bootstrap CIs
│   ├── task1/                  # Pruning (P1–P3), quantization (Q1–Q3), PPL/HellaSwag/latency eval
│   ├── task2/                  # Tiled attention (Part A), MQAR (Part B), KV compression (Part C)
│   └── task3/                  # Speculative sampler, LoRA distillation, prefix & live evaluation
└── tests/                      # Unit test suites (Task 1: 6/6, Task 2: 4/4, Task 3: 3/3 passed on L4)
```

---

## Reproduction Instructions

### 1. Setup & Data Materialization
Clone the course dataset release repository alongside this repo and materialize locked public splits:
```bash
pip install -r requirements.txt
# Note: On Colab, uninstall torchao if present to avoid PEFT import conflicts:
# pip uninstall -y torchao

git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git ../atml-pa4-datasets
cd ../atml-pa4-datasets
python scripts/setup_data.py --public
python scripts/check_setup.py
cd ../atml-pa4
```

> **Compatibility Note (`transformers==4.57.1`):** Upstream `setup_data.py --prefix-banks` passes `generator=` into `model.generate()`, which raises a `ValueError` in pinned `transformers==4.57.1`. We provide `scripts/generate_prefix_banks_compat.py`, which wraps generation with `torch.random.fork_rng` + `torch.manual_seed` to preserve exact upstream prompt ordering, seeds, temperature (`1.0`, `top_p=1.0`), and sampled positions.

### 2. Run Unit Tests
```bash
python -m unittest tests/test_task1.py tests/test_task2.py tests/test_task3.py
```

### 3. Run Experiments & Analysis
You can run via the turnkey Colab notebooks (`notebooks/colab_task{1,2,3}.ipynb`) or directly via CLI (set `--data-dir` to your `atml-pa4-datasets/data` path):

```bash
# Task 1: Model Compression (18 conditions + analysis)
python scripts/run_task1.py --data-dir ../atml-pa4-datasets/data --out-dir results/task1
python scripts/analyze_task1.py --results-dir results/task1

# Task 2: Efficient Attention & Context Memory (Parts A, B, C + analysis)
python scripts/run_task2_a.py --out-dir results/task2/part_a
python scripts/run_task2_b.py --data-dir ../atml-pa4-datasets/data --out-dir results/task2/part_b
python scripts/run_task2_c.py --data-dir ../atml-pa4-datasets/data --out-dir results/task2/part_c
python scripts/analyze_task2.py --results-dir results/task2

# Task 3: Policy Distillation & Speculative Decoding
python scripts/generate_prefix_banks_compat.py --data-dir ../atml-pa4-datasets/data
python scripts/run_task3.py --data-dir ../atml-pa4-datasets/data --out-dir results/task3
python scripts/analyze_task3.py --results-dir results/task3

# Regenerate all report figures (vector PDF + PNG)
python report/make_figures.py
```

### 4. Compile Report
Upload `report/main.tex`, `report/references.bib`, `report/neurips_2019.sty`, and `report/figures/*.pdf` (bundled in `report/ATML_PA4_Report_Overleaf.zip`) to Overleaf and compile with `pdfLaTeX`.

---

## Attributions & References

- **Models & Datasets:** Qwen2.5 (`Qwen/Qwen2.5-1.5B`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-0.5B-Instruct`; Qwen Team, 2024); WikiText-2 (Merity et al., 2017); C4 (Raffel et al., 2020); HellaSwag (Zellers et al., 2019); GSM8K (Cobbe et al., 2021); locked PA4 splits and synthetic MQAR/KV/short-task assets from the course release (`AbDu11aHHH/atml-pa4-datasets`).
- **Methods Implemented:** Wanda (Sun et al., 2024), AWQ (Lin et al., 2024), OWL (Yin et al., 2024), ShortGPT Block Influence (Men et al., 2024), Online-Softmax / FlashAttention (Milakov & Gimelshein, 2018; Dao et al., 2022, 2024), Zoology MQAR (Arora et al., 2024), Linear Attention (Katharopoulos et al., 2020), StreamingLLM (Xiao et al., 2024), H2O/SnapKV-style question-aware eviction (Zhang et al., 2023; Li et al., 2024), KIVI asymmetric KV quantization (Liu et al., 2024), Lossless Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023), DistillSpec & On-Policy GKD (Zhou et al., 2024; Agarwal et al., 2024), and LoRA (Hu et al., 2022).
- **Libraries:** PyTorch, Hugging Face `transformers`, `datasets`, `peft`, `scipy`, `matplotlib`. All experimental code in `src/` and `scripts/` was written from scratch for this assignment (no external pre-built compression or speculative decoding packages were imported).
