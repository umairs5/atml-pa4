# ATML Fall 2026 — Programming Assignment 4

Experiments and analysis for **EE-5102 / CS-6304 Programming Assignment 4**:
1. **Task 1:** Model Compression (Pruning & Weight-Only Quantization on `Qwen/Qwen2.5-1.5B`).
2. **Task 2:** Efficient Attention & Context Memory (Exact Tiling, MQAR, KV-Cache Compression).
3. **Task 3:** Draft-Model Distillation for Speculative Decoding.

## Setup & Reproduction

```bash
pip install -r requirements.txt
git clone https://github.com/AbDu11aHHH/atml-pa4-datasets.git
cp configs/local_source_revisions.json atml-pa4-datasets/data/local_source_revisions.json
python atml-pa4-datasets/scripts/setup_data.py --public
python atml-pa4-datasets/scripts/check_setup.py
```

Run Task 1 unit tests and experiments (CUDA GPU recommended):
```bash
python -m unittest tests/test_task1.py -v
python scripts/run_task1.py --dataset-root ./atml-pa4-datasets --output-dir ./results/task1
python scripts/analyze_task1.py --results-dir ./results/task1 --dataset-root ./atml-pa4-datasets
```

Or run directly in Google Colab via `notebooks/colab_task1.ipynb`.

## Attribution
- Dataset manifests, MQAR generator, and setup scripts are from the course repository [`AbDu11aHHH/atml-pa4-datasets`](https://github.com/AbDu11aHHH/atml-pa4-datasets).
- Task 1 HellaSwag prompt preprocessing (`preprocess_hellaswag_text`, `_encode_pair`) follows the `lm-evaluation-harness` convention.
- Pruning/quantization sensitivity methods build on Wanda (Sun et al., 2024), AWQ (Lin et al., 2024), OWL (Yin et al., 2024), ShortGPT (Men et al., 2025), and HAWQ-V2 / Taylor sensitivity (Dong et al., 2020; Molchanov et al., 2019).

