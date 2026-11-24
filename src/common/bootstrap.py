"""Paired bootstrap resampling helpers across Tasks 1, 2, and 3."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_bootstrap_mean_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 6304,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap CI for mean(A) - mean(B) over matched per-example metrics."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"Expected matching 1D arrays, got {a.shape} vs {b.shape}")
    n = len(a)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_diffs, alpha / 2.0))
    hi = float(np.quantile(boot_diffs, 1.0 - alpha / 2.0))
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(diff.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "n_resamples": int(n_resamples),
    }


def paired_bootstrap_ppl_diff(
    nll_sums_a: Sequence[float],
    nll_sums_b: Sequence[float],
    token_counts: Sequence[int],
    n_resamples: int = 2000,
    seed: int = 6304,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap CI for PPL(A) - PPL(B) by resampling whole evaluation windows
    and recomputing pooled token NLL before exponentiating.
    """
    sa = np.asarray(nll_sums_a, dtype=np.float64)
    sb = np.asarray(nll_sums_b, dtype=np.float64)
    tok = np.asarray(token_counts, dtype=np.float64)
    if not (sa.shape == sb.shape == tok.shape and sa.ndim == 1):
        raise ValueError("Input arrays must be matching 1D sequences")
    n = len(sa)
    total_tok = float(tok.sum())
    nll_a = float(sa.sum() / total_tok)
    nll_b = float(sb.sum() / total_tok)
    ppl_a = float(np.exp(nll_a))
    ppl_b = float(np.exp(nll_b))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_tok = tok[idx].sum(axis=1)
    boot_nll_a = sa[idx].sum(axis=1) / boot_tok
    boot_nll_b = sb[idx].sum(axis=1) / boot_tok
    boot_ppl_diff = np.exp(boot_nll_a) - np.exp(boot_nll_b)
    boot_nll_diff = boot_nll_a - boot_nll_b

    return {
        "nll_a": nll_a,
        "nll_b": nll_b,
        "nll_diff": nll_a - nll_b,
        "nll_diff_ci_low": float(np.quantile(boot_nll_diff, alpha / 2.0)),
        "nll_diff_ci_high": float(np.quantile(boot_nll_diff, 1.0 - alpha / 2.0)),
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "ppl_diff": ppl_a - ppl_b,
        "ppl_diff_ci_low": float(np.quantile(boot_ppl_diff, alpha / 2.0)),
        "ppl_diff_ci_high": float(np.quantile(boot_ppl_diff, 1.0 - alpha / 2.0)),
        "n_windows": int(n),
        "total_tokens": int(total_tok),
        "n_resamples": int(n_resamples),
    }

