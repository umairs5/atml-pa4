"""Comprehensive CPU unit tests for Task 1 compression, packing, accounting, and evaluation."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.common.bootstrap import paired_bootstrap_mean_diff, paired_bootstrap_ppl_diff
from src.task1.eval_hellaswag import encode_context_continuation, format_hellaswag_example
from src.task1.eval_ppl import evaluate_perplexity_windows
from src.task1.model_utils import (
    compute_pruned_model_storage,
    compute_unique_parameter_accounting,
    enumerate_eligible_projections,
)
from src.task1.packing import (
    calc_quantized_matrix_bytes,
    calc_sparse_matrix_bytes,
    pack_bits,
    unpack_bits,
)
from src.task1.prune import (
    allocate_marginal_distortion_sparsities,
    allocate_owl_sparsities,
    allocate_shortgpt_bi_sparsities,
    apply_pruning_to_model,
    collect_calibration_stats,
    collect_output_fisher_scales,
    make_2_4_mask,
    make_per_row_mask,
    measure_layer_output_errors,
)
from src.task1.quant import (
    allocate_mixed_precision_milp,
    apply_q1_uniform,
    apply_q2_awq,
    measure_quantization_distortion_matrix,
    quantize_weight_asymmetric,
)


def make_tiny_qwen2(tie_word_embeddings: bool = True) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
        torch_dtype=torch.float32,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model


class Task1UnitTests(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        gen = torch.Generator().manual_seed(6304)
        for bits in (2, 3, 4, 8):
            for n in (1, 7, 15, 128, 128 * 9 + 5):
                vals = torch.randint(0, 1 << bits, (n,), generator=gen, dtype=torch.uint8)
                packed = pack_bits(vals, bits)
                self.assertEqual(packed.dtype, torch.uint8)
                self.assertEqual(packed.numel(), math.ceil(n * bits / 8))
                unpacked = unpack_bits(packed, bits, n)
                self.assertTrue(torch.equal(vals, unpacked), f"Failed roundtrip for bits={bits}, n={n}")

    def test_quantize_matrix_and_packed_bytes(self):
        gen = torch.Generator().manual_seed(6304)
        w = torch.randn(32, 256, generator=gen, dtype=torch.float16) * 0.05
        for bits in (2, 3, 4, 8):
            w_hat, packed = quantize_weight_asymmetric(w, bits=bits, group_size=128)
            w_deq = packed.dequantize(dtype=torch.float16)
            self.assertTrue(torch.equal(w_hat, w_deq))
            spec = calc_quantized_matrix_bytes(32, 256, bits=bits, group_size=128, has_unfolded_channel_scale=False)
            self.assertEqual(packed.actual_bytes(), spec.total_bytes)

    def test_pruning_masks_and_2_4(self):
        scores = torch.tensor(
            [
                [1.0, 4.0, 2.0, 3.0, 10.0, 0.5, 8.0, 2.0],
                [9.0, 1.0, 2.0, 8.0, 0.1, 0.2, 5.0, 6.0],
            ]
        )
        m50 = make_per_row_mask(scores, 4)
        self.assertTrue(torch.equal(m50.sum(dim=1), torch.tensor([4, 4])))
        m24 = make_2_4_mask(scores)
        # Every contiguous group of 4 must keep exactly 2
        g = m24.reshape(2, 2, 4).sum(dim=-1)
        self.assertTrue(torch.equal(g, torch.full((2, 2), 2)))

    def test_tied_weights_and_sparse_accounting(self):
        model_tied = make_tiny_qwen2(tie_word_embeddings=True)
        model_untied = make_tiny_qwen2(tie_word_embeddings=False)
        projs = enumerate_eligible_projections(model_tied)
        self.assertEqual(len(projs), 14)

        acct_tied = compute_unique_parameter_accounting(model_tied)
        acct_untied = compute_unique_parameter_accounting(model_untied)
        self.assertEqual(acct_tied["tied_duplicate_tensors"], 1)
        self.assertEqual(acct_untied["tied_duplicate_tensors"], 0)
        self.assertEqual(
            acct_untied["total_unique_params"] - acct_tied["total_unique_params"],
            256 * 128,
        )

        # Apply 50% magnitude pruning and verify eligible sparsity and sparse bytes
        apply_pruning_to_model(model_tied, method="magnitude", sparsity_map=0.50)
        st = compute_pruned_model_storage(model_tied)
        self.assertAlmostEqual(st["achieved_eligible_sparsity"], 0.50, places=6)
        m_b, v_b, _ = calc_sparse_matrix_bytes(128, 128, 128 * 64)
        self.assertEqual(m_b, 128 * 128 // 8)
        self.assertEqual(v_b, 2 * 128 * 64)

    def test_p3_allocators_and_q3_milp_on_tiny_qwen2(self):
        model = make_tiny_qwen2(tie_word_embeddings=True)
        projs = enumerate_eligible_projections(model)
        seqs = [torch.randint(0, 256, (16,)).tolist() for _ in range(4)]

        stats = collect_calibration_stats(model, seqs, device="cpu", batch_size=2)
        fisher = collect_output_fisher_scales(model, seqs[:2], device="cpu")

        for target_s in (0.50, 0.75):
            s_owl, _ = allocate_owl_sparsities(projs, stats["input_l2"], target_s)
            s_bi, _ = allocate_shortgpt_bi_sparsities(projs, stats["block_bi"], target_s)
            s_marg, _ = allocate_marginal_distortion_sparsities(
                projs, stats["input_l2"], fisher["output_grad_msq"], target_s
            )
            for s_map in (s_owl, s_bi, s_marg):
                for v in s_map.values():
                    self.assertGreaterEqual(v, 0.10 - 1e-6)
                    self.assertLessEqual(v, 0.85 + 1e-6)
                tot_zeros = sum(int(round(s_map[p.name] * p.d_in)) * p.d_out for p in projs)
                tot_elig = sum(p.numel for p in projs)
                self.assertAlmostEqual(tot_zeros / tot_elig, target_s, delta=0.005)

        # Test Q1, Q2, Q3 end-to-end
        m_q1 = make_tiny_qwen2(tie_word_embeddings=True)
        m_q1.load_state_dict(model.state_dict())
        q1_sum = apply_q1_uniform(m_q1, bits=4, group_size=128)

        m_q2 = make_tiny_qwen2(tie_word_embeddings=True)
        m_q2.load_state_dict(model.state_dict())
        q2_sum = apply_q2_awq(
            m_q2, seqs, bits_map=4, device="cpu", group_size=128, alpha_grid=(0.0, 0.5, 1.0), batch_size=2
        )
        self.assertGreater(q2_sum["channel_scale_bytes"], 0)

        acct = compute_unique_parameter_accounting(model)
        dist = measure_quantization_distortion_matrix(model, seqs, device="cpu", group_size=128, batch_size=2)
        bmap, milp_info = allocate_mixed_precision_milp(
            projs,
            dist,
            fisher["output_grad_msq"],
            packed_byte_budget=q2_sum["total_packed_bytes"],
            uncompressed_fp16_bytes=acct["uncompressed_fp16_bytes"],
            group_size=128,
        )
        self.assertLessEqual(milp_info["allocated_total_bytes"], q2_sum["total_packed_bytes"])
        self.assertTrue(all(b in (2, 3, 4, 8) for b in bmap.values()))

        # Check error diagnostics run cleanly
        errs = measure_layer_output_errors(model, m_q2, seqs, device="cpu", batch_size=2)
        self.assertGreaterEqual(errs["final_block_cum_rel_l2"], 0.0)

    def test_ppl_and_bootstrap(self):
        model = make_tiny_qwen2(tie_word_embeddings=True)
        windows = [
            {"token_ids": [1, 2, 3, 4, 5], "score_target_offsets": [1, 5], "scored_tokens": 4},
            {"token_ids": [6, 7, 8], "score_target_offsets": [1, 3], "scored_tokens": 2},
        ]
        res = evaluate_perplexity_windows(model, windows, device="cpu", batch_size=2, expected_total_tokens=6)
        self.assertEqual(res["scored_tokens"], 6)
        self.assertAlmostEqual(res["perplexity"], math.exp(res["mean_nll"]), places=6)

        boot_ppl = paired_bootstrap_ppl_diff([10.0, 5.0, 8.0], [11.0, 5.5, 8.2], [4, 2, 3], n_resamples=2000)
        self.assertLess(boot_ppl["ppl_diff_ci_low"], boot_ppl["ppl_diff_ci_high"])
        boot_acc = paired_bootstrap_mean_diff([1, 0, 1, 1], [0, 0, 1, 0], n_resamples=2000)
        self.assertAlmostEqual(boot_acc["diff"], 0.5)


if __name__ == "__main__":
    unittest.main()

