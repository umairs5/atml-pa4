"""Unit tests for Task 2 Parts A, B, and C (executed on Colab before running full suite)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.task2.part_a_attention import (
    dense_causal_attention,
    tiled_online_softmax_causal_attention,
    verify_causality_and_edge_cases,
)
from src.task2.part_b_mqar import (
    MQARTransformer,
    build_sparse_head_masks,
    chunkwise_linear_elu1_attention,
    theoretical_cost_at_T256,
)
from src.task2.part_c_kv import (
    calc_kv_bytes_formula,
    compress_kv_tensor_pair,
    decompress_kv_tensor_pair,
    extract_five_digit_answer,
    solve_eviction_kept_count,
    solve_hybrid_fp16_int2_counts,
)


def naive_sequential_linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, H, L, d_h = q.shape
    phi_q = F.elu(q.float()) + 1.0
    phi_k = F.elu(k.float()) + 1.0
    v_f = v.float()
    S = torch.zeros((B, H, d_h, d_h), dtype=torch.float32, device=q.device)
    z = torch.zeros((B, H, d_h), dtype=torch.float32, device=q.device)
    outs = []
    for t in range(L):
        kt = phi_k[:, :, t, :]
        vt = v_f[:, :, t, :]
        qt = phi_q[:, :, t, :]
        S = S + torch.einsum("bhd,bhe->bhde", kt, vt)
        z = z + kt
        num = torch.einsum("bhd,bhde->bhe", qt, S)
        den = torch.einsum("bhd,bhd->bh", qt, z).unsqueeze(-1) + eps
        outs.append(num / den)
    return torch.stack(outs, dim=-2).to(q.dtype)


class Task2UnitTests(unittest.TestCase):
    def test_part_a_causality_and_tiled_equivalence(self):
        res = verify_causality_and_edge_cases("cpu")
        self.assertEqual(res["causality_prefix_max_abs_diff"], 0.0)
        self.assertTrue(res["extreme_score_finite"])
        self.assertLess(res["extreme_score_dense_vs_tiled_max_abs"], 1e-5)
        self.assertEqual(res["all_masked_row0_max_abs"], 0.0)

        gen = torch.Generator().manual_seed(6304)
        q = torch.randn(2, 4, 150, 64, generator=gen, requires_grad=True)
        k = torch.randn(2, 4, 150, 64, generator=gen, requires_grad=True)
        v = torch.randn(2, 4, 150, 64, generator=gen, requires_grad=True)

        out_d = dense_causal_attention(q, k, v)
        out_t = tiled_online_softmax_causal_attention(q, k, v, block_q=32, block_kv=64)
        self.assertLess((out_d - out_t).abs().max().item(), 1e-5)

        # Compare gradients
        g = torch.randn_like(out_d)
        dq_d, dk_d, dv_d = torch.autograd.grad((out_d * g).sum(), (q, k, v))
        dq_t, dk_t, dv_t = torch.autograd.grad((out_t * g).sum(), (q, k, v))
        self.assertLess((dq_d - dq_t).abs().max().item(), 1e-4)
        self.assertLess((dk_d - dk_t).abs().max().item(), 1e-4)
        self.assertLess((dv_d - dv_t).abs().max().item(), 1e-4)

    def test_part_b_linear_equivalence_and_masks(self):
        gen = torch.Generator().manual_seed(6304)
        q = torch.randn(2, 4, 140, 32, generator=gen)
        k = torch.randn(2, 4, 140, 32, generator=gen)
        v = torch.randn(2, 4, 140, 32, generator=gen)

        out_chunk = chunkwise_linear_elu1_attention(q, k, v, chunk_size=64)
        out_seq = naive_sequential_linear_attention(q, k, v)
        self.assertLess((out_chunk - out_seq).abs().max().item(), 1e-5)

        m_ld = build_sparse_head_masks(256, "local_dilated", torch.device("cpu"))[0]
        # Head 0 (local W=64) at pos 200 sees 64 positions; head 2 (dilated) sees {0,4,...,200} = 51 positions
        self.assertEqual(int(m_ld[0, 200].sum().item()), 64)
        self.assertEqual(int(m_ld[2, 200].sum().item()), 51)
        self.assertEqual(int(m_ld[2, 255].sum().item()), 64)

        # Verify parameter count parity across B1/B2/B3
        p_dense = sum(p.numel() for p in MQARTransformer(mechanism="dense").parameters())
        p_topk = sum(p.numel() for p in MQARTransformer(mechanism="topk_64").parameters())
        p_lin = sum(p.numel() for p in MQARTransformer(mechanism="linear_elu1").parameters())
        self.assertEqual(p_dense, p_topk)
        self.assertEqual(p_dense, p_lin)

        th = theoretical_cost_at_T256()
        self.assertEqual(th["topk_64"]["attended_pairs_per_layer"], th["sliding_w64"]["attended_pairs_per_layer"])

    def test_part_c_kv_formulas_packing_and_budgets(self):
        for L in (2048, 2055, 4096, 4111):
            for bits in (2, 4):
                spec = calc_kv_bytes_formula(L, bits=bits, n_layers=2, n_kv_heads=2, head_dim=128)
                k = torch.randn(1, 2, L, 128, dtype=torch.float16)
                v = torch.randn(1, 2, L, 128, dtype=torch.float16)
                for layout in ("kivi", "tok"):
                    qkv = compress_kv_tensor_pair(k, v, bits=bits, layout=layout)
                    self.assertEqual(qkv.packed_bytes() * 2, spec["M_b_bytes"])
                    kh, vh = decompress_kv_tensor_pair(qkv)
                    self.assertEqual(kh.shape, k.shape)
                    self.assertEqual(vh.shape, v.shape)
                    # Trailing L % 64 tokens must be exact FP16 lossless
                    n_comp = (L // 64) * 64
                    self.assertTrue(torch.equal(kh[:, :, n_comp:, :], k[:, :, n_comp:, :]))
                    self.assertTrue(torch.equal(vh[:, :, n_comp:, :], v[:, :, n_comp:, :]))

        # Budget checks
        L_ctx = 2055
        b4 = calc_kv_bytes_formula(L_ctx, bits=4)["M_b_bytes"]
        k_ev, ev_bytes = solve_eviction_kept_count(L_ctx, b4, per_head_indexed=True)
        self.assertLessEqual(ev_bytes, b4)
        n_fp16, n_int2, hyb_bytes = solve_hybrid_fp16_int2_counts(L_ctx, b4)
        self.assertEqual(n_fp16 + n_int2, L_ctx)
        self.assertEqual(n_int2 % 64, 0)
        self.assertLessEqual(hyb_bytes, b4)

        self.assertEqual(extract_five_digit_answer("26836"), "26836")
        self.assertEqual(extract_five_digit_answer("Code is 04192."), "04192")
        self.assertIsNone(extract_five_digit_answer("123456"))

    def test_part_c_split_prefill_causality_on_tiny_qwen2(self):
        cfg = Qwen2Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            torch_dtype=torch.float32,
        )
        m = Qwen2ForCausalLM(cfg).eval()
        ids = torch.randint(0, 256, (1, 100))
        with torch.no_grad():
            full_logits = m(input_ids=ids, use_cache=False).logits[:, 80:, :]
            ctx_out = m(input_ids=ids[:, :80], use_cache=True)
            pos_ids = torch.arange(80, 100).unsqueeze(0)
            c_pos = torch.arange(80, 100)
            split_logits = m(
                input_ids=ids[:, 80:],
                past_key_values=ctx_out.past_key_values,
                position_ids=pos_ids,
                cache_position=c_pos,
                use_cache=True,
            ).logits
            self.assertLess((full_logits - split_logits).abs().max().item(), 1e-4)

            # Verify evicted suffix causality: perturbing future suffix token 10 must NOT change suffix token 5!
            ev_cache = ctx_out.past_key_values
            for layer in ev_cache.layers:
                layer.keys = layer.keys[:, :, :30, :].clone()
                layer.values = layer.values[:, :, :30, :].clone()
            ids_suf_a = ids[:, 80:].clone()
            ids_suf_b = ids[:, 80:].clone()
            ids_suf_b[0, 10] = (ids_suf_b[0, 10] + 1) % 256

            from copy import deepcopy
            c_a = deepcopy(ev_cache)
            c_b = deepcopy(ev_cache)
            la = m(input_ids=ids_suf_a, past_key_values=c_a, position_ids=pos_ids, cache_position=torch.arange(30, 50), use_cache=True).logits
            lb = m(input_ids=ids_suf_b, past_key_values=c_b, position_ids=pos_ids, cache_position=torch.arange(30, 50), use_cache=True).logits
            self.assertEqual((la[:, :6, :] - lb[:, :6, :]).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
