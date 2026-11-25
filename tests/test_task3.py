"""Unit tests for Task 3 speculative sampling, Markov chi2, cache rollback, and cost model."""

from __future__ import annotations

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.task3.eval_task3 import check_short_compliance, extract_gsm8k_number
from src.task3.speculative import (
    compute_analytical_speculative_model,
    speculative_generate,
    target_only_generate,
    validate_markov_chi2,
)


def make_tiny_qwen2(seed: int = 6304) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        torch_dtype=torch.float32,
    )
    return Qwen2ForCausalLM(cfg).eval()


class Task3UnitTests(unittest.TestCase):
    @staticmethod
    def _crop_copy(cache, length: int):
        from transformers import DynamicCache

        c = DynamicCache()
        for lidx, layer in enumerate(cache.layers):
            c.update(layer.keys[:, :, :length, :].clone(), layer.values[:, :, :length, :].clone(), lidx)
        return c

    def test_markov_chi2_100k_and_negative_control(self):
        res = validate_markov_chi2(n_trials=100000, seed=6304, gamma=2)
        self.assertGreaterEqual(res["min_expected_count"], 10.0)
        self.assertTrue(
            res["correct_sampler"]["passed_p_ge_0_01"],
            f"Correct sampler rejected unexpectedly: p={res['correct_sampler']['pvalue']}",
        )
        self.assertTrue(
            res["broken_residual_negative_control"]["rejected_p_lt_0_01"],
            f"Broken residual negative control not rejected: p={res['broken_residual_negative_control']['pvalue']}",
        )

    def test_speculative_real_model_cache_consistency_and_edge_cases(self):
        target = make_tiny_qwen2(6304)
        draft = make_tiny_qwen2(6305)
        prompt = torch.randint(10, 100, (1, 12), dtype=torch.long)

        # 1. Identical draft == target -> 100% acceptance across multiple gammas & lengths
        for gamma in (1, 2, 4):
            out_self = speculative_generate(
                target, target, prompt, max_new_tokens=9, gamma=gamma, eos_token_id=None, seed=6304
            )
            self.assertEqual(out_self["n_generated"], 9)
            self.assertEqual(out_self["accepted_over_examined"], 1.0)

        # 2. Distinct draft + cache rollback/bonus alignment check against direct target forward
        out_mix = speculative_generate(
            target, draft, prompt, max_new_tokens=11, gamma=3, eos_token_id=None, seed=6304, return_final_caches=True
        )
        self.assertEqual(out_mix["n_generated"], 11)
        t_cache = out_mix["_t_cache"]
        l_cached = out_mix["_l_cached"]
        self.assertEqual(t_cache.layers[0].get_seq_length(), l_cached)

        # Compare next logits from cached target vs direct full-sequence forward
        full_ids = torch.cat([prompt, torch.tensor([out_mix["generated_ids"]], dtype=torch.long)], dim=1)
        with torch.no_grad():
            direct_logits = target(input_ids=full_ids, use_cache=False).logits[:, -1, :]
            if out_mix["_target_pending"] is not None:
                self.assertEqual(l_cached, full_ids.shape[1] - 1)
                cached_logits = target(
                    input_ids=out_mix["_target_pending"],
                    past_key_values=t_cache,
                    use_cache=True,
                ).logits[:, -1, :]
            else:
                self.assertEqual(l_cached, full_ids.shape[1])
                cached_logits = target(
                    input_ids=full_ids[:, -1:],
                    past_key_values=self._crop_copy(t_cache, full_ids.shape[1] - 1),
                    use_cache=True,
                ).logits[:, -1, :]
            self.assertLess((direct_logits - cached_logits).abs().max().item(), 1e-4)

        # 3. EOS stopping test (force EOS ID = 5 and check both stop cleanly <= max_new_tokens)
        out_eos = speculative_generate(
            target, draft, prompt, max_new_tokens=15, gamma=4, eos_token_id=5, seed=6304
        )
        self.assertLessEqual(out_eos["n_generated"], 15)
        if out_eos["terminated_by_eos"]:
            self.assertEqual(out_eos["generated_ids"][-1], 5)
            self.assertNotIn(5, out_eos["generated_ids"][:-1])

    def test_analytical_cost_model_and_extractors(self):
        ana = compute_analytical_speculative_model(alpha=0.7, c=0.2, v_gamma=1.1, gamma=4)
        self.assertIsNotNone(ana["break_even_alpha"])
        self.assertGreater(ana["predicted_speedup_S_hat"], 1.0)
        # Impossible break-even when 1 + gamma <= gamma*c + v_gamma
        imp = compute_analytical_speculative_model(alpha=0.9, c=1.0, v_gamma=2.0, gamma=4)
        self.assertIsNone(imp["break_even_alpha"])

        self.assertEqual(extract_gsm8k_number("Blah #### 1,234"), "1234")
        self.assertEqual(extract_gsm8k_number("Result is \\boxed{42}"), "42")
        self.assertTrue(check_short_compliance("February", 0))
        self.assertFalse(check_short_compliance("Month: February", 0))
        self.assertTrue(check_short_compliance("8441", 1))
        self.assertTrue(check_short_compliance("PK", 2))


if __name__ == "__main__":
    unittest.main()
