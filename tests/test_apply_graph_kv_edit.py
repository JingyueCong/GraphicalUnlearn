import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "apply_graph_kv_edit.py"
SPEC = importlib.util.spec_from_file_location("apply_graph_kv_edit", SCRIPT_PATH)
EDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EDIT)


class GraphKVEditTest(unittest.TestCase):
    def test_replace_answers_preserves_original_records(self):
        records = [{"question": "Q", "answer": "fact", "id": 7}]

        counterfactual = EDIT.replace_answers(records, "answer", "I don't know.")

        self.assertEqual(counterfactual[0]["answer"], "I don't know.")
        self.assertEqual(counterfactual[0]["id"], 7)
        self.assertEqual(records[0]["answer"], "fact")

    def test_prompt_tokens_include_requested_system_prompt(self):
        class Tokenizer:
            def __init__(self):
                self.chat = None

            def apply_chat_template(self, chat, **kwargs):
                self.chat = chat
                self.kwargs = kwargs
                return [1, 2, 3, 4]

        tokenizer = Tokenizer()
        args = SimpleNamespace(
            question_key="question", date_string="10 Apr 2025", max_length=3
        )

        token_ids = EDIT._prompt_token_ids(
            tokenizer,
            {"question": "Who is the author?"},
            args,
            "Do not disclose the answer.",
        )

        self.assertEqual(token_ids, [1, 2, 3])
        self.assertEqual(tokenizer.chat[0]["role"], "system")
        self.assertEqual(tokenizer.chat[0]["content"], "Do not disclose the answer.")
        self.assertEqual(tokenizer.chat[1]["content"], "Who is the author?")
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_graph_columns_are_weighted_differences(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        columns = EDIT.graph_difference_columns(keys, [(0, 1, 2.0)], gamma=2.0)

        torch.testing.assert_close(columns[:, 0], torch.tensor([2.0, -2.0]))

    def test_truncated_product_is_exact_at_full_rank(self):
        torch.manual_seed(0)
        left = torch.randn(5, 3)
        right = torch.randn(3, 7)

        product, _singular, explained = EDIT.truncated_product(left, right, rank=3)

        torch.testing.assert_close(product, left @ right, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(explained, 1.0, places=5)

    def test_solver_edits_forget_more_than_anchored_retain(self):
        forget_keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        forget_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        retain_keys = torch.tensor([[1.0, 1.0]])

        _delta, diagnostics = EDIT.solve_edit(
            forget_keys=forget_keys,
            desired_deltas=-forget_values,
            retain_keys=retain_keys,
            edges=[],
            strength=0.2,
            retain_weight=10.0,
            graph_gamma=0.0,
            ridge_scale=1e-3,
            rank=2,
            device="cpu",
        )

        self.assertLess(diagnostics["retain_to_forget_response_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
