import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_iterative_kv_lora.py"
SPEC = importlib.util.spec_from_file_location("train_iterative_kv_lora", SCRIPT_PATH)
ITERATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ITERATIVE)


class IterativeKVLoRATest(unittest.TestCase):
    def test_forget_loss_decreases_when_factual_nll_increases(self):
        reference = torch.tensor([2.0])
        unchanged = ITERATIVE.bounded_forget_loss(reference, reference, beta=1.0)
        forgotten = ITERATIVE.bounded_forget_loss(
            torch.tensor([4.0]), reference, beta=1.0
        )

        self.assertLess(float(forgotten), float(unchanged))

    def test_topk_distillation_is_zero_for_identical_logits(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
        labels = torch.tensor([[-100, 2]])

        loss = ITERATIVE.topk_logit_distillation(logits, logits, labels, top_k=2)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_low_rank_hook_matches_merged_weight(self):
        linear = torch.nn.Linear(3, 2, bias=False)
        edit = ITERATIVE.LowRankOutputEdit(3, 2, rank=1, device="cpu", seed=0)
        with torch.no_grad():
            edit.a.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
            edit.b.copy_(torch.tensor([[2.0], [-1.0]]))
        inputs = torch.tensor([[1.0, -1.0, 2.0]])
        base = linear(inputs)
        handle = linear.register_forward_hook(edit)
        try:
            edited = linear(inputs)
        finally:
            handle.remove()

        expected = base + torch.nn.functional.linear(inputs, edit.merged_weight())
        torch.testing.assert_close(edited, expected)


if __name__ == "__main__":
    unittest.main()
