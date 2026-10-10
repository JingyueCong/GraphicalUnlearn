import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "src" / "model" / "gated_kv.py"
SPEC = importlib.util.spec_from_file_location("gated_kv", SCRIPT_PATH)
GATED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATED)


class GraphGatedKVTest(unittest.TestCase):
    def test_matching_key_receives_larger_gate(self):
        hook = GATED.GraphGatedLowRankHook(
            factor_a=torch.tensor([[1.0, 0.0]]),
            factor_b=torch.tensor([[1.0], [1.0]]),
            prototypes=torch.tensor([[1.0, 0.0]]),
            threshold=0.5,
            temperature=0.1,
            device="cpu",
        )

        matching = hook.gate(torch.tensor([[1.0, 0.0]]))
        orthogonal = hook.gate(torch.tensor([[0.0, 1.0]]))

        self.assertGreater(float(matching), 0.99)
        self.assertLess(float(orthogonal), 0.01)

    def test_zero_gate_preserves_linear_output(self):
        linear = torch.nn.Linear(2, 2, bias=False)
        hook = GATED.GraphGatedLowRankHook(
            factor_a=torch.tensor([[1.0, 0.0]]),
            factor_b=torch.tensor([[1.0], [1.0]]),
            prototypes=torch.tensor([[1.0, 0.0]]),
            threshold=2.0,
            temperature=0.01,
            device="cpu",
        )
        inputs = torch.tensor([[1.0, 0.0]])
        base = linear(inputs)

        edited = hook(linear, (inputs,), base)

        torch.testing.assert_close(edited, base)


if __name__ == "__main__":
    unittest.main()
