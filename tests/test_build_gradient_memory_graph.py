import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_gradient_memory_graph.py"
SPEC = importlib.util.spec_from_file_location("build_gradient_memory_graph", SCRIPT_PATH)
GRADIENT_GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRADIENT_GRAPH)


class GradientMemoryGraphTest(unittest.TestCase):
    def test_projection_is_deterministic_normalized_and_has_expected_shape(self):
        gradients = [torch.tensor([1.0, 2.0]), torch.tensor([[3.0, 4.0]])]
        generator = torch.Generator().manual_seed(7)
        projection = torch.randn(4, 3, generator=generator)

        first = GRADIENT_GRAPH.project_gradient_tensors(gradients, projection)
        second = GRADIENT_GRAPH.project_gradient_tensors(gradients, projection)

        self.assertEqual(first.shape, (3,))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.norm(), torch.tensor(1.0))

    def test_zero_gradient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero gradient"):
            GRADIENT_GRAPH.project_gradient_tensors(
                [torch.zeros(2)], torch.ones(2, 3)
            )


if __name__ == "__main__":
    unittest.main()
