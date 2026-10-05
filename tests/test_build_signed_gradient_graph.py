import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_signed_gradient_graph.py"
SPEC = importlib.util.spec_from_file_location("build_signed_gradient_graph", SCRIPT_PATH)
SIGNED_GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIGNED_GRAPH)


class SignedGradientGraphTest(unittest.TestCase):
    def test_countsketch_is_deterministic_normalized_and_nonzero(self):
        parameters = [torch.zeros(3), torch.zeros(2, 2)]
        gradients = [torch.tensor([1.0, -2.0, 3.0]), torch.ones(2, 2)]
        maps = SIGNED_GRAPH.make_countsketch_maps(parameters, 8, seed=7)

        first = SIGNED_GRAPH.countsketch_gradient_tensors(gradients, maps, 8)
        second = SIGNED_GRAPH.countsketch_gradient_tensors(gradients, maps, 8)

        self.assertEqual(first.shape, (8,))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first.norm(), torch.tensor(1.0))

    def test_zero_countsketch_is_rejected(self):
        parameters = [torch.zeros(2)]
        maps = SIGNED_GRAPH.make_countsketch_maps(parameters, 4, seed=0)
        with self.assertRaisesRegex(ValueError, "zero gradient"):
            SIGNED_GRAPH.countsketch_gradient_tensors(
                [torch.zeros(2)], maps, projection_dim=4
            )

    def test_signed_knn_contains_cooperative_and_conflict_edges(self):
        vectors = [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]

        adjacency, edges = SIGNED_GRAPH.build_signed_knn_graph(
            vectors,
            positive_top_k=1,
            negative_top_k=1,
            min_abs_similarity=0.1,
        )

        self.assertTrue(any(edge["sign"] == 1 for edge in edges))
        self.assertTrue(any(edge["sign"] == -1 for edge in edges))
        self.assertTrue(all(edge["weight"] > 0 for edge in edges))
        self.assertTrue(all(adjacency))


if __name__ == "__main__":
    unittest.main()
