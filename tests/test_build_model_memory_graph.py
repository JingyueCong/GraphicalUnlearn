import importlib.util
import math
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_model_memory_graph.py"
SPEC = importlib.util.spec_from_file_location("build_model_memory_graph", SCRIPT_PATH)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


class ModelMemoryGraphTest(unittest.TestCase):
    def test_mutual_knn_backfills_isolated_nodes(self):
        vectors = [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
        ]
        adjacency, edges = GRAPH.build_mutual_knn_graph(vectors, top_k=1)

        pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertIn((0, 1), pairs)
        self.assertTrue(all(adjacency))
        self.assertTrue(all(edge["weight"] > 0 for edge in edges))
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            for neighbor, _ in adjacency[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        self.assertEqual(reached, {0, 1, 2})

    def test_answer_mean_pool_ignores_prompt_and_normalizes(self):
        hidden = torch.tensor(
            [[[10.0, 10.0], [2.0, 0.0], [0.0, 2.0]]]
        )
        labels = torch.tensor([[-100, 4, 5]])

        pooled = GRAPH.answer_mean_pool(hidden, labels)

        expected = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)]])
        torch.testing.assert_close(pooled, expected)


if __name__ == "__main__":
    unittest.main()
