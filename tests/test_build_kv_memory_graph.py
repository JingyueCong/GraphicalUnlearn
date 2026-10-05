import importlib.util
import math
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_kv_memory_graph.py"
SPEC = importlib.util.spec_from_file_location("build_kv_memory_graph", SCRIPT_PATH)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


class KVMemoryGraphTest(unittest.TestCase):
    def test_pool_answer_activations_ignores_prompt(self):
        activations = torch.tensor([[[9.0, 9.0], [2.0, 0.0], [0.0, 2.0]]])
        labels = torch.tensor([[-100, 1, 2]])

        pooled = GRAPH.pool_answer_activations(activations, labels)

        expected = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)]])
        torch.testing.assert_close(pooled, expected)


if __name__ == "__main__":
    unittest.main()
