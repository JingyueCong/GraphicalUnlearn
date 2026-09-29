import importlib.util
import unittest
from pathlib import Path

import networkx as nx


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "randomize_forget_graph.py"
SPEC = importlib.util.spec_from_file_location("randomize_forget_graph", SCRIPT_PATH)
RANDOMIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANDOMIZE)


class RandomizeForgetGraphTest(unittest.TestCase):
    def test_preserves_degrees_weights_and_connectivity(self):
        graph = nx.cycle_graph(10)
        graph.add_edges_from((node, node + 5) for node in range(5))
        edges = [
            {"source": left, "target": right, "weight": 0.1 + index}
            for index, (left, right) in enumerate(sorted(graph.edges))
        ]
        payload = {
            "version": 2,
            "method": "test_graph",
            "num_nodes": 10,
            "num_edges": len(edges),
            "weights": {str(node): 1.0 for node in range(10)},
            "edges": edges,
        }

        result = RANDOMIZE.randomize_graph(
            payload, seed=3, swaps_per_edge=2
        )
        randomized = nx.Graph()
        randomized.add_nodes_from(range(10))
        randomized.add_edges_from(
            (edge["source"], edge["target"]) for edge in result["edges"]
        )

        self.assertEqual(
            sorted(dict(graph.degree()).values()),
            sorted(dict(randomized.degree()).values()),
        )
        self.assertEqual(
            sorted(edge["weight"] for edge in edges),
            sorted(edge["weight"] for edge in result["edges"]),
        )
        self.assertTrue(nx.is_connected(randomized))
        self.assertEqual(result["num_edges"], len(edges))


if __name__ == "__main__":
    unittest.main()
