import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from trainer.unlearn.signed_gradient_conflict_npo import SignedGradientConflictNPO


class SignedGradientConflictNPOTest(unittest.TestCase):
    @staticmethod
    def make_bare_trainer():
        trainer = SignedGradientConflictNPO.__new__(SignedGradientConflictNPO)
        trainer.num_nodes = 3
        trainer.positive_adjacency = {0: [(2, 1.0)], 1: [], 2: [(0, 1.0)]}
        trainer.negative_adjacency = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: []}
        trainer.residual_ema = torch.tensor([3.0, 1.0, 1.0])
        trainer.cooperative_strength = 0.0
        trainer.conflict_strength = 0.7
        trainer.conflict_temperature = 0.5
        trainer.communities = {0: 0, 1: 0, 2: 0}
        trainer.community_sizes = Counter({0: 3})
        trainer.mean_community_size = 3.0
        trainer.community_balance_power = 0.0
        trainer.normalize_global_weights = True
        trainer.weight_floor = 0.1
        trainer.weight_ceiling = 3.0
        return trainer

    def test_conflict_allocation_prioritizes_larger_residual(self):
        trainer = self.make_bare_trainer()

        weights = trainer._refresh_global_node_weights()

        self.assertGreater(float(weights[0]), float(weights[1]))
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
        self.assertGreater(trainer._mean_abs_conflict_contrast, 0.0)

    def test_cooperative_edge_smooths_log_residual(self):
        trainer = self.make_bare_trainer()
        trainer.negative_adjacency = {0: [], 1: [], 2: []}
        trainer.cooperative_strength = 0.5

        weights = trainer._refresh_global_node_weights()

        self.assertAlmostEqual(float(weights[0]), float(weights[2]), places=5)

    def test_loads_signed_adjacency(self):
        payload = {
            "num_nodes": 3,
            "edges": [
                {"source": 0, "target": 1, "weight": 0.8, "sign": -1},
                {"source": 1, "target": 2, "weight": 0.6, "sign": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            graph = SignedGradientConflictNPO._load_signed_adjacency(path, 3)

        self.assertEqual(graph["num_positive_edges"], 1)
        self.assertEqual(graph["num_negative_edges"], 1)
        self.assertEqual(graph["negative_adjacency"][0], [(1, 0.8)])


if __name__ == "__main__":
    unittest.main()
