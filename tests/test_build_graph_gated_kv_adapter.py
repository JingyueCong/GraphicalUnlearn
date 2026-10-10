import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_graph_gated_kv_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("build_graph_gated_kv_adapter", SCRIPT_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class GraphGateBuilderTest(unittest.TestCase):
    def test_graph_smoothing_uses_neighbor_keys(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        smoothed = BUILDER.graph_smooth(keys, [(0, 1, 1.0)], alpha=1.0)

        torch.testing.assert_close(smoothed, torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    def test_threshold_respects_retain_tail(self):
        forget = torch.tensor([0.9, 0.8])
        retain = torch.tensor([0.1, 0.2, 0.3, 0.4])

        threshold, tpr, fpr = BUILDER.calibrate_threshold(
            forget, retain, target_retain_fpr=0.25
        )

        self.assertGreaterEqual(threshold, 0.3)
        self.assertEqual(tpr, 1.0)
        self.assertLessEqual(fpr, 0.5)


if __name__ == "__main__":
    unittest.main()
