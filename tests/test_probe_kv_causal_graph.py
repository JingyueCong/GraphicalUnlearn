import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "probe_kv_causal_graph.py"
SPEC = importlib.util.spec_from_file_location("probe_kv_causal_graph", SCRIPT_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class KVCausalGraphProbeTest(unittest.TestCase):
    def test_rank_one_hook_erases_source_value(self):
        key = torch.tensor([1.0, 2.0])
        value = torch.tensor([3.0, -1.0])
        hook = PROBE.RankOneValueEraser(key, value, strength=0.25, ridge=1e-8)
        activations = key.reshape(1, 1, -1)
        output = value.reshape(1, 1, -1)

        edited = hook(None, (activations,), output)

        torch.testing.assert_close(edited, output * 0.75)

    def test_predictor_metrics_reward_correct_ranking(self):
        scores = [0.1, 0.9, 0.4, 0.2]
        effects = [0.2, 2.0, 0.8, 0.3]

        self.assertAlmostEqual(PROBE.spearman(scores, effects), 1.0)
        self.assertEqual(PROBE.precision_at_k(scores, effects, 2), 1.0)

    def test_selection_includes_sources(self):
        selected = PROBE.select_probe_indices(10, 5, seed=3, required=[8, 2])

        self.assertEqual(len(selected), 5)
        self.assertIn(2, selected)
        self.assertIn(8, selected)


if __name__ == "__main__":
    unittest.main()
