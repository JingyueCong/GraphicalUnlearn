import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from trainer.unlearn.graph_coverage_npo import GraphCoverageNPO


class GraphCoverageNPOTest(unittest.TestCase):
    class TinyCausalLM(torch.nn.Module):
        def __init__(self, vocab_size=7, hidden_size=5):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
            self.projection = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask=None, labels=None):
            logits = self.projection(self.embedding(input_ids))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(logits=logits, loss=loss)

    @staticmethod
    def make_bare_trainer():
        trainer = GraphCoverageNPO.__new__(GraphCoverageNPO)
        trainer.num_nodes = 3
        trainer.adjacency = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: []}
        trainer.communities = {0: 0, 1: 0, 2: 1}
        trainer.community_sizes = Counter({0: 2, 1: 1})
        trainer.mean_community_size = 1.5
        trainer.residual_ema = torch.ones(3)
        trainer.residual_temperature = 1.0
        trainer.residual_ema_decay = 0.0
        trainer.propagation_strength = 0.0
        trainer.propagation_interval = 1
        trainer.community_balance_power = 0.0
        trainer.weight_floor = 0.1
        trainer.weight_ceiling = 3.0
        trainer.normalize_global_weights = True
        trainer.global_node_weights = torch.ones(3)
        trainer.strict_index = True
        trainer._last_batch_weights = None
        trainer.diagnostic_interval = 0
        trainer._coverage_updates = 0
        trainer._optimizer_updates = 0
        trainer._retain_violation_sum = 0.0
        trainer._retain_violation_count = 0
        trainer._microbatches_in_step = 0
        trainer.alpha_update_rule = "additive"
        trainer.retain_violation_ema_decay = 0.0
        trainer.retain_violation_clip = 1.0
        trainer.retain_violation_deadband = 0.0
        trainer._retain_violation_ema = 0.0
        trainer._retain_violation_ema_initialized = False
        trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
        return trainer

    def test_loads_edges_and_communities(self):
        payload = {
            "num_nodes": 3,
            "edges": [{"source": 0, "target": 1, "weight": 0.75}],
            "communities": {"0": 0, "1": 0, "2": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            graph = GraphCoverageNPO._load_graph(path)

        self.assertEqual(graph["num_nodes"], 3)
        self.assertEqual(graph["num_edges"], 1)
        self.assertEqual(graph["adjacency"][0], [(1, 0.75)])
        self.assertEqual(graph["communities"], {0: 0, 1: 0, 2: 1})

    def test_dynamic_weights_focus_on_underforgotten_examples(self):
        trainer = self.make_bare_trainer()
        trainer._update_residuals(
            indices=torch.tensor([0, 1]),
            per_token_margins=torch.tensor([0.0, 2.0]),
        )
        trainer._refresh_global_node_weights()
        weights = trainer._dynamic_weights_for_batch(
            indices=torch.tensor([0, 1]),
            per_token_margins=torch.tensor([0.0, 2.0]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertGreater(float(weights[0]), float(weights[1]))
        torch.testing.assert_close(
            trainer.global_node_weights.mean(), torch.tensor(1.0)
        )
        self.assertNotAlmostEqual(float(weights.mean()), 1.0)

    def test_graph_propagates_neighbor_residual(self):
        trainer = self.make_bare_trainer()
        trainer.propagation_strength = 1.0
        trainer.residual_ema = torch.tensor([1.0, 3.0, 1.0])
        trainer._refresh_global_node_weights()
        weights = trainer._dynamic_weights_for_batch(
            indices=torch.tensor([0, 2]),
            per_token_margins=torch.tensor([0.0, 0.0]),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_adaptive_alpha_updates_once_after_gradient_accumulation(self):
        trainer = GraphCoverageNPO.__new__(GraphCoverageNPO)
        trainer.adaptive_retain = True
        trainer.retain_budget = 0.05
        trainer.adaptive_alpha_lr = 0.5
        trainer.current_alpha = 1.0
        trainer.alpha_min = 0.0
        trainer.alpha_max = 4.0
        trainer.alpha_update_rule = "additive"
        trainer.retain_violation_ema_decay = 0.0
        trainer.retain_violation_clip = 1.0
        trainer.retain_violation_deadband = 0.0
        trainer._last_retain_violation = 0.0
        trainer._retain_violation_ema = 0.0
        trainer._retain_violation_ema_initialized = False
        trainer._optimizer_updates = 0
        trainer._retain_violation_sum = 0.0
        trainer._retain_violation_count = 0
        trainer._microbatches_in_step = 0
        trainer.propagation_interval = 10
        trainer.global_node_weights = torch.ones(1)
        trainer.args = SimpleNamespace(gradient_accumulation_steps=4)

        for _ in range(3):
            updated = trainer._record_retain_violation(
                torch.tensor(2.0), torch.tensor(1.0)
            )
            self.assertFalse(updated)
            self.assertEqual(trainer.current_alpha, 1.0)

        updated = trainer._record_retain_violation(
            torch.tensor(2.0), torch.tensor(1.0)
        )
        self.assertTrue(updated)
        self.assertGreater(trainer.current_alpha, 1.0)
        self.assertAlmostEqual(trainer._last_retain_violation, 0.95, places=6)
        self.assertEqual(trainer._optimizer_updates, 1)
        self.assertEqual(trainer._retain_violation_count, 0)

    def test_additive_controller_can_lower_alpha_inside_budget(self):
        trainer = self.make_bare_trainer()
        trainer.adaptive_retain = True
        trainer.adaptive_alpha_lr = 0.2
        trainer.current_alpha = 1.0
        trainer.alpha_min = 0.0
        trainer.alpha_max = 2.0

        updated = trainer._apply_alpha_update(-0.5)

        self.assertAlmostEqual(updated, 0.9)
        self.assertAlmostEqual(trainer._retain_violation_ema, -0.5)

    def test_violation_ema_damps_opposite_consecutive_updates(self):
        trainer = self.make_bare_trainer()
        trainer.adaptive_retain = True
        trainer.adaptive_alpha_lr = 0.1
        trainer.retain_violation_ema_decay = 0.5
        trainer.current_alpha = 1.0
        trainer.alpha_min = 0.0
        trainer.alpha_max = 2.0

        trainer._apply_alpha_update(0.5)
        after_positive = trainer.current_alpha
        trainer._apply_alpha_update(-0.5)

        self.assertAlmostEqual(after_positive, 1.05)
        self.assertAlmostEqual(trainer._retain_violation_ema, 0.0)
        self.assertAlmostEqual(trainer.current_alpha, after_positive)

    def test_global_weights_refresh_without_adaptive_retain(self):
        trainer = self.make_bare_trainer()
        trainer.adaptive_retain = False
        trainer.residual_ema = torch.tensor([2.0, 1.0, 1.0])

        updated = trainer._record_retain_violation(torch.tensor(1.0))

        self.assertTrue(updated)
        self.assertEqual(trainer._optimizer_updates, 1)
        self.assertGreater(
            float(trainer.global_node_weights[0]),
            float(trainer.global_node_weights[1]),
        )
        torch.testing.assert_close(
            trainer.global_node_weights.mean(), torch.tensor(1.0)
        )

    def test_objective_runs_backward(self):
        torch.manual_seed(0)
        model = self.TinyCausalLM()
        reference = self.TinyCausalLM()
        reference.load_state_dict(model.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

        trainer = self.make_bare_trainer()
        trainer.beta = 0.1
        trainer.gamma = 1.0
        trainer.alpha = 1.0
        trainer.base_alpha = 1.0
        trainer.current_alpha = 1.0
        trainer.alpha_min = 0.0
        trainer.alpha_max = 4.0
        trainer.adaptive_retain = True
        trainer.retain_budget = 0.05
        trainer.adaptive_alpha_lr = 0.05
        trainer.alpha_update_rule = "additive"
        trainer.retain_violation_ema_decay = 0.9
        trainer.retain_violation_clip = 1.0
        trainer.retain_violation_deadband = 0.0
        trainer._last_retain_violation = 0.0
        trainer._retain_violation_ema = 0.0
        trainer._retain_violation_ema_initialized = False
        trainer._optimizer_updates = 0
        trainer._retain_violation_sum = 0.0
        trainer._retain_violation_count = 0
        trainer._microbatches_in_step = 0
        trainer.retain_loss_type = "NLL"
        trainer.ref_model = reference

        tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        attention = torch.ones_like(tokens)
        batch = {
            "forget": {
                "input_ids": tokens,
                "attention_mask": attention,
                "labels": tokens.clone(),
                "index": torch.tensor([0, 1]),
            },
            "retain": {
                "input_ids": tokens.flip(0),
                "attention_mask": attention,
                "labels": tokens.flip(0),
                "index": torch.tensor([1, 0]),
            },
        }

        loss = trainer.compute_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
