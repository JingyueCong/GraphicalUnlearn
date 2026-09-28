import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from trainer.unlearn.graph_npo import GraphNPO


class GraphNPOTest(unittest.TestCase):
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

    def test_loads_wrapped_weight_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            path.write_text(
                json.dumps({"weights": {"0": 0.5, "3": 1.5}}),
                encoding="utf-8",
            )
            self.assertEqual(GraphNPO._load_graph_weights(path), {0: 0.5, 3: 1.5})

    def test_index_lookup_preserves_graph_weights(self):
        trainer = GraphNPO.__new__(GraphNPO)
        trainer.graph_weights = {0: 0.5, 1: 1.5}
        trainer.strict_index = True
        trainer.normalize_batch_weights = False

        weights = trainer._weights_for_indices(
            torch.tensor([1, 0]), device=torch.device("cpu"), dtype=torch.float32
        )
        torch.testing.assert_close(weights, torch.tensor([1.5, 0.5]))

    def test_missing_index_raises_in_strict_mode(self):
        trainer = GraphNPO.__new__(GraphNPO)
        trainer.graph_weights = {0: 1.0}
        trainer.strict_index = True
        trainer.normalize_batch_weights = False

        with self.assertRaises(KeyError):
            trainer._weights_for_indices(
                torch.tensor([1]), device=torch.device("cpu"), dtype=torch.float32
            )

    def test_weighted_objective_runs_backward(self):
        torch.manual_seed(0)
        model = self.TinyCausalLM()
        reference = self.TinyCausalLM()
        reference.load_state_dict(model.state_dict())
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

        trainer = GraphNPO.__new__(GraphNPO)
        trainer.beta = 0.1
        trainer.gamma = 1.0
        trainer.alpha = 1.0
        trainer.retain_loss_type = "NLL"
        trainer.ref_model = reference
        trainer.graph_weights = {0: 0.5, 1: 1.5}
        trainer.strict_index = True
        trainer.normalize_batch_weights = False

        tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        attention = torch.ones_like(tokens)
        nested_batch = {
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

        loss = trainer.compute_loss(model, nested_batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
