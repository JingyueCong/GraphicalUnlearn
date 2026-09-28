import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll


logger = logging.getLogger(__name__)


class GraphNPO(GradDiff):
    """NPO with graph-derived, per-example forgetting strengths.

    The graph is built before training. Its output is a JSON file containing a
    mapping from the original forget-dataset index to a positive weight.  A
    mean weight of one keeps GraphNPO's loss scale comparable to vanilla NPO.
    """

    def __init__(
        self,
        graph_weights_path,
        beta=0.1,
        strict_index=True,
        normalize_batch_weights=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.strict_index = strict_index
        self.normalize_batch_weights = normalize_batch_weights
        self.graph_weights_path = Path(to_absolute_path(graph_weights_path))
        self.graph_weights = self._load_graph_weights(self.graph_weights_path)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        values = torch.tensor(list(self.graph_weights.values()), dtype=torch.float32)
        logger.info(
            "Loaded %d graph weights from %s (min=%.4f, mean=%.4f, max=%.4f)",
            len(values),
            self.graph_weights_path,
            values.min().item(),
            values.mean().item(),
            values.max().item(),
        )

    @staticmethod
    def _load_graph_weights(path):
        if not path.is_file():
            raise FileNotFoundError(f"Graph weight file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_weights = payload.get("weights", payload)
        if not isinstance(raw_weights, dict) or not raw_weights:
            raise ValueError("Graph weight file must contain a non-empty 'weights' mapping")

        weights = {int(index): float(weight) for index, weight in raw_weights.items()}
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("All graph weights must be strictly positive")
        return weights

    def _weights_for_indices(self, indices, device, dtype):
        missing = [
            int(index)
            for index in indices.tolist()
            if int(index) not in self.graph_weights
        ]
        if missing and self.strict_index:
            raise KeyError(
                "Missing graph weights for forget indices: "
                + ", ".join(map(str, missing[:10]))
            )

        weights = torch.tensor(
            [self.graph_weights.get(int(index), 1.0) for index in indices.tolist()],
            device=device,
            dtype=dtype,
        )
        if self.normalize_batch_weights:
            weights = weights / weights.mean().clamp_min(torch.finfo(dtype).eps)
        return weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_batch = inputs["forget"]
        if "index" not in forget_batch:
            raise KeyError(
                "GraphNPO requires forget example indices. Use "
                "collator=DataCollatorForSupervisedDatasetwithIndex."
            )

        indices = forget_batch["index"]
        forget_inputs = {
            "input_ids": forget_batch["input_ids"],
            "attention_mask": forget_batch["attention_mask"],
            "labels": forget_batch["labels"],
        }

        forget_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        with torch.no_grad():
            reference_nll, _ = compute_batch_nll(self.ref_model, forget_inputs)

        # This is the unreduced lose-only NPO objective used by compute_dpo_loss.
        per_example_npo = -2.0 / self.beta * F.logsigmoid(
            self.beta * (forget_nll - reference_nll)
        )
        weights = self._weights_for_indices(
            indices=indices,
            device=per_example_npo.device,
            dtype=per_example_npo.dtype,
        )
        forget_loss = (weights * per_example_npo).mean()

        retain_batch = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_batch["input_ids"],
            "attention_mask": retain_batch["attention_mask"],
            "labels": retain_batch["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
