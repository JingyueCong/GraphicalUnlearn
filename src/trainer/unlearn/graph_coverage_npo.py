import json
import logging
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll


logger = logging.getLogger(__name__)


class GraphCoverageNPO(GradDiff):
    """NPO with dynamic graph coverage and a retain-utility constraint.

    Static graph centrality is a weak proxy for how much an example still needs
    to be forgotten.  This trainer instead tracks a per-example forgetting
    residual, propagates it over the forget graph, and balances the resulting
    weights across graph communities.  Examples that lag behind their semantic
    neighbors receive more optimization power; already-forgotten examples fade
    out automatically.

    The retain coefficient is a lightweight Lagrange multiplier. Its controller
    tracks an EMA of retain-budget violations and can raise or lower the
    coefficient as the model moves outside or back inside the utility budget.
    """

    def __init__(
        self,
        graph_weights_path,
        beta=0.1,
        residual_temperature=1.0,
        residual_ema_decay=0.9,
        propagation_strength=0.3,
        propagation_interval=10,
        community_balance_power=1.0,
        weight_floor=0.5,
        weight_ceiling=2.0,
        normalize_global_weights=True,
        strict_index=True,
        adaptive_retain=True,
        retain_budget=0.05,
        adaptive_alpha_lr=0.05,
        alpha_update_rule="additive",
        retain_violation_ema_decay=0.9,
        retain_violation_clip=1.0,
        retain_violation_deadband=0.0,
        alpha_min=None,
        alpha_max=4.0,
        diagnostic_interval=10,
        *args,
        **kwargs,
    ):
        self._validate_hyperparameters(
            residual_temperature=residual_temperature,
            residual_ema_decay=residual_ema_decay,
            propagation_strength=propagation_strength,
            propagation_interval=propagation_interval,
            community_balance_power=community_balance_power,
            weight_floor=weight_floor,
            weight_ceiling=weight_ceiling,
            retain_budget=retain_budget,
            adaptive_alpha_lr=adaptive_alpha_lr,
            alpha_update_rule=alpha_update_rule,
            retain_violation_ema_decay=retain_violation_ema_decay,
            retain_violation_clip=retain_violation_clip,
            retain_violation_deadband=retain_violation_deadband,
            alpha_max=alpha_max,
            diagnostic_interval=diagnostic_interval,
        )
        self.graph_path = Path(to_absolute_path(graph_weights_path))
        graph = self._load_graph(self.graph_path)

        super().__init__(*args, **kwargs)
        self.beta = beta
        self.residual_temperature = residual_temperature
        self.residual_ema_decay = residual_ema_decay
        self.propagation_strength = propagation_strength
        self.propagation_interval = int(propagation_interval)
        self.community_balance_power = community_balance_power
        self.weight_floor = weight_floor
        self.weight_ceiling = weight_ceiling
        self.normalize_global_weights = normalize_global_weights
        self.strict_index = strict_index
        self.adaptive_retain = adaptive_retain
        self.retain_budget = retain_budget
        self.adaptive_alpha_lr = adaptive_alpha_lr
        self.alpha_update_rule = alpha_update_rule
        self.retain_violation_ema_decay = retain_violation_ema_decay
        self.retain_violation_clip = retain_violation_clip
        self.retain_violation_deadband = retain_violation_deadband
        self.base_alpha = float(self.alpha)
        self.alpha_min = self.base_alpha if alpha_min is None else float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.diagnostic_interval = int(diagnostic_interval)
        if not 0.0 <= self.alpha_min <= self.alpha_max:
            raise ValueError("alpha bounds must satisfy 0 <= alpha_min <= alpha_max")
        self.current_alpha = min(self.alpha_max, max(self.alpha_min, self.base_alpha))

        self.num_nodes = graph["num_nodes"]
        self.adjacency = graph["adjacency"]
        self.communities = graph["communities"]
        self.community_sizes = Counter(self.communities.values())
        self.mean_community_size = self.num_nodes / max(1, len(self.community_sizes))
        self.residual_ema = torch.ones(self.num_nodes, dtype=torch.float32)
        self.global_node_weights = torch.ones(self.num_nodes, dtype=torch.float32)
        self._last_batch_weights = None
        self._last_retain_violation = 0.0
        self._retain_violation_ema = 0.0
        self._retain_violation_ema_initialized = False
        self._coverage_updates = 0
        self._optimizer_updates = 0
        self._retain_violation_sum = 0.0
        self._retain_violation_count = 0
        self._microbatches_in_step = 0

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        logger.info(
            "Loaded coverage graph from %s (%d nodes, %d edges, %d communities)",
            self.graph_path,
            self.num_nodes,
            graph["num_edges"],
            len(self.community_sizes),
        )

    @staticmethod
    def _validate_hyperparameters(
        residual_temperature,
        residual_ema_decay,
        propagation_strength,
        propagation_interval,
        community_balance_power,
        weight_floor,
        weight_ceiling,
        retain_budget,
        adaptive_alpha_lr,
        alpha_update_rule,
        retain_violation_ema_decay,
        retain_violation_clip,
        retain_violation_deadband,
        alpha_max,
        diagnostic_interval,
    ):
        if residual_temperature <= 0:
            raise ValueError("residual_temperature must be positive")
        if not 0.0 <= residual_ema_decay < 1.0:
            raise ValueError("residual_ema_decay must be in [0, 1)")
        if not 0.0 <= propagation_strength <= 1.0:
            raise ValueError("propagation_strength must be in [0, 1]")
        if propagation_interval < 1:
            raise ValueError("propagation_interval must be at least 1")
        if community_balance_power < 0:
            raise ValueError("community_balance_power must be non-negative")
        if not 0.0 < weight_floor <= weight_ceiling:
            raise ValueError("weight bounds must satisfy 0 < floor <= ceiling")
        if retain_budget < 0:
            raise ValueError("retain_budget must be non-negative")
        if adaptive_alpha_lr < 0:
            raise ValueError("adaptive_alpha_lr must be non-negative")
        if alpha_update_rule not in {"additive", "multiplicative"}:
            raise ValueError(
                "alpha_update_rule must be 'additive' or 'multiplicative'"
            )
        if not 0.0 <= retain_violation_ema_decay < 1.0:
            raise ValueError("retain_violation_ema_decay must be in [0, 1)")
        if retain_violation_clip <= 0:
            raise ValueError("retain_violation_clip must be positive")
        if retain_violation_deadband < 0:
            raise ValueError("retain_violation_deadband must be non-negative")
        if retain_violation_deadband > retain_violation_clip:
            raise ValueError(
                "retain_violation_deadband must not exceed retain_violation_clip"
            )
        if alpha_max < 0:
            raise ValueError("alpha_max must be non-negative")
        if diagnostic_interval < 0:
            raise ValueError("diagnostic_interval must be non-negative")

    @staticmethod
    def _load_graph(path):
        if not path.is_file():
            raise FileNotFoundError(f"Graph file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        raw_weights = payload.get("weights", {})
        num_nodes = int(payload.get("num_nodes", len(raw_weights)))
        if num_nodes <= 0:
            raise ValueError("Graph must contain at least one node")

        adjacency = {index: [] for index in range(num_nodes)}
        edges = payload.get("edges", [])
        for edge in edges:
            source = int(edge["source"])
            target = int(edge["target"])
            weight = float(edge.get("weight", 1.0))
            if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
                raise ValueError("Graph edge contains an out-of-range node index")
            if weight <= 0:
                raise ValueError("Graph edge weights must be positive")
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))

        raw_communities = payload.get("communities")
        if raw_communities is None:
            logger.warning(
                "Graph has no communities; using one fallback community. "
                "Rebuild it with scripts/build_forget_graph.py for coverage balancing."
            )
            communities = {index: 0 for index in range(num_nodes)}
        else:
            communities = {
                int(index): int(community)
                for index, community in raw_communities.items()
            }
            missing = [index for index in range(num_nodes) if index not in communities]
            if missing:
                raise ValueError(
                    "Graph community mapping is missing node indices: "
                    + ", ".join(map(str, missing[:10]))
                )

        return {
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "adjacency": adjacency,
            "communities": communities,
        }

    def _validate_indices(self, indices):
        invalid = [index for index in indices if not 0 <= index < self.num_nodes]
        if invalid and self.strict_index:
            raise KeyError(
                "Graph is missing forget indices: "
                + ", ".join(map(str, invalid[:10]))
            )
        return invalid

    def _bounded_global_mean_one(self, values):
        if self.normalize_global_weights:
            values = values / values.mean().clamp_min(torch.finfo(values.dtype).eps)
        for _ in range(3):
            values = values.clamp(self.weight_floor, self.weight_ceiling)
            if self.normalize_global_weights:
                values = values / values.mean().clamp_min(
                    torch.finfo(values.dtype).eps
                )
        return values.clamp(self.weight_floor, self.weight_ceiling)

    @torch.no_grad()
    def _update_residuals(self, indices, per_token_margins):
        index_values = [int(index) for index in indices.detach().cpu().tolist()]
        self._validate_indices(index_values)

        residuals = torch.exp(
            (-per_token_margins.detach().float() / self.residual_temperature)
            .clamp(-6.0, 6.0)
            .cpu()
        )
        for index, residual in zip(index_values, residuals):
            if 0 <= index < self.num_nodes:
                old = self.residual_ema[index]
                self.residual_ema[index] = (
                    self.residual_ema_decay * old
                    + (1.0 - self.residual_ema_decay) * residual
                )
        return index_values

    @torch.no_grad()
    def _refresh_global_node_weights(self):
        raw_weights = []
        for index in range(self.num_nodes):
            own_residual = float(self.residual_ema[index])
            neighbors = self.adjacency[index]
            if neighbors:
                total_edge_weight = sum(weight for _, weight in neighbors)
                neighbor_residual = sum(
                    weight * float(self.residual_ema[neighbor])
                    for neighbor, weight in neighbors
                ) / total_edge_weight
            else:
                neighbor_residual = own_residual
            propagated = (
                (1.0 - self.propagation_strength) * own_residual
                + self.propagation_strength * neighbor_residual
            )

            community = self.communities[index]
            community_size = self.community_sizes[community]
            balance = (
                self.mean_community_size / community_size
            ) ** self.community_balance_power
            raw_weights.append(propagated * balance)

        weights = torch.tensor(raw_weights, dtype=torch.float32)
        self.global_node_weights = self._bounded_global_mean_one(weights).cpu()
        return self.global_node_weights

    @torch.no_grad()
    def _dynamic_weights_for_batch(
        self, indices, per_token_margins, device, dtype
    ):
        index_values = self._update_residuals(indices, per_token_margins)
        selected = [
            float(self.global_node_weights[index])
            if 0 <= index < self.num_nodes
            else 1.0
            for index in index_values
        ]
        weights = torch.tensor(selected, device=device, dtype=dtype)
        self._last_batch_weights = weights.detach().float().cpu()
        return weights

    def _retain_violation(self, retain_loss, reference_retain_loss=None):
        if reference_retain_loss is None:
            reference_value = 0.0
        else:
            reference_value = float(reference_retain_loss.detach())
        return float(retain_loss.detach()) - reference_value - self.retain_budget

    def _apply_alpha_update(self, violation):
        self._last_retain_violation = violation
        if not self._retain_violation_ema_initialized:
            self._retain_violation_ema = violation
            self._retain_violation_ema_initialized = True
        else:
            decay = self.retain_violation_ema_decay
            self._retain_violation_ema = (
                decay * self._retain_violation_ema + (1.0 - decay) * violation
            )
        if not self.adaptive_retain:
            return self.current_alpha
        control_violation = self._retain_violation_ema
        if abs(control_violation) <= self.retain_violation_deadband:
            control_violation = 0.0
        control_violation = max(
            -self.retain_violation_clip,
            min(self.retain_violation_clip, control_violation),
        )
        if self.alpha_update_rule == "additive":
            updated = self.current_alpha + self.adaptive_alpha_lr * control_violation
        else:
            updated = self.current_alpha * math.exp(
                self.adaptive_alpha_lr * control_violation
            )
        self.current_alpha = min(self.alpha_max, max(self.alpha_min, updated))
        return self.current_alpha

    def _is_optimizer_step_boundary(self):
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "sync_gradients"):
            return bool(accelerator.sync_gradients)
        gradient_accumulation_steps = max(
            1,
            int(
                getattr(
                    getattr(self, "args", None),
                    "gradient_accumulation_steps",
                    1,
                )
            ),
        )
        return self._microbatches_in_step >= gradient_accumulation_steps

    def _record_retain_violation(self, retain_loss, reference_retain_loss=None):
        self._microbatches_in_step += 1
        if self.adaptive_retain:
            violation = self._retain_violation(retain_loss, reference_retain_loss)
            self._retain_violation_sum += violation
            self._retain_violation_count += 1
        if not self._is_optimizer_step_boundary():
            return False

        self._microbatches_in_step = 0
        self._optimizer_updates += 1
        if self.adaptive_retain:
            mean_violation = self._retain_violation_sum / self._retain_violation_count
            self._apply_alpha_update(mean_violation)
        self._retain_violation_sum = 0.0
        self._retain_violation_count = 0
        if self._optimizer_updates % self.propagation_interval == 0:
            self._refresh_global_node_weights()
        return True

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_batch = inputs["forget"]
        if "index" not in forget_batch:
            raise KeyError(
                "GraphCoverageNPO requires forget example indices. Use "
                "collator=DataCollatorForSupervisedDatasetwithIndex."
            )

        forget_inputs = {
            "input_ids": forget_batch["input_ids"],
            "attention_mask": forget_batch["attention_mask"],
            "labels": forget_batch["labels"],
        }
        forget_nll, forget_outputs = compute_batch_nll(model, forget_inputs)
        with torch.no_grad():
            reference_nll, _ = compute_batch_nll(self.ref_model, forget_inputs)

        per_example_npo = -2.0 / self.beta * F.logsigmoid(
            self.beta * (forget_nll - reference_nll)
        )
        token_counts = (
            forget_inputs["labels"][..., 1:] != -100
        ).sum(dim=-1).clamp_min(1)
        per_token_margins = (forget_nll - reference_nll) / token_counts
        weights = self._dynamic_weights_for_batch(
            indices=forget_batch["index"],
            per_token_margins=per_token_margins,
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
        reference_retain_loss = None
        if self.adaptive_retain and self.retain_loss_type == "NLL":
            with torch.no_grad():
                reference_retain_loss = self.ref_model(**retain_inputs).loss
        # The current coefficient applies to every micro-batch in this optimizer
        # step.  The averaged constraint violation updates alpha for the next
        # optimizer step only, avoiding a gradient-accumulation-dependent rate.
        alpha = self.current_alpha
        alpha_updated = self._record_retain_violation(
            retain_loss, reference_retain_loss
        )

        self._coverage_updates += 1
        if (
            self.diagnostic_interval
            and alpha_updated
            and self._optimizer_updates % self.diagnostic_interval == 0
        ):
            logger.info(
                "optimizer update=%d microbatches=%d "
                "global_weights[min=%.4f mean=%.4f max=%.4f] "
                "alpha=%.4f retain_violation=%.4f violation_ema=%.4f",
                self._optimizer_updates,
                self._coverage_updates,
                float(self.global_node_weights.min()),
                float(self.global_node_weights.mean()),
                float(self.global_node_weights.max()),
                self.current_alpha,
                self._last_retain_violation,
                self._retain_violation_ema,
            )

        loss = self.gamma * forget_loss + alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
