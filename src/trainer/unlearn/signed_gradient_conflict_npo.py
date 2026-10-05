import json
import logging
import math

import torch

from trainer.unlearn.graph_coverage_npo import GraphCoverageNPO


logger = logging.getLogger(__name__)


class SignedGradientConflictNPO(GraphCoverageNPO):
    """NPO with signed gradient-graph allocation.

    Positive edges smooth forgetting demand between examples whose answer-NLL
    gradients cooperate. Negative edges sharpen the allocation: of two
    conflicting examples, the endpoint with the larger forgetting residual
    receives more weight while the other is temporarily suppressed. This
    avoids spending equal update budget on directions that cancel each other.
    """

    def __init__(
        self,
        cooperative_strength=0.25,
        conflict_strength=0.7,
        conflict_temperature=0.5,
        require_negative_edges=True,
        *args,
        **kwargs,
    ):
        if not 0.0 <= cooperative_strength <= 1.0:
            raise ValueError("cooperative_strength must be in [0, 1]")
        if conflict_strength < 0.0:
            raise ValueError("conflict_strength must be non-negative")
        if conflict_temperature <= 0.0:
            raise ValueError("conflict_temperature must be positive")

        self.cooperative_strength = float(cooperative_strength)
        self.conflict_strength = float(conflict_strength)
        self.conflict_temperature = float(conflict_temperature)
        self.require_negative_edges = bool(require_negative_edges)
        # The signed refresh below replaces the unsigned propagation in the
        # parent class. Keep the legacy coefficient disabled explicitly.
        kwargs["propagation_strength"] = 0.0
        super().__init__(*args, **kwargs)

        signed = self._load_signed_adjacency(self.graph_path, self.num_nodes)
        self.positive_adjacency = signed["positive_adjacency"]
        self.negative_adjacency = signed["negative_adjacency"]
        self.num_positive_edges = signed["num_positive_edges"]
        self.num_negative_edges = signed["num_negative_edges"]
        if self.require_negative_edges and self.num_negative_edges == 0:
            raise ValueError("Signed conflict graph contains no negative edges")

        logger.info(
            "Loaded signed gradient graph with %d cooperative and %d conflict edges",
            self.num_positive_edges,
            self.num_negative_edges,
        )

    @staticmethod
    def _load_signed_adjacency(path, num_nodes):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        positive_adjacency = {index: [] for index in range(num_nodes)}
        negative_adjacency = {index: [] for index in range(num_nodes)}
        num_positive_edges = 0
        num_negative_edges = 0
        for edge in payload.get("edges", []):
            source = int(edge["source"])
            target = int(edge["target"])
            weight = float(edge.get("weight", abs(float(edge.get("similarity", 0)))))
            sign = int(edge.get("sign", 1))
            if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
                raise ValueError("Signed graph edge has an out-of-range node index")
            if source == target:
                raise ValueError("Signed graph must not contain self edges")
            if weight <= 0.0 or not math.isfinite(weight):
                raise ValueError("Signed graph edge weights must be finite and positive")
            if sign not in {-1, 1}:
                raise ValueError("Signed graph edge sign must be -1 or 1")
            adjacency = positive_adjacency if sign > 0 else negative_adjacency
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))
            if sign > 0:
                num_positive_edges += 1
            else:
                num_negative_edges += 1

        return {
            "positive_adjacency": positive_adjacency,
            "negative_adjacency": negative_adjacency,
            "num_positive_edges": num_positive_edges,
            "num_negative_edges": num_negative_edges,
        }

    @staticmethod
    def _weighted_mean(values, neighbors, fallback):
        if not neighbors:
            return fallback
        total_weight = sum(weight for _, weight in neighbors)
        return sum(weight * values[index] for index, weight in neighbors) / total_weight

    @torch.no_grad()
    def _refresh_global_node_weights(self):
        eps = torch.finfo(torch.float32).tiny
        log_residuals = self.residual_ema.float().clamp_min(eps).log().tolist()
        raw_weights = []
        conflict_contrasts = []

        for index in range(self.num_nodes):
            own_log = log_residuals[index]
            cooperative_log = self._weighted_mean(
                log_residuals,
                self.positive_adjacency[index],
                own_log,
            )
            score_log = (
                (1.0 - self.cooperative_strength) * own_log
                + self.cooperative_strength * cooperative_log
            )

            conflict_neighbors = self.negative_adjacency[index]
            if conflict_neighbors:
                total_weight = sum(weight for _, weight in conflict_neighbors)
                contrast = sum(
                    weight
                    * math.tanh(
                        (own_log - log_residuals[neighbor])
                        / self.conflict_temperature
                    )
                    for neighbor, weight in conflict_neighbors
                ) / total_weight
                score_log += self.conflict_strength * contrast
                conflict_contrasts.append(abs(contrast))

            community = self.communities[index]
            community_size = self.community_sizes[community]
            balance = (
                self.mean_community_size / community_size
            ) ** self.community_balance_power
            raw_weights.append(math.exp(max(-6.0, min(6.0, score_log))) * balance)

        weights = torch.tensor(raw_weights, dtype=torch.float32)
        self.global_node_weights = self._bounded_global_mean_one(weights).cpu()
        self._mean_abs_conflict_contrast = (
            sum(conflict_contrasts) / len(conflict_contrasts)
            if conflict_contrasts
            else 0.0
        )
        return self.global_node_weights
