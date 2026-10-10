from __future__ import annotations

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM


def resolve_module(model, path):
    current = model
    for component in path.split("."):
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


class GraphGatedLowRankHook:
    """Conditionally apply B(Ax) using similarity to graph-smoothed KV prototypes."""

    def __init__(self, factor_a, factor_b, prototypes, threshold, temperature, device):
        self.factor_a = factor_a.to(device=device, dtype=torch.float32)
        self.factor_b = factor_b.to(device=device, dtype=torch.float32)
        self.prototypes = functional.normalize(
            prototypes.to(device=device, dtype=torch.float32), dim=-1
        )
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("gate temperature must be positive")

    def gate(self, keys):
        normalized = functional.normalize(keys.float(), dim=-1)
        similarity = normalized @ self.prototypes.T
        max_similarity = similarity.amax(dim=-1)
        return torch.sigmoid(
            (max_similarity - self.threshold) / self.temperature
        )

    def __call__(self, _module, inputs, output):
        hidden = functional.linear(inputs[0].float(), self.factor_a)
        update = functional.linear(hidden, self.factor_b)
        gate = self.gate(inputs[0]).unsqueeze(-1)
        return output + (gate * update).to(dtype=output.dtype)


class GatedKVAutoModelForCausalLM:
    """AutoModel loader that restores a graph-gated low-rank forward hook."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        adapter_path = kwargs.pop("gated_adapter_path")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )
        artifact = torch.load(adapter_path, map_location="cpu", weights_only=True)
        module = resolve_module(model, artifact["module"])
        hook = GraphGatedLowRankHook(
            artifact["factor_a"],
            artifact["factor_b"],
            artifact["prototypes"],
            artifact["threshold"],
            artifact["temperature"],
            module.weight.device,
        )
        handle = module.register_forward_hook(hook)
        model._graph_gated_kv_hook = hook
        model._graph_gated_kv_handle = handle
        return model
