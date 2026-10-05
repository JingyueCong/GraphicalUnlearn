#!/usr/bin/env python3
"""Test whether an internal MLP KV graph predicts causal edit interference.

For a sample of forget examples, this script extracts answer-token MLP keys
(the input to ``down_proj``) and values (its output). It then applies a
temporary rank-one value-erasure hook for each source example and measures the
exact answer-NLL change on held-out forget and retain examples. No checkpoint
is modified or saved.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_model_memory_graph import load_jsonl, tokenize_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-jsonl", type=Path, required=True)
    parser.add_argument("--retain-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num-sources", type=int, default=24)
    parser.add_argument("--num-forget-targets", type=int, default=96)
    parser.add_argument("--num-retain-targets", type=int, default=96)
    parser.add_argument("--causal-top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--intervention-strength", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_probe_indices(size, count, seed, required=()):
    """Select a deterministic subset while always including required indices."""
    required = list(dict.fromkeys(int(index) for index in required))
    if any(index < 0 or index >= size for index in required):
        raise ValueError("required index is out of range")
    count = min(size, max(count, len(required)))
    candidates = [index for index in range(size) if index not in set(required)]
    random.Random(seed).shuffle(candidates)
    return sorted(required + candidates[: count - len(required)])


def answer_mean_pool(tensor, labels):
    """Mean-pool a [batch, sequence, width] tensor over answer positions."""
    import torch

    mask = labels.ne(-100).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (tensor.float() * mask).sum(dim=1) / counts


def per_example_answer_nll(logits, labels):
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    mask = shifted_labels.ne(-100)
    return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def cosine_matrix(left, right):
    import torch.nn.functional as F

    left = F.normalize(left.float(), p=2, dim=-1)
    right = F.normalize(right.float(), p=2, dim=-1)
    return left @ right.T


def _rankdata(values):
    """Return average ranks, including ties, without requiring scipy."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left_rank)
    right_scale = sum((y - right_mean) ** 2 for y in right_rank)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else 0.0


def precision_at_k(scores, effects, k):
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    predicted = set(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])
    actual = set(sorted(range(len(effects)), key=lambda i: (-effects[i], i))[:k])
    return len(predicted & actual) / k


class RankOneValueEraser:
    """Forward hook equivalent to a temporary rank-one down_proj update."""

    def __init__(self, key, value, strength, ridge):
        self.key = key.detach().float()
        self.value = value.detach().float()
        self.strength = float(strength)
        self.ridge = float(ridge)

    def __call__(self, _module, inputs, output):
        import torch

        activations = inputs[0]
        key = self.key.to(device=activations.device)
        value = self.value.to(device=output.device)
        denominator = torch.dot(key, key).clamp_min(self.ridge)
        coefficient = torch.einsum("...d,d->...", activations.float(), key)
        correction = coefficient.div(denominator).unsqueeze(-1) * value
        return output - self.strength * correction.to(dtype=output.dtype)


def _tokenize(records, args):
    return [
        tokenize_record(
            tokenizer=args.tokenizer,
            record=record,
            question_key=args.question_key,
            answer_key=args.answer_key,
            system_prompt=args.system_prompt,
            date_string=args.date_string,
            max_length=args.max_length,
        )
        for record in records
    ]


def _batches(tokenized, batch_size, tokenizer, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    for start in range(0, len(tokenized), batch_size):
        batch = tokenized[start : start + batch_size]
        input_ids = pad_sequence(
            [torch.tensor(item[0]) for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)
        labels = pad_sequence(
            [torch.tensor(item[1]) for item in batch],
            batch_first=True,
            padding_value=-100,
        ).to(device)
        attention_mask = pad_sequence(
            [torch.ones(len(item[0]), dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=0,
        ).to(device)
        yield input_ids, labels, attention_mask


def encode_targets(model, module, tokenized, args):
    import torch

    losses, keys, values, hidden = [], [], [], []
    captured = {}

    def capture(_module, inputs, output):
        captured["key"] = inputs[0].detach()
        captured["value"] = output.detach()

    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for input_ids, labels, attention_mask in _batches(
                tokenized, args.batch_size, args.tokenizer, args.device
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                losses.extend(per_example_answer_nll(outputs.logits, labels).cpu())
                keys.extend(answer_mean_pool(captured["key"], labels).cpu())
                values.extend(answer_mean_pool(captured["value"], labels).cpu())
                hidden.extend(answer_mean_pool(outputs.hidden_states[-1], labels).cpu())
    finally:
        handle.remove()
    return (
        torch.stack(losses),
        torch.stack(keys),
        torch.stack(values),
        torch.stack(hidden),
    )


def evaluate_losses(model, tokenized, args):
    import torch

    losses = []
    with torch.inference_mode():
        for input_ids, labels, attention_mask in _batches(
            tokenized, args.batch_size, args.tokenizer, args.device
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            losses.extend(per_example_answer_nll(outputs.logits, labels).cpu())
    return torch.stack(losses)


def summarize_predictors(predictors, effects, target_splits, source_positions, top_k):
    split_masks = {
        "all": list(range(len(target_splits))),
        "forget": [i for i, split in enumerate(target_splits) if split == "forget"],
        "retain": [i for i, split in enumerate(target_splits) if split == "retain"],
    }
    summary = {}
    for name, score_matrix in predictors.items():
        summary[name] = {}
        for split, base_indices in split_masks.items():
            correlations, precisions = [], []
            for source_row, source_position in enumerate(source_positions):
                indices = [index for index in base_indices if index != source_position]
                scores = [abs(float(score_matrix[source_row, index])) for index in indices]
                causal = [abs(float(effects[source_row, index])) for index in indices]
                correlations.append(spearman(scores, causal))
                precisions.append(precision_at_k(scores, causal, top_k))
            summary[name][split] = {
                "mean_spearman_abs_effect": sum(correlations) / len(correlations),
                "mean_precision_at_k": sum(precisions) / len(precisions),
                "k": min(top_k, max(0, len(base_indices) - (split != "retain"))),
            }
    return summary


def main():
    args = parse_args()
    if args.num_sources < 1 or args.batch_size < 1:
        raise ValueError("source count and batch size must be positive")
    if not 0.0 < args.intervention_strength <= 1.0:
        raise ValueError("--intervention-strength must be in (0, 1]")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    forget_records = load_jsonl(args.forget_jsonl)
    retain_records = load_jsonl(args.retain_jsonl)
    source_indices = select_probe_indices(
        len(forget_records), args.num_sources, args.seed
    )
    forget_indices = select_probe_indices(
        len(forget_records),
        args.num_forget_targets,
        args.seed + 1,
        required=source_indices,
    )
    retain_indices = select_probe_indices(
        len(retain_records), args.num_retain_targets, args.seed + 2
    )
    target_records = [forget_records[index] for index in forget_indices] + [
        retain_records[index] for index in retain_indices
    ]
    target_splits = ["forget"] * len(forget_indices) + ["retain"] * len(retain_indices)
    target_dataset_indices = forget_indices + retain_indices
    forget_positions = {index: position for position, index in enumerate(forget_indices)}
    source_positions = [forget_positions[index] for index in source_indices]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.tokenizer.pad_token_id is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer {args.layer} is outside [0, {len(layers)})")
    module = layers[args.layer].mlp.down_proj

    tokenized = _tokenize(target_records, args)
    baseline, keys, values, hidden = encode_targets(model, module, tokenized, args)
    source_keys = keys[source_positions]
    source_values = values[source_positions]
    predictors = {
        "key_abs_cosine": cosine_matrix(source_keys, keys).abs(),
        "value_abs_cosine": cosine_matrix(source_values, values).abs(),
        "kv_product": (
            cosine_matrix(source_keys, keys).abs()
            * cosine_matrix(source_values, values).abs()
        ),
        "final_hidden_abs_cosine": cosine_matrix(
            hidden[source_positions], hidden
        ).abs(),
    }

    effects = []
    for source_number, (key, value) in enumerate(zip(source_keys, source_values)):
        eraser = RankOneValueEraser(
            key=key,
            value=value,
            strength=args.intervention_strength,
            ridge=args.ridge,
        )
        handle = module.register_forward_hook(eraser)
        try:
            intervened = evaluate_losses(model, tokenized, args)
        finally:
            handle.remove()
        effects.append(intervened - baseline)
        print(
            f"Layer {args.layer}: interventions {source_number + 1}/"
            f"{len(source_indices)}",
            flush=True,
        )
    effects = torch.stack(effects)
    summary = summarize_predictors(
        predictors, effects, target_splits, source_positions, args.causal_top_k
    )

    causal_edges = []
    for source_row, source_index in enumerate(source_indices):
        ranked = sorted(
            range(len(target_records)),
            key=lambda position: (-abs(float(effects[source_row, position])), position),
        )
        for target_position in ranked[: args.causal_top_k]:
            causal_edges.append(
                {
                    "source_forget_index": source_index,
                    "target_split": target_splits[target_position],
                    "target_dataset_index": target_dataset_indices[target_position],
                    "delta_nll": float(effects[source_row, target_position]),
                    "key_cosine": float(
                        cosine_matrix(source_keys[source_row : source_row + 1], keys[target_position : target_position + 1])[0, 0]
                    ),
                    "value_cosine": float(
                        cosine_matrix(source_values[source_row : source_row + 1], values[target_position : target_position + 1])[0, 0]
                    ),
                }
            )

    source_effects = [
        float(effects[row, position])
        for row, position in enumerate(source_positions)
    ]
    payload = {
        "version": 1,
        "method": "mlp_kv_rank_one_causal_interference_probe",
        "model_path": args.model_path,
        "layer": args.layer,
        "down_projection": f"model.layers.{args.layer}.mlp.down_proj",
        "parameters": {
            "num_sources": len(source_indices),
            "num_forget_targets": len(forget_indices),
            "num_retain_targets": len(retain_indices),
            "intervention_strength": args.intervention_strength,
            "ridge": args.ridge,
            "causal_top_k": args.causal_top_k,
            "seed": args.seed,
            "pooling": "answer_token_mean",
        },
        "source_forget_indices": source_indices,
        "source_delta_nll": source_effects,
        "source_delta_nll_summary": {
            "min": min(source_effects),
            "mean": sum(source_effects) / len(source_effects),
            "max": max(source_effects),
            "positive_fraction": sum(value > 0 for value in source_effects)
            / len(source_effects),
        },
        "predictor_summary": summary,
        "causal_edges": causal_edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload["source_delta_nll_summary"], indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote causal KV probe to {args.output}")


if __name__ == "__main__":
    main()
