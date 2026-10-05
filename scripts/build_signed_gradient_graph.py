#!/usr/bin/env python3
"""Build a signed forget graph from multi-layer answer-NLL gradients."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_forget_graph import detect_communities
from build_model_memory_graph import load_jsonl, tokenize_record


DEFAULT_PARAMETER_REGEX = (
    r"(?:input_layernorm|post_attention_layernorm|model\.norm)\.weight$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--parameter-regex", default=DEFAULT_PARAMETER_REGEX)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--positive-top-k", type=int, default=4)
    parser.add_argument("--negative-top-k", type=int, default=4)
    parser.add_argument("--min-abs-similarity", type=float, default=0.02)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_countsketch_maps(parameters, projection_dim, seed):
    """Create one deterministic bucket and sign map per parameter tensor."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    maps = []
    for parameter in parameters:
        size = parameter.numel()
        buckets = torch.randint(
            projection_dim, (size,), generator=generator, dtype=torch.int64
        )
        signs = torch.randint(2, (size,), generator=generator, dtype=torch.int8)
        signs = signs.float().mul_(2.0).sub_(1.0)
        maps.append((buckets, signs))
    return maps


def countsketch_gradient_tensors(gradients, sketch_maps, projection_dim):
    """Project gradients with a memory-efficient signed hashing transform."""
    import torch
    import torch.nn.functional as F

    if len(gradients) != len(sketch_maps):
        raise ValueError("Gradient tensors and CountSketch maps must have equal length")
    sketch = torch.zeros(projection_dim, dtype=torch.float32)
    for gradient, (buckets, signs) in zip(gradients, sketch_maps):
        values = gradient.detach().float().reshape(-1).cpu()
        if values.numel() != buckets.numel():
            raise ValueError("Gradient tensor does not match its CountSketch map")
        sketch.scatter_add_(0, buckets, values * signs)
    if not torch.isfinite(sketch).all() or float(sketch.norm()) == 0.0:
        raise ValueError("Encountered an invalid or zero gradient sketch")
    return F.normalize(sketch, p=2, dim=0)


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def build_signed_knn_graph(
    vectors,
    positive_top_k=4,
    negative_top_k=4,
    min_abs_similarity=0.02,
):
    """Build a union-kNN graph containing cooperative and conflicting edges."""
    size = len(vectors)
    adjacency = [[] for _ in range(size)]
    if size < 2:
        return adjacency, []

    positive_top_k = max(0, min(positive_top_k, size - 1))
    negative_top_k = max(0, min(negative_top_k, size - 1))
    similarities = [[0.0] * size for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            value = float(cosine(vectors[left], vectors[right]))
            similarities[left][right] = value
            similarities[right][left] = value

    selected = {}
    for left in range(size):
        candidates = [
            (right, similarities[left][right])
            for right in range(size)
            if right != left
        ]
        positives = sorted(
            (item for item in candidates if item[1] >= min_abs_similarity),
            key=lambda item: (-item[1], item[0]),
        )[:positive_top_k]
        negatives = sorted(
            (item for item in candidates if item[1] <= -min_abs_similarity),
            key=lambda item: (item[1], item[0]),
        )[:negative_top_k]
        for right, similarity in positives + negatives:
            edge = (min(left, right), max(left, right))
            previous = selected.get(edge)
            if previous is None or abs(similarity) > abs(previous):
                selected[edge] = similarity

    edges = []
    for (left, right), similarity in sorted(selected.items()):
        sign = 1 if similarity > 0 else -1
        weight = abs(float(similarity))
        adjacency[left].append((right, weight, sign))
        adjacency[right].append((left, weight, sign))
        edges.append(
            {
                "source": left,
                "target": right,
                "weight": weight,
                "sign": sign,
                "similarity": float(similarity),
            }
        )
    return adjacency, edges


def encode_gradient_sketches(records, args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    pattern = re.compile(args.parameter_regex)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if pattern.search(name)
    ]
    if not selected:
        raise ValueError(
            f"No parameters matched --parameter-regex={args.parameter_regex!r}"
        )
    selected_names = [name for name, _ in selected]
    selected_parameters = [parameter for _, parameter in selected]
    for parameter in selected_parameters:
        parameter.requires_grad_(True)

    sketch_maps = make_countsketch_maps(
        selected_parameters, args.projection_dim, args.projection_seed
    )
    sketches = []
    losses = []
    for index, record in enumerate(records):
        input_ids, labels = tokenize_record(
            tokenizer=tokenizer,
            record=record,
            question_key=args.question_key,
            answer_key=args.answer_key,
            system_prompt=args.system_prompt,
            date_string=args.date_string,
            max_length=args.max_length,
        )
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        labels = torch.tensor(labels, device=device).unsqueeze(0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid_tokens = shifted_labels.reshape(-1).ne(-100)
        loss = token_losses[valid_tokens].mean()
        gradients = torch.autograd.grad(
            loss,
            selected_parameters,
            retain_graph=False,
            create_graph=False,
        )
        sketch = countsketch_gradient_tensors(
            gradients, sketch_maps, args.projection_dim
        )
        sketches.append(sketch.tolist())
        losses.append(float(loss.detach()))
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"Encoded signed gradient sketches: {index + 1}/{len(records)}")

    for parameter in selected_parameters:
        parameter.requires_grad_(False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gradient_dimension = sum(parameter.numel() for parameter in selected_parameters)
    return sketches, losses, selected_names, gradient_dimension


def main():
    args = parse_args()
    if args.projection_dim < 1:
        raise ValueError("--projection-dim must be at least 1")
    if args.positive_top_k < 0 or args.negative_top_k < 0:
        raise ValueError("top-k values must be non-negative")
    if args.positive_top_k + args.negative_top_k < 1:
        raise ValueError("at least one signed top-k value must be positive")
    if not 0.0 <= args.min_abs_similarity <= 1.0:
        raise ValueError("--min-abs-similarity must be in [0, 1]")

    records = load_jsonl(args.input_jsonl)
    sketches, losses, parameter_names, gradient_dimension = encode_gradient_sketches(
        records, args
    )
    adjacency, edges = build_signed_knn_graph(
        sketches,
        positive_top_k=args.positive_top_k,
        negative_top_k=args.negative_top_k,
        min_abs_similarity=args.min_abs_similarity,
    )
    positive_edges = [edge for edge in edges if edge["sign"] > 0]
    negative_edges = [edge for edge in edges if edge["sign"] < 0]
    if not negative_edges:
        raise ValueError(
            "No conflicting gradient pairs passed the threshold; lower "
            "--min-abs-similarity or change the gradient parameter set"
        )

    communities, community_sizes = detect_communities(
        num_nodes=len(records),
        edges=positive_edges,
        resolution=args.community_resolution,
        seed=args.community_seed,
    )
    degrees = [len(neighbors) for neighbors in adjacency]
    payload = {
        "version": 4,
        "method": "signed_multilayer_answer_nll_gradient_countsketch_knn",
        "source": {
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
        "parameters": {
            "parameter_regex": args.parameter_regex,
            "parameter_names": parameter_names,
            "gradient_dimension": gradient_dimension,
            "projection": "countsketch",
            "projection_dim": args.projection_dim,
            "projection_seed": args.projection_seed,
            "positive_top_k": args.positive_top_k,
            "negative_top_k": args.negative_top_k,
            "min_abs_similarity": args.min_abs_similarity,
            "max_length": args.max_length,
            "union_knn": True,
            "community_resolution": args.community_resolution,
            "community_seed": args.community_seed,
        },
        "answer_nll_summary": {
            "min": min(losses),
            "mean": sum(losses) / len(losses),
            "max": max(losses),
        },
        "num_nodes": len(records),
        "num_edges": len(edges),
        "num_positive_edges": len(positive_edges),
        "num_negative_edges": len(negative_edges),
        "num_communities": len(community_sizes),
        "degree_summary": {
            "min": min(degrees, default=0),
            "mean": sum(degrees) / max(1, len(degrees)),
            "max": max(degrees, default=0),
        },
        "weights": {str(index): 1.0 for index in range(len(records))},
        "communities": communities,
        "community_sizes": community_sizes,
        "edges": edges,
    }
    payload["parameters"]["num_parameter_tensors"] = len(parameter_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote signed gradient graph with {len(records)} nodes, "
        f"{len(positive_edges)} positive edges, {len(negative_edges)} negative "
        f"edges, and degree {payload['degree_summary']} to {args.output}"
    )


if __name__ == "__main__":
    main()
