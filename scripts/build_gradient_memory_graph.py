#!/usr/bin/env python3
"""Build a forget graph from per-example answer-loss gradient sketches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_forget_graph import detect_communities
from build_model_memory_graph import (
    build_mutual_knn_graph,
    load_jsonl,
    tokenize_record,
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
    parser.add_argument("--parameter-pattern", default="model.norm")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def project_gradient_tensors(gradients, projection):
    import torch
    import torch.nn.functional as F

    flattened = torch.cat(
        [gradient.detach().float().reshape(-1).cpu() for gradient in gradients]
    )
    sketch = flattened @ projection
    if not torch.isfinite(sketch).all() or float(sketch.norm()) == 0.0:
        raise ValueError("Encountered an invalid or zero gradient sketch")
    return F.normalize(sketch, p=2, dim=0)


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

    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if args.parameter_pattern in name
    ]
    if not selected:
        raise ValueError(
            f"No parameters matched --parameter-pattern={args.parameter_pattern!r}"
        )
    selected_names = [name for name, _ in selected]
    selected_parameters = [parameter for _, parameter in selected]
    for parameter in selected_parameters:
        parameter.requires_grad_(True)

    total_dimension = sum(parameter.numel() for parameter in selected_parameters)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.projection_seed)
    projection = torch.randn(
        total_dimension,
        args.projection_dim,
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(args.projection_dim)

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
        sketches.append(project_gradient_tensors(gradients, projection).tolist())
        losses.append(float(loss.detach()))
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"Encoded gradient sketches: {index + 1}/{len(records)}")

    for parameter in selected_parameters:
        parameter.requires_grad_(False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sketches, losses, selected_names, total_dimension


def main():
    args = parse_args()
    if args.projection_dim < 1:
        raise ValueError("--projection-dim must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1")

    records = load_jsonl(args.input_jsonl)
    sketches, losses, parameter_names, gradient_dimension = encode_gradient_sketches(
        records, args
    )
    adjacency, edges = build_mutual_knn_graph(
        sketches,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    communities, community_sizes = detect_communities(
        num_nodes=len(records),
        edges=edges,
        resolution=args.community_resolution,
        seed=args.community_seed,
    )
    degrees = [len(neighbors) for neighbors in adjacency]
    payload = {
        "version": 3,
        "method": "answer_nll_gradient_sketch_mutual_knn",
        "source": {
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
        "parameters": {
            "parameter_pattern": args.parameter_pattern,
            "parameter_names": parameter_names,
            "gradient_dimension": gradient_dimension,
            "projection_dim": args.projection_dim,
            "projection_seed": args.projection_seed,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "max_length": args.max_length,
            "mutual_knn": True,
            "isolated_node_backfill": True,
            "component_backfill": True,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote gradient-memory graph with {len(records)} nodes, {len(edges)} "
        f"edges, {len(community_sizes)} communities, degree "
        f"{payload['degree_summary']} to {args.output}"
    )


if __name__ == "__main__":
    main()
