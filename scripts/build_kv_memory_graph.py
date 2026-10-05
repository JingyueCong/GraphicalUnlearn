#!/usr/bin/env python3
"""Build a forget graph from internal MLP key/value activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_forget_graph import detect_communities
from build_model_memory_graph import (
    build_mutual_knn_graph,
    cosine,
    load_jsonl,
    tokenize_record,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pool_answer_activations(activations, labels):
    import torch.nn.functional as F

    mask = labels.ne(-100).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    pooled = (activations.float() * mask).sum(dim=1) / counts
    return F.normalize(pooled, p=2, dim=-1)


def encode_kv(records, args):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    if args.layer < 0 or args.layer >= len(model.model.layers):
        raise ValueError(
            f"layer {args.layer} is outside [0, {len(model.model.layers)})"
        )
    module = model.model.layers[args.layer].mlp.down_proj
    captured = {}

    def capture(_module, inputs, output):
        captured["key"] = inputs[0].detach()
        captured["value"] = output.detach()

    handle = module.register_forward_hook(capture)
    tokenized = [
        tokenize_record(
            tokenizer=tokenizer,
            record=record,
            question_key=args.question_key,
            answer_key=args.answer_key,
            system_prompt=args.system_prompt,
            date_string=args.date_string,
            max_length=args.max_length,
        )
        for record in records
    ]
    keys, values = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(tokenized), args.batch_size):
                batch = tokenized[start : start + args.batch_size]
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
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                keys.extend(pool_answer_activations(captured["key"], labels).cpu().tolist())
                values.extend(
                    pool_answer_activations(captured["value"], labels).cpu().tolist()
                )
    finally:
        handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return keys, values


def main():
    args = parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch size must be positive")
    records = load_jsonl(args.input_jsonl)
    keys, values = encode_kv(records, args)
    adjacency, edges = build_mutual_knn_graph(
        keys,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        absolute_similarity=True,
    )
    for edge in edges:
        left, right = edge["source"], edge["target"]
        edge["key_abs_cosine"] = edge["weight"]
        edge["value_cosine"] = float(cosine(values[left], values[right]))
    communities, community_sizes = detect_communities(
        num_nodes=len(records),
        edges=edges,
        resolution=args.community_resolution,
        seed=args.community_seed,
    )
    degrees = [len(neighbors) for neighbors in adjacency]
    payload = {
        "version": 1,
        "method": "mlp_answer_key_abs_cosine_mutual_knn",
        "source": {
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
        "parameters": {
            "layer": args.layer,
            "module": f"model.layers.{args.layer}.mlp.down_proj",
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "max_length": args.max_length,
            "pooling": "answer_token_mean",
            "edge_weight": "absolute_key_cosine",
            "mutual_knn": True,
            "community_resolution": args.community_resolution,
            "community_seed": args.community_seed,
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
        f"Wrote layer-{args.layer} KV graph with {len(records)} nodes, "
        f"{len(edges)} edges and {len(community_sizes)} communities to {args.output}"
    )


if __name__ == "__main__":
    main()
