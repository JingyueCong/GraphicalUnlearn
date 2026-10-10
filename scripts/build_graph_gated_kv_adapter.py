#!/usr/bin/env python3
"""Build a graph-prototype gate around an existing merged low-rank KV edit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_graph_kv_edit import extract_kv, load_graph_edges, select_indices
from build_model_memory_graph import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--edited-model-path", required=True)
    parser.add_argument("--forget-jsonl", type=Path, required=True)
    parser.add_argument("--retain-jsonl", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--graph-alpha", type=float, default=1.0)
    parser.add_argument("--target-retain-fpr", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--num-retain-anchors", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def resolve_module(model, path):
    current = model
    for component in path.split("."):
        current = current[int(component)] if component.isdigit() else getattr(current, component)
    return current


def graph_smooth(keys, edges, alpha):
    """One normalized adjacency smoothing step over forget keys."""
    import torch

    if alpha <= 0 or not edges:
        return keys.clone()
    accumulated = torch.zeros_like(keys)
    degrees = torch.zeros(len(keys), 1, dtype=keys.dtype, device=keys.device)
    for left, right, weight in edges:
        accumulated[left] += weight * keys[right]
        accumulated[right] += weight * keys[left]
        degrees[left] += weight
        degrees[right] += weight
    return (keys + alpha * accumulated) / (1.0 + alpha * degrees)


def max_cosine(keys, prototypes):
    import torch.nn.functional as functional

    keys = functional.normalize(keys.float(), dim=-1)
    prototypes = functional.normalize(prototypes.float(), dim=-1)
    return (keys @ prototypes.T).amax(dim=-1)


def calibrate_threshold(forget_scores, retain_scores, target_retain_fpr):
    import torch

    quantile = min(max(1.0 - target_retain_fpr, 0.0), 1.0)
    threshold = torch.quantile(retain_scores.float(), quantile)
    forget_tpr = (forget_scores >= threshold).float().mean()
    retain_fpr = (retain_scores >= threshold).float().mean()
    return float(threshold), float(forget_tpr), float(retain_fpr)


def factorize_delta(delta, rank, seed):
    import torch

    torch.manual_seed(seed)
    q = min(rank + 16, min(delta.shape))
    left, singular, right = torch.svd_lowrank(delta, q=q, niter=4)
    left = left[:, :rank]
    singular = singular[:rank]
    right = right[:, :rank]
    root = singular.sqrt()
    factor_b = left * root.unsqueeze(0)
    factor_a = root.unsqueeze(1) * right.T
    reconstruction = factor_b @ factor_a
    error = (reconstruction - delta).norm() / delta.norm().clamp_min(1e-30)
    return factor_a, factor_b, float(error), [float(value) for value in singular[:10]]


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rank < 1 or args.num_retain_anchors < 1:
        raise ValueError("rank and retain anchor count must be positive")
    if args.graph_alpha < 0 or not 0 <= args.target_retain_fpr <= 1:
        raise ValueError("invalid graph alpha or retain FPR")

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    module_path = f"model.layers.{args.layer}.mlp.down_proj"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    edited_model = AutoModelForCausalLM.from_pretrained(
        args.edited_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    base_model.eval()
    edited_model.eval()
    base_model.config.use_cache = False
    base_module = resolve_module(base_model, module_path)
    edited_module = resolve_module(edited_model, module_path)
    delta = edited_module.weight.float() - base_module.weight.float()
    factor_a, factor_b, reconstruction_error, singular_values = factorize_delta(
        delta, args.rank, args.seed
    )
    del edited_model, delta
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    forget_records = load_jsonl(args.forget_jsonl)
    retain_records = load_jsonl(args.retain_jsonl)
    retain_indices = select_indices(
        len(retain_records), args.num_retain_anchors, args.seed
    )
    forget_keys, _ = extract_kv(
        base_model,
        tokenizer,
        forget_records,
        base_module,
        args,
        include_values=False,
        prediction_positions=True,
    )
    retain_keys, _ = extract_kv(
        base_model,
        tokenizer,
        [retain_records[index] for index in retain_indices],
        base_module,
        args,
        include_values=False,
        prediction_positions=True,
    )
    edges = load_graph_edges(args.graph, len(forget_records)) if args.graph else []
    prototypes = graph_smooth(forget_keys, edges, args.graph_alpha)
    prototypes = functional.normalize(prototypes.float(), dim=-1)
    forget_scores = max_cosine(forget_keys, prototypes)
    retain_scores = max_cosine(retain_keys, prototypes)
    threshold, forget_tpr, retain_fpr = calibrate_threshold(
        forget_scores, retain_scores, args.target_retain_fpr
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "module": module_path,
        "factor_a": factor_a.half().cpu(),
        "factor_b": factor_b.half().cpu(),
        "prototypes": prototypes.half().cpu(),
        "threshold": threshold,
        "temperature": args.temperature,
    }
    torch.save(artifact, args.output)
    metadata = {
        "method": "graph_gated_low_rank_kv_adapter",
        "base_model_path": args.base_model_path,
        "edited_model_path": args.edited_model_path,
        "module": module_path,
        "rank": args.rank,
        "graph": str(args.graph) if args.graph else None,
        "graph_alpha": args.graph_alpha,
        "num_edges": len(edges),
        "num_prototypes": len(prototypes),
        "target_retain_fpr": args.target_retain_fpr,
        "threshold": threshold,
        "temperature": args.temperature,
        "calibration_forget_tpr": forget_tpr,
        "calibration_retain_fpr": retain_fpr,
        "forget_score_mean": float(forget_scores.mean()),
        "forget_score_min": float(forget_scores.min()),
        "retain_score_mean": float(retain_scores.mean()),
        "retain_score_max": float(retain_scores.max()),
        "delta_reconstruction_error": reconstruction_error,
        "top_singular_values": singular_values,
        "retain_indices": retain_indices,
    }
    metadata_path = args.output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
