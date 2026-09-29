#!/usr/bin/env python3
"""Build a forget graph from answer-token representations of the base model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_forget_graph import detect_communities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _answer_text(answer):
    if isinstance(answer, list):
        return " ".join(map(str, answer))
    return str(answer)


def tokenize_record(
    tokenizer,
    record,
    question_key,
    answer_key,
    system_prompt,
    date_string,
    max_length,
):
    if question_key not in record or answer_key not in record:
        raise KeyError(f"Record lacks '{question_key}' or '{answer_key}'")

    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.extend(
        [
            {"role": "user", "content": str(record[question_key])},
            {"role": "assistant", "content": _answer_text(record[answer_key])},
        ]
    )
    date_info = {"date_string": date_string} if date_string else {}
    input_ids = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=False,
        **date_info,
    )[:max_length]
    prompt_ids = tokenizer.apply_chat_template(
        chat[:-1],
        tokenize=True,
        add_generation_prompt=True,
        **date_info,
    )[:max_length]
    if input_ids and input_ids[-1] != tokenizer.eos_token_id:
        if len(input_ids) < max_length:
            input_ids.append(tokenizer.eos_token_id)
        else:
            input_ids[-1] = tokenizer.eos_token_id

    answer_start = min(len(prompt_ids), len(input_ids))
    labels = [-100] * answer_start + input_ids[answer_start:]
    if not any(label != -100 for label in labels):
        raise ValueError("Answer was fully truncated; increase --max-length")
    return input_ids, labels


def answer_mean_pool(hidden_states, labels):
    """Mean-pool final-layer states at answer-token positions."""
    import torch
    import torch.nn.functional as F

    mask = labels.ne(-100).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    pooled = (hidden_states.float() * mask).sum(dim=1) / counts
    return F.normalize(pooled, p=2, dim=-1)


def encode_answer_representations(records, args):
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

    embeddings = []
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            pooled = answer_mean_pool(outputs.hidden_states[-1], labels)
            embeddings.extend(pooled.cpu().tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def build_mutual_knn_graph(vectors, top_k=8, min_similarity=0.0):
    """Build a connected mutual-kNN graph with conservative backfill edges."""
    size = len(vectors)
    if size < 2:
        return [[] for _ in range(size)], []
    top_k = max(1, min(top_k, size - 1))
    similarities = [[-1.0] * size for _ in range(size)]
    directed = []
    for left in range(size):
        candidates = []
        for right in range(size):
            if left == right:
                continue
            value = cosine(vectors[left], vectors[right])
            similarities[left][right] = value
            if value >= min_similarity:
                candidates.append((right, value))
        directed.append(
            sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]
        )

    neighbor_sets = [{node for node, _ in row} for row in directed]
    edge_weights = {}
    for left, neighbors in enumerate(directed):
        for right, weight in neighbors:
            if left in neighbor_sets[right]:
                edge_weights[(min(left, right), max(left, right))] = max(
                    float(weight), 1e-6
                )

    degree = [0] * size
    for left, right in edge_weights:
        degree[left] += 1
        degree[right] += 1
    for left in range(size):
        if degree[left] > 0:
            continue
        right = max(
            (node for node in range(size) if node != left),
            key=lambda node: (similarities[left][node], -node),
        )
        edge = (min(left, right), max(left, right))
        if edge not in edge_weights:
            edge_weights[edge] = max(float(similarities[left][right]), 1e-6)
            degree[left] += 1
            degree[right] += 1

    parent = list(range(size))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    for left, right in edge_weights:
        union(left, right)
    while len({find(node) for node in range(size)}) > 1:
        cross_component = (
            (similarities[left][right], left, right)
            for left in range(size)
            for right in range(left + 1, size)
            if find(left) != find(right)
        )
        similarity, left, right = max(cross_component)
        edge_weights[(left, right)] = max(float(similarity), 1e-6)
        union(left, right)

    adjacency = [[] for _ in range(size)]
    edges = []
    for (left, right), weight in sorted(edge_weights.items()):
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
        edges.append({"source": left, "target": right, "weight": weight})
    return adjacency, edges


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be at least 1")

    records = load_jsonl(args.input_jsonl)
    embeddings = encode_answer_representations(records, args)
    adjacency, edges = build_mutual_knn_graph(
        embeddings,
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
        "version": 2,
        "method": "answer_hidden_state_mutual_knn",
        "source": {
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
        "parameters": {
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "max_length": args.max_length,
            "pooling": "answer_token_mean_last_hidden_state",
            "mutual_knn": True,
            "isolated_node_backfill": True,
            "component_backfill": True,
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
        f"Wrote model-memory graph with {len(records)} nodes, {len(edges)} edges, "
        f"{len(community_sizes)} communities, degree {payload['degree_summary']} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
