#!/usr/bin/env python3
"""Build a lightweight forget-set graph and export normalized node weights.

This is deliberately a dependency-light feasibility baseline.  Nodes are
forget examples, weighted edges are TF-IDF cosine similarities, and node
weights are weighted PageRank scores normalized to mean one.  The emitted
indices match OpenUnlearning's QADataset indices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--dataset-path")
    parser.add_argument("--dataset-name", default="forget10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-similarity", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--weight-floor", type=float, default=0.25)
    parser.add_argument("--weight-ceiling", type=float, default=4.0)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(args):
    if args.input_jsonl is not None:
        with args.input_jsonl.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face input requires the 'datasets' package. Install the "
            "OpenUnlearning requirements or use --input-jsonl."
        ) from exc

    dataset = load_dataset(
        args.dataset_path,
        name=args.dataset_name,
        split=args.split,
    )
    return [dict(record) for record in dataset]


def tokenize(text: str):
    return TOKEN_PATTERN.findall(text.lower())


def make_documents(records, question_key, answer_key):
    documents = []
    for position, record in enumerate(records):
        if question_key not in record or answer_key not in record:
            raise KeyError(
                f"Record {position} lacks '{question_key}' or '{answer_key}'"
            )
        answer = record[answer_key]
        if isinstance(answer, list):
            answer = " ".join(map(str, answer))
        documents.append(f"{record[question_key]} {answer}")
    return documents


def tfidf_vectors(documents: Iterable[str]):
    token_counts = [Counter(tokenize(document)) for document in documents]
    document_frequency = Counter()
    for counts in token_counts:
        document_frequency.update(counts.keys())

    total = len(token_counts)
    vectors = []
    for counts in token_counts:
        vector = {
            token: (1.0 + math.log(count))
            * (math.log((1.0 + total) / (1.0 + document_frequency[token])) + 1.0)
            for token, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({token: value / norm for token, value in vector.items()})
    return vectors


def cosine(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def build_knn_graph(vectors, top_k, min_similarity):
    size = len(vectors)
    if size < 2:
        return [[] for _ in range(size)], []
    top_k = max(1, min(top_k, size - 1))
    directed = [[] for _ in range(size)]
    for left in range(size):
        candidates = []
        for right in range(size):
            if left == right:
                continue
            similarity = cosine(vectors[left], vectors[right])
            if similarity >= min_similarity:
                candidates.append((right, similarity))
        directed[left] = sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]

    undirected = {}
    for left, neighbors in enumerate(directed):
        for right, weight in neighbors:
            key = (min(left, right), max(left, right))
            undirected[key] = max(weight, undirected.get(key, 0.0))

    adjacency = [[] for _ in range(size)]
    edges = []
    for (left, right), weight in sorted(undirected.items()):
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
        edges.append({"source": left, "target": right, "weight": weight})
    return adjacency, edges


def weighted_pagerank(adjacency, damping=0.85, max_iterations=200, tolerance=1e-10):
    size = len(adjacency)
    if size == 0:
        return []
    scores = [1.0 / size] * size
    outgoing = [sum(weight for _, weight in neighbors) for neighbors in adjacency]

    for _ in range(max_iterations):
        dangling_mass = sum(scores[i] for i, total in enumerate(outgoing) if total == 0)
        updated = [(1.0 - damping) / size + damping * dangling_mass / size] * size
        for source, neighbors in enumerate(adjacency):
            if outgoing[source] == 0:
                continue
            for target, weight in neighbors:
                updated[target] += damping * scores[source] * weight / outgoing[source]
        delta = sum(abs(new - old) for new, old in zip(updated, scores))
        scores = updated
        if delta < tolerance:
            break
    return scores


def normalize_weights(scores, floor, ceiling):
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    normalized = [min(ceiling, max(floor, score / mean)) for score in scores]
    # Clipping changes the mean; restore it so GraphNPO matches NPO's loss scale.
    clipped_mean = sum(normalized) / len(normalized)
    return [weight / clipped_mean for weight in normalized]


def detect_communities(num_nodes, edges, resolution=1.0, seed=0):
    """Return deterministic Louvain community labels for every graph node."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_weighted_edges_from(
        (edge["source"], edge["target"], edge["weight"]) for edge in edges
    )
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    communities = sorted((sorted(group) for group in communities), key=lambda x: x[0])
    labels = {
        str(node): community_id
        for community_id, group in enumerate(communities)
        for node in group
    }
    sizes = {
        str(community_id): len(group)
        for community_id, group in enumerate(communities)
    }
    return labels, sizes


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not 0.0 <= args.damping < 1.0:
        raise ValueError("--damping must be in [0, 1)")
    if not 0.0 < args.weight_floor <= args.weight_ceiling:
        raise ValueError("Weight bounds must satisfy 0 < floor <= ceiling")
    if args.community_resolution <= 0:
        raise ValueError("--community-resolution must be positive")

    records = load_records(args)
    documents = make_documents(records, args.question_key, args.answer_key)
    vectors = tfidf_vectors(documents)
    adjacency, edges = build_knn_graph(
        vectors=vectors,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    scores = weighted_pagerank(
        adjacency=adjacency,
        damping=args.damping,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    weights = normalize_weights(scores, args.weight_floor, args.weight_ceiling)
    communities, community_sizes = detect_communities(
        num_nodes=len(records),
        edges=edges,
        resolution=args.community_resolution,
        seed=args.community_seed,
    )

    payload = {
        "version": 1,
        "method": "tfidf_weighted_pagerank",
        "source": {
            "input_jsonl": str(args.input_jsonl) if args.input_jsonl else None,
            "dataset_path": args.dataset_path,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "question_key": args.question_key,
            "answer_key": args.answer_key,
        },
        "parameters": {
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "damping": args.damping,
            "weight_floor": args.weight_floor,
            "weight_ceiling": args.weight_ceiling,
            "community_resolution": args.community_resolution,
            "community_seed": args.community_seed,
        },
        "num_nodes": len(records),
        "num_edges": len(edges),
        "num_communities": len(community_sizes),
        "weights": {str(index): weight for index, weight in enumerate(weights)},
        "pagerank": {str(index): score for index, score in enumerate(scores)},
        "communities": communities,
        "community_sizes": community_sizes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Wrote {len(records)} nodes, {len(edges)} edges, "
        f"{len(community_sizes)} communities, and {len(weights)} weights "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
