#!/usr/bin/env python3
"""Create a degree- and weight-distribution-matched random graph ablation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_forget_graph import detect_communities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    return parser.parse_args()


def randomize_graph(payload, seed=0, swaps_per_edge=10, resolution=1.0):
    if swaps_per_edge < 1:
        raise ValueError("swaps_per_edge must be at least 1")
    num_nodes = int(payload["num_nodes"])
    original_edges = payload.get("edges", [])
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(edge["source"]), int(edge["target"])) for edge in original_edges
    )
    if graph.number_of_edges() != len(original_edges):
        raise ValueError("Input must be a simple graph without duplicate edges")

    swaps = swaps_per_edge * graph.number_of_edges()
    randomized = None
    for attempt in range(10):
        candidate = graph.copy()
        nx.double_edge_swap(
            candidate,
            nswap=swaps,
            max_tries=max(100, swaps * 100),
            seed=seed + attempt,
        )
        if not nx.is_connected(graph) or nx.is_connected(candidate):
            randomized = candidate
            break
    if randomized is None:
        raise RuntimeError("Could not produce a connected randomized graph")

    weights = [float(edge.get("weight", 1.0)) for edge in original_edges]
    random.Random(seed).shuffle(weights)
    pairs = sorted((min(left, right), max(left, right)) for left, right in randomized.edges)
    edges = [
        {"source": left, "target": right, "weight": weight}
        for (left, right), weight in zip(pairs, weights)
    ]
    communities, community_sizes = detect_communities(
        num_nodes=num_nodes,
        edges=edges,
        resolution=resolution,
        seed=seed,
    )

    result = dict(payload)
    result.update(
        {
            "version": max(2, int(payload.get("version", 1))),
            "method": f"degree_preserving_randomized_{payload.get('method', 'graph')}",
            "randomization": {
                "seed": seed,
                "swaps_per_edge": swaps_per_edge,
                "degree_preserved": True,
                "weight_distribution_preserved": True,
                "connectivity_preserved": nx.is_connected(graph),
            },
            "num_edges": len(edges),
            "num_communities": len(community_sizes),
            "communities": communities,
            "community_sizes": community_sizes,
            "edges": edges,
        }
    )
    return result


def main():
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    randomized = randomize_graph(
        payload,
        seed=args.seed,
        swaps_per_edge=args.swaps_per_edge,
        resolution=args.community_resolution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(randomized, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote {randomized['num_nodes']} nodes and {randomized['num_edges']} "
        f"degree-preserving randomized edges to {args.output}"
    )


if __name__ == "__main__":
    main()
