#!/usr/bin/env python3
"""Apply a graph-constrained low-rank edit to one Llama MLP down projection."""

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
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument(
        "--edit-target",
        choices=(
            "erase_answer_value",
            "counterfactual_refusal_value",
            "hybrid_refusal_value",
            "contrastive_output_gradient",
            "refusal_prompt_value",
        ),
        default="erase_answer_value",
    )
    parser.add_argument(
        "--refusal-answer",
        default="I don't know the answer to that question.",
    )
    parser.add_argument(
        "--refusal-mix",
        type=float,
        default=0.5,
        help="Beta in hybrid target -v_fact + beta*v_refusal.",
    )
    parser.add_argument("--refusal-gradient-weight", type=float, default=1.0)
    parser.add_argument("--gradient-target-scale", type=float, default=1.0)
    parser.add_argument(
        "--refusal-system-prompt",
        default=(
            "You are a helpful assistant. You do not know biographical "
            "information about the person in the user's question. Say that "
            "you do not know and do not provide factual details."
        ),
    )
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--graph-gamma", type=float, default=1.0)
    parser.add_argument("--ridge-scale", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num-retain-anchors", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def select_indices(size, count, seed):
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return sorted(indices[: min(size, count)])


def replace_answers(records, answer_key, answer):
    """Return record copies with one shared counterfactual answer."""
    counterfactual = []
    for record in records:
        if answer_key not in record:
            raise KeyError(f"Record lacks '{answer_key}'")
        updated = dict(record)
        updated[answer_key] = answer
        counterfactual.append(updated)
    return counterfactual


def answer_value_target(factual_values, refusal_values, edit_target, refusal_mix):
    """Construct erase, counterfactual, or interpolated answer-value targets."""
    if edit_target == "erase_answer_value":
        return -factual_values
    if refusal_values is None:
        raise ValueError("refusal values are required for a refusal answer target")
    if edit_target == "counterfactual_refusal_value":
        return refusal_values - factual_values
    if edit_target == "hybrid_refusal_value":
        return -factual_values + refusal_mix * refusal_values
    raise ValueError(f"unsupported answer value target: {edit_target}")


def masked_mean_pool(activations, mask):
    mask = mask.unsqueeze(-1)
    counts = mask.sum(dim=1).clamp_min(1)
    return (activations.float() * mask).sum(dim=1) / counts


def answer_mean_pool(activations, labels):
    return masked_mean_pool(activations, labels.ne(-100))


def shifted_supervision_mask(labels):
    """Positions whose logits predict supervised tokens after causal shifting."""
    import torch

    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:].ne(-100)
    return mask


def extract_kv(
    model,
    tokenizer,
    records,
    module,
    args,
    include_values,
    prediction_positions=False,
):
    import torch
    from torch.nn.utils.rnn import pad_sequence

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
    captured = {}

    def capture(_module, inputs, output):
        captured["key"] = inputs[0].detach()
        if include_values:
            captured["value"] = output.detach()

    keys, values = [], []
    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for start in range(0, len(tokenized), args.batch_size):
                batch = tokenized[start : start + args.batch_size]
                input_ids = pad_sequence(
                    [torch.tensor(item[0]) for item in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(args.device)
                labels = pad_sequence(
                    [torch.tensor(item[1]) for item in batch],
                    batch_first=True,
                    padding_value=-100,
                ).to(args.device)
                attention_mask = pad_sequence(
                    [torch.ones(len(item[0]), dtype=torch.long) for item in batch],
                    batch_first=True,
                    padding_value=0,
                ).to(args.device)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                mask = (
                    shifted_supervision_mask(labels)
                    if prediction_positions
                    else labels.ne(-100)
                )
                keys.append(masked_mean_pool(captured["key"], mask).cpu())
                if include_values:
                    values.append(masked_mean_pool(captured["value"], mask).cpu())
    finally:
        handle.remove()
    return torch.cat(keys), torch.cat(values) if values else None


def extract_output_gradients(model, tokenizer, records, module, args):
    """Extract VJP directions at positions that predict supervised answers."""
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence

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
    captured = {}

    def capture(_module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        captured["key"] = inputs[0].detach()
        captured["value"] = output.detach()
        captured["leaf"] = leaf
        return leaf

    keys, values, gradients = [], [], []
    handle = module.register_forward_hook(capture)
    try:
        for start in range(0, len(tokenized), args.batch_size):
            batch = tokenized[start : start + args.batch_size]
            input_ids = pad_sequence(
                [torch.tensor(item[0]) for item in batch],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(args.device)
            labels = pad_sequence(
                [torch.tensor(item[1]) for item in batch],
                batch_first=True,
                padding_value=-100,
            ).to(args.device)
            attention_mask = pad_sequence(
                [torch.ones(len(item[0]), dtype=torch.long) for item in batch],
                batch_first=True,
                padding_value=0,
            ).to(args.device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            shifted_labels = labels[:, 1:]
            loss = functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss.backward()
            mask = shifted_supervision_mask(labels)
            keys.append(masked_mean_pool(captured["key"], mask).cpu())
            values.append(masked_mean_pool(captured["value"], mask).cpu())
            gradients.append(masked_mean_pool(captured["leaf"].grad, mask).cpu())
    finally:
        handle.remove()
    return torch.cat(keys), torch.cat(values), torch.cat(gradients)


def contrastive_gradient_target(
    factual_gradients,
    refusal_gradients,
    factual_values,
    refusal_weight,
    target_scale,
):
    """Build an activation target from factual-loss ascent and refusal-loss descent."""
    factual_unit = factual_gradients / factual_gradients.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    refusal_unit = refusal_gradients / refusal_gradients.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    direction = factual_unit - refusal_weight * refusal_unit
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    target_norm = target_scale * factual_values.norm(dim=1, keepdim=True)
    return target_norm * direction


def _prompt_token_ids(tokenizer, record, args, system_prompt):
    if args.question_key not in record:
        raise KeyError(f"Record lacks '{args.question_key}'")
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": str(record[args.question_key])})
    date_info = {"date_string": args.date_string} if args.date_string else {}
    input_ids = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        **date_info,
    )[: args.max_length]
    if not input_ids:
        raise ValueError("Prompt tokenization produced an empty sequence")
    return input_ids


def extract_prompt_kv(model, tokenizer, records, module, args, system_prompt):
    """Extract the last pre-generation MLP key/value for every question."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    tokenized = [
        _prompt_token_ids(tokenizer, record, args, system_prompt) for record in records
    ]
    captured = {}

    def capture(_module, inputs, output):
        captured["key"] = inputs[0].detach()
        captured["value"] = output.detach()

    keys, values = [], []
    handle = module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for start in range(0, len(tokenized), args.batch_size):
                batch = tokenized[start : start + args.batch_size]
                lengths = torch.tensor(
                    [len(item) for item in batch], device=args.device, dtype=torch.long
                )
                input_ids = pad_sequence(
                    [torch.tensor(item) for item in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(args.device)
                attention_mask = pad_sequence(
                    [torch.ones(len(item), dtype=torch.long) for item in batch],
                    batch_first=True,
                    padding_value=0,
                ).to(args.device)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                rows = torch.arange(len(batch), device=args.device)
                positions = lengths - 1
                keys.append(captured["key"][rows, positions].float().cpu())
                values.append(captured["value"][rows, positions].float().cpu())
    finally:
        handle.remove()
    return torch.cat(keys), torch.cat(values)


def load_graph_edges(path, num_nodes):
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("num_nodes", -1)) != num_nodes:
        raise ValueError("graph node count does not match the forget set")
    edges = []
    for edge in payload.get("edges", []):
        left, right = int(edge["source"]), int(edge["target"])
        weight = float(edge["weight"])
        if left == right or not 0 <= left < num_nodes or not 0 <= right < num_nodes:
            raise ValueError("graph contains an invalid edge")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("graph edge weights must be positive and finite")
        edges.append((left, right, weight))
    return edges


def graph_difference_columns(keys, edges, gamma):
    """Return sqrt(gamma*w)*(k_i-k_j) columns for a Laplacian penalty."""
    import torch

    if gamma <= 0 or not edges:
        return keys.new_empty((keys.shape[1], 0))
    columns = [
        math.sqrt(gamma * weight) * (keys[left] - keys[right])
        for left, right, weight in edges
    ]
    return torch.stack(columns, dim=1)


def truncated_product(left, right, rank):
    """Best rank-r approximation of left @ right using a small core SVD."""
    import torch

    max_rank = min(left.shape[0], left.shape[1], right.shape[0], right.shape[1])
    rank = min(rank, max_rank)
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular, vh_core = torch.linalg.svd(core, full_matrices=False)
    u = q_left @ u_core[:, :rank]
    v = q_right @ vh_core[:rank].T
    delta = (u * singular[:rank].unsqueeze(0)) @ v.T
    explained = singular[:rank].square().sum() / singular.square().sum().clamp_min(1e-30)
    return delta, singular, float(explained)


def solve_edit(
    forget_keys,
    desired_deltas,
    retain_keys,
    edges,
    strength,
    retain_weight,
    graph_gamma,
    ridge_scale,
    rank,
    device,
):
    """Solve ridge regression with retain anchors and a graph Laplacian."""
    import torch

    forget_keys = forget_keys.to(device=device, dtype=torch.float32)
    desired_deltas = desired_deltas.to(device=device, dtype=torch.float32)
    retain_keys = retain_keys.to(device=device, dtype=torch.float32)
    forget_norms = forget_keys.norm(dim=1).clamp_min(1e-6)
    retain_norms = retain_keys.norm(dim=1).clamp_min(1e-6)
    forget_unit = forget_keys / forget_norms.unsqueeze(1)
    retain_unit = retain_keys / retain_norms.unsqueeze(1)

    graph_columns = graph_difference_columns(forget_unit, edges, graph_gamma)
    design = torch.cat(
        [
            forget_unit.T,
            math.sqrt(retain_weight) * retain_unit.T,
            graph_columns,
        ],
        dim=1,
    )
    target = strength * (desired_deltas / forget_norms.unsqueeze(1)).T
    gram = design.T @ design
    ridge = ridge_scale * gram.diagonal().mean().clamp_min(1e-12)
    gram.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(gram)
    coefficient = torch.cholesky_solve(design.T, factor)
    # Only forget targets are nonzero in Y=[D_f, 0, 0].
    right = coefficient[: len(forget_keys)]
    delta, singular, explained = truncated_product(target, right, rank)

    achieved_forget = (delta @ forget_keys.T).T
    desired_forget = strength * desired_deltas
    achieved_retain = (delta @ retain_keys.T).T
    fit_error = (achieved_forget - desired_forget).norm() / desired_forget.norm().clamp_min(1e-30)
    retain_response = achieved_retain.norm(dim=1)
    forget_response = achieved_forget.norm(dim=1)
    diagnostics = {
        "ridge": float(ridge),
        "design_columns": design.shape[1],
        "graph_columns": graph_columns.shape[1],
        "rank": min(rank, len(singular)),
        "rank_explained_frobenius": explained,
        "relative_forget_fit_error": float(fit_error),
        "forget_response_norm_mean": float(forget_response.mean()),
        "retain_response_norm_mean": float(retain_response.mean()),
        "retain_to_forget_response_ratio": float(
            retain_response.mean() / forget_response.mean().clamp_min(1e-30)
        ),
        "top_singular_values": [float(value) for value in singular[:10]],
    }
    return delta, diagnostics


def main():
    args = parse_args()
    if not 0 < args.strength <= 1:
        raise ValueError("--strength must be in (0, 1]")
    if args.retain_weight < 0 or args.graph_gamma < 0 or args.ridge_scale <= 0:
        raise ValueError("regularization weights must be non-negative and ridge positive")
    if args.rank < 1 or args.num_retain_anchors < 1:
        raise ValueError("rank and retain-anchor count must be positive")
    if not 0 <= args.refusal_mix <= 1:
        raise ValueError("--refusal-mix must be in [0, 1]")
    if args.refusal_gradient_weight < 0 or args.gradient_target_scale <= 0:
        raise ValueError("gradient weight must be non-negative and scale positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    model.config.use_cache = False
    if args.layer < 0 or args.layer >= len(model.model.layers):
        raise ValueError(f"layer {args.layer} is outside the model")
    module = model.model.layers[args.layer].mlp.down_proj

    forget_records = load_jsonl(args.forget_jsonl)
    retain_records = load_jsonl(args.retain_jsonl)
    retain_indices = select_indices(
        len(retain_records), args.num_retain_anchors, args.seed
    )
    answer_targets = (
        "erase_answer_value",
        "counterfactual_refusal_value",
        "hybrid_refusal_value",
    )
    if args.edit_target in answer_targets:
        forget_keys, forget_values = extract_kv(
            model, tokenizer, forget_records, module, args, include_values=True
        )
        refusal_values = None
        if args.edit_target != "erase_answer_value":
            counterfactual_records = replace_answers(
                forget_records, args.answer_key, args.refusal_answer
            )
            _, refusal_values = extract_kv(
                model,
                tokenizer,
                counterfactual_records,
                module,
                args,
                include_values=True,
            )
        desired_deltas = answer_value_target(
            forget_values,
            refusal_values,
            args.edit_target,
            args.refusal_mix,
        )
        retain_keys, _ = extract_kv(
            model,
            tokenizer,
            [retain_records[index] for index in retain_indices],
            module,
            args,
            include_values=False,
        )
    elif args.edit_target == "contrastive_output_gradient":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        forget_keys, forget_values, factual_gradients = extract_output_gradients(
            model, tokenizer, forget_records, module, args
        )
        counterfactual_records = replace_answers(
            forget_records, args.answer_key, args.refusal_answer
        )
        _, _, refusal_gradients = extract_output_gradients(
            model, tokenizer, counterfactual_records, module, args
        )
        desired_deltas = contrastive_gradient_target(
            factual_gradients,
            refusal_gradients,
            forget_values,
            args.refusal_gradient_weight,
            args.gradient_target_scale,
        )
        retain_keys, _ = extract_kv(
            model,
            tokenizer,
            [retain_records[index] for index in retain_indices],
            module,
            args,
            include_values=False,
            prediction_positions=True,
        )
    else:
        forget_keys, original_values = extract_prompt_kv(
            model,
            tokenizer,
            forget_records,
            module,
            args,
            system_prompt=args.system_prompt,
        )
        _, refusal_values = extract_prompt_kv(
            model,
            tokenizer,
            forget_records,
            module,
            args,
            system_prompt=args.refusal_system_prompt,
        )
        desired_deltas = refusal_values - original_values
        retain_keys, _ = extract_prompt_kv(
            model,
            tokenizer,
            [retain_records[index] for index in retain_indices],
            module,
            args,
            system_prompt=args.system_prompt,
        )
    edges = load_graph_edges(args.graph, len(forget_records))
    delta, diagnostics = solve_edit(
        forget_keys=forget_keys,
        desired_deltas=desired_deltas,
        retain_keys=retain_keys,
        edges=edges,
        strength=args.strength,
        retain_weight=args.retain_weight,
        graph_gamma=args.graph_gamma,
        ridge_scale=args.ridge_scale,
        rank=args.rank,
        device=args.device,
    )
    with torch.no_grad():
        module.weight.add_(delta.to(dtype=module.weight.dtype))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "method": "graph_constrained_low_rank_kv_edit",
        "model_path": args.model_path,
        "forget_jsonl": str(args.forget_jsonl),
        "retain_jsonl": str(args.retain_jsonl),
        "graph": str(args.graph) if args.graph else None,
        "layer": args.layer,
        "module": f"model.layers.{args.layer}.mlp.down_proj",
        "strength": args.strength,
        "edit_target": args.edit_target,
        "refusal_answer": (
            args.refusal_answer
            if args.edit_target
            in ("counterfactual_refusal_value", "hybrid_refusal_value")
            else None
        ),
        "refusal_mix": (
            args.refusal_mix if args.edit_target == "hybrid_refusal_value" else None
        ),
        "refusal_gradient_weight": (
            args.refusal_gradient_weight
            if args.edit_target == "contrastive_output_gradient"
            else None
        ),
        "gradient_target_scale": (
            args.gradient_target_scale
            if args.edit_target == "contrastive_output_gradient"
            else None
        ),
        "refusal_system_prompt": (
            args.refusal_system_prompt
            if args.edit_target == "refusal_prompt_value"
            else None
        ),
        "retain_weight": args.retain_weight,
        "graph_gamma": args.graph_gamma,
        "ridge_scale": args.ridge_scale,
        "rank": args.rank,
        "num_forget": len(forget_records),
        "num_retain_anchors": len(retain_indices),
        "retain_indices": retain_indices,
        "diagnostics": diagnostics,
    }
    with (args.output_dir / "kv_edit_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved edited model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
