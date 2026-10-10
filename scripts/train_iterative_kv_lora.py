#!/usr/bin/env python3
"""Train and merge a low-rank KV edit with iterative contrastive objectives."""

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

from apply_graph_kv_edit import replace_answers
from build_model_memory_graph import load_jsonl, tokenize_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget-jsonl", type=Path, required=True)
    parser.add_argument("--retain-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--forget-beta", type=float, default=1.0)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--refusal-weight", type=float, default=0.5)
    parser.add_argument("--retain-distill-weight", type=float, default=5.0)
    parser.add_argument("--retain-activation-weight", type=float, default=1.0)
    parser.add_argument("--factor-l2-weight", type=float, default=1e-5)
    parser.add_argument("--retain-top-k", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-retain-anchors", type=int, default=400)
    parser.add_argument("--refusal-answer", default="I don't know the answer to that question.")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--date-string", default="10 Apr 2025")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


class LowRankOutputEdit:
    """Add B(Ax) to a linear module output without modifying base weights."""

    def __init__(self, in_features, out_features, rank, device, seed):
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.a = torch.nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=torch.float32)
        )
        self.b = torch.nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=torch.float32)
        )
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=1.0 / math.sqrt(in_features), generator=generator)
        self.enabled = True
        self.last_update = None

    def parameters(self):
        return [self.a, self.b]

    def __call__(self, _module, inputs, output):
        import torch.nn.functional as functional

        if not self.enabled:
            self.last_update = None
            return output
        hidden = functional.linear(inputs[0].float(), self.a)
        self.last_update = functional.linear(hidden, self.b)
        return output + self.last_update.to(dtype=output.dtype)

    def merged_weight(self):
        return self.b @ self.a


def per_example_nll(logits, labels):
    import torch.nn.functional as functional

    shifted_labels = labels[:, 1:]
    losses = functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(-100)
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def bounded_forget_loss(current_nll, reference_nll, beta):
    import torch.nn.functional as functional

    return functional.softplus(-beta * (current_nll - reference_nll)).mean() / beta


def topk_logit_distillation(current_logits, reference_logits, labels, top_k):
    """Distill the reference distribution over its top-k answer candidates."""
    import torch
    import torch.nn.functional as functional

    mask = labels[:, 1:].ne(-100)
    current = current_logits[:, :-1].float()[mask]
    reference = reference_logits[:, :-1].float()[mask]
    if not len(current):
        return current_logits.new_zeros(())
    top_k = min(top_k, reference.shape[-1])
    indices = reference.topk(top_k, dim=-1).indices
    reference_top = reference.gather(-1, indices)
    current_top = current.gather(-1, indices)
    reference_prob = functional.softmax(reference_top, dim=-1)
    return functional.kl_div(
        functional.log_softmax(current_top, dim=-1),
        reference_prob,
        reduction="batchmean",
    )


def tokenize_records(tokenizer, records, args):
    return [
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


def collate(tokenized, indices, pad_token_id, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    batch = [tokenized[index] for index in indices]
    input_ids = pad_sequence(
        [torch.tensor(item[0]) for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
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
    return input_ids, attention_mask, labels


def sample_indices(rng, size, count):
    if size < count:
        return [rng.randrange(size) for _ in range(count)]
    return rng.sample(range(size), count)


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.rank < 1 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("rank, steps, and batch size must be positive")
    if args.learning_rate <= 0 or args.forget_beta <= 0:
        raise ValueError("learning rate and forget beta must be positive")
    if min(
        args.forget_weight,
        args.refusal_weight,
        args.retain_distill_weight,
        args.retain_activation_weight,
        args.factor_l2_weight,
    ) < 0:
        raise ValueError("objective weights must be non-negative")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.layer < 0 or args.layer >= len(model.model.layers):
        raise ValueError(f"layer {args.layer} is outside the model")
    module = model.model.layers[args.layer].mlp.down_proj
    edit = LowRankOutputEdit(
        module.in_features,
        module.out_features,
        args.rank,
        args.device,
        args.seed,
    )
    hook = module.register_forward_hook(edit)
    optimizer = torch.optim.AdamW(
        edit.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    forget_records = load_jsonl(args.forget_jsonl)
    retain_records = load_jsonl(args.retain_jsonl)
    retain_indices = sample_indices(
        rng, len(retain_records), min(args.num_retain_anchors, len(retain_records))
    )
    retain_records = [retain_records[index] for index in retain_indices]
    refusal_records = replace_answers(
        forget_records, args.answer_key, args.refusal_answer
    )
    factual_tokens = tokenize_records(tokenizer, forget_records, args)
    refusal_tokens = tokenize_records(tokenizer, refusal_records, args)
    retain_tokens = tokenize_records(tokenizer, retain_records, args)

    history = []
    try:
        for step in range(1, args.steps + 1):
            forget_batch = sample_indices(
                rng, len(factual_tokens), args.batch_size
            )
            retain_batch = sample_indices(rng, len(retain_tokens), args.batch_size)
            factual = collate(
                factual_tokens,
                forget_batch,
                tokenizer.pad_token_id,
                args.device,
            )
            refusal = collate(
                refusal_tokens,
                forget_batch,
                tokenizer.pad_token_id,
                args.device,
            )
            retain = collate(
                retain_tokens,
                retain_batch,
                tokenizer.pad_token_id,
                args.device,
            )
            optimizer.zero_grad(set_to_none=True)

            edit.enabled = False
            with torch.no_grad():
                factual_reference = model(
                    input_ids=factual[0], attention_mask=factual[1], use_cache=False
                ).logits
                factual_reference_nll = per_example_nll(
                    factual_reference, factual[2]
                )
            edit.enabled = True
            factual_logits = model(
                input_ids=factual[0], attention_mask=factual[1], use_cache=False
            ).logits
            factual_nll = per_example_nll(factual_logits, factual[2])
            forget_loss = bounded_forget_loss(
                factual_nll, factual_reference_nll, args.forget_beta
            )
            (args.forget_weight * forget_loss).backward()
            del factual_logits, factual_reference

            refusal_logits = model(
                input_ids=refusal[0], attention_mask=refusal[1], use_cache=False
            ).logits
            refusal_loss = per_example_nll(refusal_logits, refusal[2]).mean()
            (args.refusal_weight * refusal_loss).backward()
            del refusal_logits

            edit.enabled = False
            with torch.no_grad():
                retain_reference = model(
                    input_ids=retain[0], attention_mask=retain[1], use_cache=False
                ).logits
            edit.enabled = True
            retain_logits = model(
                input_ids=retain[0], attention_mask=retain[1], use_cache=False
            ).logits
            retain_distill = topk_logit_distillation(
                retain_logits, retain_reference, retain[2], args.retain_top_k
            )
            retain_activation = edit.last_update.float().square().mean()
            factor_l2 = edit.a.square().mean() + edit.b.square().mean()
            regularization = (
                args.retain_distill_weight * retain_distill
                + args.retain_activation_weight * retain_activation
                + args.factor_l2_weight * factor_l2
            )
            regularization.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                edit.parameters(), args.max_grad_norm
            )
            optimizer.step()
            del retain_logits, retain_reference

            record = {
                "step": step,
                "forget_loss": float(forget_loss.detach()),
                "factual_nll": float(factual_nll.mean().detach()),
                "factual_reference_nll": float(factual_reference_nll.mean()),
                "refusal_loss": float(refusal_loss.detach()),
                "retain_distill": float(retain_distill.detach()),
                "retain_activation": float(retain_activation.detach()),
                "grad_norm": float(grad_norm),
            }
            history.append(record)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps(record), flush=True)
    finally:
        hook.remove()

    delta = edit.merged_weight().detach()
    with torch.no_grad():
        module.weight.add_(delta.to(dtype=module.weight.dtype))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "method": "iterative_contrastive_low_rank_kv_edit",
        "model_path": args.model_path,
        "forget_jsonl": str(args.forget_jsonl),
        "retain_jsonl": str(args.retain_jsonl),
        "layer": args.layer,
        "rank": args.rank,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "forget_beta": args.forget_beta,
        "forget_weight": args.forget_weight,
        "refusal_weight": args.refusal_weight,
        "retain_distill_weight": args.retain_distill_weight,
        "retain_activation_weight": args.retain_activation_weight,
        "factor_l2_weight": args.factor_l2_weight,
        "retain_top_k": args.retain_top_k,
        "refusal_answer": args.refusal_answer,
        "num_forget": len(forget_records),
        "num_retain_anchors": len(retain_records),
        "delta_frobenius_norm": float(delta.norm()),
        "history": history,
    }
    with (args.output_dir / "iterative_kv_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Saved edited model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
