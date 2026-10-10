# GraphNPO feasibility experiment

This experiment tests one narrow question before implementing a causal
hypergraph method:

> Does graph-derived, per-example forgetting strength improve the
> forget/retain trade-off over a matched NPO run?

It is intentionally not presented as the final HyperCut method.

## Design

- Dataset: TOFU.
- Default model: `Llama-3.2-1B-Instruct`.
- Default split: `forget10` / `retain90`.
- Control: OpenUnlearning NPO.
- Treatment: GraphNPO.
- Only treatment difference: NPO's per-example forget loss is multiplied by a
  precomputed graph weight.
- Graph nodes: forget QA examples.
- Edge weights: TF-IDF cosine similarity over question plus answer.
- Node weights: weighted PageRank, clipped and renormalized to mean one.

## Run

Install OpenUnlearning according to its README, then run on a CUDA host:

```bash
python setup_data.py --eval_logs
bash scripts/run_graph_npo_tofu.sh
```

For a smaller split:

```bash
FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 \
  bash scripts/run_graph_npo_tofu.sh
```

Use `SEED=0`, `SEED=1`, and `SEED=2` for the final matched comparison. The
script also accepts `MODEL`, `MODEL_PATH`, `GRAPH_PATH`, and
`RETAIN_LOGS_PATH` overrides. It uses `adamw_torch` by default so the matched
runs do not depend on bitsandbytes; set `OPTIMIZER` to override it. On an
offline host, place TOFU JSONL files in `data/tofu_offline/` or set
`TOFU_LOCAL_DIR` to another directory.

The graph artifact is written under `artifacts/graphs/`. Both training jobs use
the same OpenUnlearning evaluation configuration.

For a one-step CPU smoke test with a tiny model:

```bash
bash scripts/smoke_graph_npo_tofu.sh
```

The smoke test only validates the integration and must not be reported as an
unlearning result.

## Decision rule

Continue to causal hyperedges only if GraphNPO improves forgetting under a
matched retention constraint, or improves retention at matched forgetting.
Do not select a method from a single scalar aggregate. Inspect at least:

- forget quality and forget-set probability;
- retain-set utility;
- real-author and world-fact utility;
- paraphrased forget queries;
- membership-inference behavior.

Run at least three seeds after hyperparameters have been fixed on a development
configuration.

## GraphCoverageNPO

The first matched run showed that static PageRank weights affected which
examples were forgotten, but did not improve the final trade-off over NPO.
`GraphCoverageNPO` makes the graph signal dynamic:

- it tracks the per-token NLL margin to the frozen reference model for every
  forget example;
- an exponential residual prioritizes examples that still lag in forgetting;
- residuals are propagated to weighted graph neighbors on the full graph;
- globally normalized node weights are refreshed at a fixed optimizer-step
  interval, rather than normalized independently inside every mini-batch; and
- an adaptive retain coefficient responds when retain NLL moves outside or
  back inside a fixed budget around the reference model.

The adaptive coefficient is updated once per optimizer step from the mean
violation over all gradient-accumulation micro-batches. This keeps its update
rate independent of `gradient_accumulation_steps`; the updated coefficient is
used starting with the next optimizer step. The default controller smooths the
violation with an exponential moving average and applies a clipped additive
update. Its lower bound is zero, so retain pressure can decrease again after
utility recovers instead of monotonically saturating at `alpha_max`. The old
multiplicative rule remains available through
`trainer.method_args.alpha_update_rule=multiplicative`.

The model-memory graph replaces TF-IDF similarity with the base model's own
answer-token representations. For each forget example, the builder mean-pools
the final hidden states over answer tokens, L2-normalizes the result, and builds
a mutual 8-nearest-neighbor graph. Isolated nodes receive their strongest edge.
This makes an edge describe similarity in the model's representation space
rather than surface word overlap.

Build the graph directly when needed:

```bash
python scripts/build_model_memory_graph.py \
  --input-jsonl data/tofu_offline/forget10.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --output artifacts/graphs/forget10_model_memory.json
```

The run script builds this artifact automatically when it is missing. Set
`GRAPH_TYPE=tfidf` to run a controlled ablation with the old graph.

Create a topology-control graph that preserves the exact node degrees and edge
weight distribution while randomly rewiring the edges:

```bash
python scripts/randomize_forget_graph.py \
  --input artifacts/graphs/forget10_model_memory.json \
  --output artifacts/graphs/forget10_model_memory_randomized.json \
  --seed 0
```

Running the same trainer with this artifact separates the effect of meaningful
model-memory neighborhoods from generic graph smoothing.

Build a gradient/influence graph from deterministic random projections of each
example's answer-NLL gradient with respect to the final model normalization
parameters:

```bash
python scripts/build_gradient_memory_graph.py \
  --input-jsonl data/tofu_offline/forget10.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --parameter-pattern model.norm \
  --projection-dim 256 \
  --output artifacts/graphs/forget10_gradient_memory.json
```

Only the graph artifact contains forget-set-derived information and remains
ignored by Git. The builder and its deterministic projection settings are
versioned for reproducibility.

Run only the new treatment (the completed NPO run remains the single baseline):

```bash
bash scripts/run_graph_coverage_npo_tofu.sh
```

For a one-step integration check:

```bash
bash scripts/smoke_graph_coverage_npo_tofu.sh
```

Useful method overrides include:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/graph_coverage_npo \
  trainer.method_args.propagation_strength=0.3 \
  trainer.method_args.propagation_interval=10 \
  trainer.method_args.residual_temperature=0.5 \
  trainer.method_args.retain_budget=0.10 \
  trainer.method_args.adaptive_alpha_lr=0.02 \
  trainer.method_args.retain_violation_ema_decay=0.9
```

Set `TASK_NAME` when comparing variants so an existing result is not
overwritten:

```bash
TASK_NAME=MODEL_MEMORY_GRAPH_NPO_forget10_SEED0 \
  bash scripts/run_graph_coverage_npo_tofu.sh
```

## Signed gradient-conflict graph

The signed-conflict treatment tests a stronger use of graph structure than
neighbor smoothing. It sketches answer-NLL gradients from every transformer
LayerNorm, then creates two edge types:

- a positive edge connects examples whose forgetting gradients cooperate;
- a negative edge connects examples whose forgetting gradients conflict.

During training, positive neighbors smooth forgetting demand in log-residual
space. Negative neighbors compete for update budget: the endpoint with the
larger current residual is amplified and the other is suppressed. This is a
deterministic allocation approximation to per-example gradient surgery that
does not require retaining a full backward graph for every example in a batch.

Build the private graph artifact:

```bash
python scripts/build_signed_gradient_graph.py \
  --input-jsonl data/tofu_offline/forget10.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --projection-dim 512 \
  --positive-top-k 4 \
  --negative-top-k 4 \
  --output artifacts/graphs/forget10_signed_gradient_conflict.json
```

Then run the treatment:

```bash
TASK_NAME=SIGNED_GRADIENT_CONFLICT_NPO_forget10_SEED0 \
  bash scripts/run_signed_gradient_conflict_npo_tofu.sh
```

Its required controls are the same trainer with `conflict_strength=0`, a
degree-preserving sign shuffle, and ordinary NPO. A signed-graph claim requires
the real graph to beat all controls across multiple seeds; a single TOFU run is
only a feasibility test.

## Internal KV causal probe

Before training another graph variant, test whether an MLP key/value graph
predicts the collateral effect of a memory edit. The probe extracts the
answer-token input and output of a selected `mlp.down_proj`, temporarily
applies a rank-one value-erasure hook for each sampled forget example, and
measures the exact NLL change on sampled forget and retain examples. It does
not modify or save the model.

```bash
python scripts/probe_kv_causal_graph.py \
  --forget-jsonl data/tofu_offline/forget10.json \
  --retain-jsonl data/tofu_offline/retain90.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --layer 3 --num-sources 32 \
  --num-forget-targets 128 --num-retain-targets 128 \
  --intervention-strength 0.5 \
  --output artifacts/kv_probe/forget10_layer3_seed0.json
```

Compare key cosine, value cosine, their product, and final-hidden-state cosine
by per-source Spearman correlation and precision at the largest eight causal
effects. Only layers that beat final-hidden and random retrieval consistently
should be used to build the training graph.

Build the selected full graph as follows:

```bash
python scripts/build_kv_memory_graph.py \
  --input-jsonl data/tofu_offline/forget10.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --layer 3 \
  --output artifacts/graphs/forget10_mlp_key_layer3.json
```

## Graph-constrained low-rank KV edit

The causal probe motivates using the graph in parameter space rather than as
an NPO example-weighting heuristic. The direct editor solves a ridge-regression
update to one MLP `down_proj`. Forget keys are mapped toward the negative of
their current values, sampled retain keys are zero-response anchors, and every
graph edge adds a Laplacian column that penalizes different update responses
for neighboring keys. The dense solution is truncated to the requested rank
before being applied.

```bash
python scripts/apply_graph_kv_edit.py \
  --forget-jsonl data/tofu_offline/forget10.json \
  --retain-jsonl data/tofu_offline/retain90.json \
  --graph artifacts/graphs/forget10_mlp_key_layer3.json \
  --model-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --output-dir saves/unlearn/KV_DIRECT_GRAPH_L3_S080_G1_R64 \
  --layer 3 --strength 0.8 \
  --retain-weight 1.0 --graph-gamma 1.0 \
  --ridge-scale 1e-3 --rank 64 \
  --num-retain-anchors 400
```

The mandatory ablation uses the same command and anchors with
`--graph-gamma 0`. Compare graph and no-graph at the same edit strength, and
also compare interpolated utility at matched forget probability. Strength is a
development hyperparameter; do not select it on the final test split.

An experimental alternative maps the last pre-generation prompt key toward
the value induced by a refusal system prompt instead of erasing answer-token
values. Enable it with `--edit-target refusal_prompt_value`; the exact refusal
instruction can be changed with `--refusal-system-prompt`. This target should
be treated as an ablation: matching a local prompt value does not guarantee
that the generated answer will follow the refusal instruction.

A stronger answer-side alternative uses
`--edit-target counterfactual_refusal_value`. It keeps each forget question
fixed, replaces its answer with a shared refusal string, and maps the factual
answer key toward the paired difference between refusal and factual values.
Override the default refusal with `--refusal-answer`. Screen this target first
with `--graph-gamma 0`; only add the graph after it improves the erase target
at matched forget probability.

The interpolation experiment uses `--edit-target hybrid_refusal_value` and
constructs `-v_factual + beta*v_refusal`. Set beta with `--refusal-mix`; beta
zero recovers pure erasure and beta one recovers the counterfactual target.
Sweep beta at fixed strength before adding graph regularization.

The output-gradient experiment uses
`--edit-target contrastive_output_gradient`. At the causal positions whose
logits predict answer tokens, it backpropagates factual and refusal NLL through
the upper model. The target follows factual-loss ascent minus refusal-loss
descent, normalized to the factual value norm. Control the refusal term with
`--refusal-gradient-weight` and the activation norm with
`--gradient-target-scale`. Screen this target without graph regularization.
