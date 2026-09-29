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
`GraphCoverageNPO` is the next treatment and makes the graph signal dynamic:

- it tracks the per-token NLL margin to the frozen reference model for every
  forget example;
- an exponential residual prioritizes examples that still lag in forgetting;
- residuals are propagated to weighted graph neighbors;
- Louvain communities receive balanced total optimization mass; and
- an adaptive retain coefficient increases when retain NLL exceeds a fixed
  budget over the reference model.

The adaptive coefficient is updated once per optimizer step from the mean
violation over all gradient-accumulation micro-batches. This keeps its update
rate independent of `gradient_accumulation_steps`; the updated coefficient is
used starting with the next optimizer step.

The balanced configuration uses a `0.10` retain budget, caps the adaptive
coefficient at `2.0`, and reacts faster to current forgetting residuals. It is
intended to recover more forgetting than the initial high-utility run while
retaining its utility advantage.

The graph builder now stores deterministic Louvain community assignments. Old
artifacts are rebuilt automatically by the new run script.

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
  trainer.method_args.propagation_strength=0.2 \
  trainer.method_args.residual_temperature=0.5 \
  trainer.method_args.retain_budget=0.10
```

Set `TASK_NAME` when comparing variants so an existing result is not
overwritten:

```bash
TASK_NAME=GRAPH_COVERAGE_NPO_BALANCED_forget10_SEED0 \
  bash scripts/run_graph_coverage_npo_tofu.sh
```
