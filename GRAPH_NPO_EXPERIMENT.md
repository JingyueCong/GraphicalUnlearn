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
`RETAIN_LOGS_PATH` overrides.

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
