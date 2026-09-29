#!/usr/bin/env bash
set -euo pipefail

GRAPH_PATH="${GRAPH_PATH:-artifacts/graphs/forget10_tfidf_pagerank.json}"

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/graph_coverage_npo \
  model=Tiny-GPT2 \
  model.model_args.pretrained_model_name_or_path=sshleifer/tiny-gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=sshleifer/tiny-gpt2 \
  graph_weights_path="${GRAPH_PATH}" \
  '~eval.tofu' \
  trainer.args.do_eval=false \
  trainer.args.eval_on_start=false \
  trainer.args.eval_strategy=no \
  trainer.args.bf16=false \
  trainer.args.bf16_full_eval=false \
  trainer.args.optim=adamw_torch \
  trainer.args.report_to=none \
  trainer.args.warmup_epochs=null \
  trainer.args.per_device_train_batch_size=2 \
  trainer.args.gradient_accumulation_steps=1 \
  +trainer.args.max_steps=1 \
  task_name=GRAPH_COVERAGE_NPO_SMOKE
