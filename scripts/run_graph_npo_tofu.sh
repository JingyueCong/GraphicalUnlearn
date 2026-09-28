#!/usr/bin/env bash
set -euo pipefail

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
GRAPH_PATH="${GRAPH_PATH:-artifacts/graphs/${FORGET_SPLIT}_tfidf_pagerank.json}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL}_full}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
SEED="${SEED:-0}"

if [[ ! -f "${RETAIN_LOGS_PATH}" ]]; then
  echo "Missing retain-model evaluation log: ${RETAIN_LOGS_PATH}" >&2
  echo "Run 'python setup_data.py --eval_logs' before the matched experiment." >&2
  exit 1
fi

python scripts/build_forget_graph.py \
  --dataset-path locuslab/TOFU \
  --dataset-name "${FORGET_SPLIT}" \
  --split train \
  --output "${GRAPH_PATH}"

# Matched baseline: same model, data split, and optimizer settings.
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model="${MODEL}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  trainer=NPO \
  forget_split="${FORGET_SPLIT}" \
  retain_split="${RETAIN_SPLIT}" \
  retain_logs_path="${RETAIN_LOGS_PATH}" \
  trainer.args.seed="${SEED}" \
  task_name="NPO_BASELINE_${FORGET_SPLIT}_SEED${SEED}"

# Treatment: only the per-example forget strength changes.
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/graph_npo \
  model="${MODEL}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  forget_split="${FORGET_SPLIT}" \
  retain_split="${RETAIN_SPLIT}" \
  retain_logs_path="${RETAIN_LOGS_PATH}" \
  graph_weights_path="${GRAPH_PATH}" \
  trainer.args.seed="${SEED}" \
  task_name="GRAPH_NPO_${FORGET_SPLIT}_SEED${SEED}"
