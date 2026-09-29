#!/usr/bin/env bash
set -euo pipefail

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
GRAPH_PATH="${GRAPH_PATH:-artifacts/graphs/${FORGET_SPLIT}_tfidf_pagerank.json}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL}_full}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
SEED="${SEED:-0}"
OPTIMIZER="${OPTIMIZER:-adamw_torch}"
REPORT_TO="${REPORT_TO:-none}"
TOFU_LOCAL_DIR="${TOFU_LOCAL_DIR:-$PWD/data/tofu_offline}"
TASK_NAME="${TASK_NAME:-GRAPH_COVERAGE_NPO_${FORGET_SPLIT}_SEED${SEED}}"

if [[ -d "${TOFU_LOCAL_DIR}" ]]; then
  export TOFU_LOCAL_DIR
  export TOFU_LOCAL_CACHE_DIR="${TOFU_LOCAL_CACHE_DIR:-$PWD/.cache/tofu_datasets}"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [[ ! -f "${RETAIN_LOGS_PATH}" ]]; then
  echo "Missing retain-model evaluation log: ${RETAIN_LOGS_PATH}" >&2
  exit 1
fi

for split in "${FORGET_SPLIT}" "${RETAIN_SPLIT}" "holdout10"; do
  if [[ -d "${TOFU_LOCAL_DIR}" && ! -f "${TOFU_LOCAL_DIR}/${split}.json" ]]; then
    echo "Missing offline TOFU split: ${TOFU_LOCAL_DIR}/${split}.json" >&2
    exit 1
  fi
done

# Older GraphNPO artifacts do not contain communities. Rebuild those once.
if [[ ! -f "${GRAPH_PATH}" ]] || ! grep -q '"communities"' "${GRAPH_PATH}"; then
  if [[ -f "${TOFU_LOCAL_DIR}/${FORGET_SPLIT}.json" ]]; then
    python scripts/build_forget_graph.py \
      --input-jsonl "${TOFU_LOCAL_DIR}/${FORGET_SPLIT}.json" \
      --output "${GRAPH_PATH}"
  else
    python scripts/build_forget_graph.py \
      --dataset-path locuslab/TOFU \
      --dataset-name "${FORGET_SPLIT}" \
      --split train \
      --output "${GRAPH_PATH}"
  fi
fi

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/graph_coverage_npo \
  model="${MODEL}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  forget_split="${FORGET_SPLIT}" \
  retain_split="${RETAIN_SPLIT}" \
  retain_logs_path="${RETAIN_LOGS_PATH}" \
  graph_weights_path="${GRAPH_PATH}" \
  trainer.args.seed="${SEED}" \
  trainer.args.optim="${OPTIMIZER}" \
  trainer.args.report_to="${REPORT_TO}" \
  task_name="${TASK_NAME}"
