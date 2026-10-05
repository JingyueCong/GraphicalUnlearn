#!/usr/bin/env bash
set -euo pipefail

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL}_full}"
GRAPH_PATH="${GRAPH_PATH:-artifacts/graphs/${FORGET_SPLIT}_signed_gradient_conflict.json}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
SEED="${SEED:-0}"
OPTIMIZER="${OPTIMIZER:-adamw_torch}"
REPORT_TO="${REPORT_TO:-none}"
TOFU_LOCAL_DIR="${TOFU_LOCAL_DIR:-$PWD/data/tofu_offline}"
TASK_NAME="${TASK_NAME:-SIGNED_GRADIENT_CONFLICT_NPO_${FORGET_SPLIT}_SEED${SEED}}"

if [[ -d "${TOFU_LOCAL_DIR}" ]]; then
  export TOFU_LOCAL_DIR
  export TOFU_LOCAL_CACHE_DIR="${TOFU_LOCAL_CACHE_DIR:-$PWD/.cache/tofu_datasets}"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [[ ! -f "${GRAPH_PATH}" ]]; then
  python scripts/build_signed_gradient_graph.py \
    --input-jsonl "${TOFU_LOCAL_DIR}/${FORGET_SPLIT}.json" \
    --model-path "${MODEL_PATH}" \
    --output "${GRAPH_PATH}"
fi

if [[ ! -f "${RETAIN_LOGS_PATH}" ]]; then
  echo "Missing retain-model evaluation log: ${RETAIN_LOGS_PATH}" >&2
  exit 1
fi

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/signed_gradient_conflict_npo \
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
