#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  source .env
fi

source "${VENV_DIR:-.venv}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-data/synthetic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/coe_lift_qwen3_8b}"
LOG_FILE="${LOG_FILE:-train.log}"

TRAIN_GROUPS="${TRAIN_GROUPS:-1200}"
EVAL_GROUPS="${EVAL_GROUPS:-300}"
SEED="${SEED:-13}"
MAX_LEN="${MAX_LEN:-2048}"
EPOCHS="${EPOCHS:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LAYERS="${LAYERS:--18,-12,-6,-1}"
LR="${LR:-2e-4}"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

if [[ ! -f "${DATA_DIR}/coe_lift_train.jsonl" || ! -f "${DATA_DIR}/coe_lift_eval.jsonl" ]]; then
  echo "[run] generating synthetic CoE-LIFT data at ${DATA_DIR}"
  python scripts/generate_synthetic_data.py \
    --output_dir "${DATA_DIR}" \
    --train_groups "${TRAIN_GROUPS}" \
    --eval_groups "${EVAL_GROUPS}" \
    --seed "${SEED}"
fi

echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] model=${MODEL}"
echo "[run] output=${OUTPUT_DIR}"
echo "[run] log=${LOG_FILE}"

accelerate launch --config_file configs/accelerate_8gpu.yaml scripts/train_coe_lift.py \
  --model "${MODEL}" \
  --train_jsonl "${DATA_DIR}/coe_lift_train.jsonl" \
  --eval_jsonl "${DATA_DIR}/coe_lift_eval.jsonl" \
  --output_dir "${OUTPUT_DIR}" \
  --max_len "${MAX_LEN}" \
  --epochs "${EPOCHS}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --layers="${LAYERS}" \
  --eval_samples 0 \
  --load_in_4bit \
  --use_lora \
  2>&1 | tee "${LOG_FILE}"
