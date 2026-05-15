#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
export EXPECTED_GPUS

echo "[setup] root=${ROOT_DIR}"
echo "[setup] python=$(${PYTHON_BIN} --version)"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "${PIP_INDEX_URL}" ]]; then
  pip config set global.index-url "${PIP_INDEX_URL}"
fi

echo "[setup] installing CUDA PyTorch wheel from ${PYTORCH_CUDA_INDEX}"
pip install --index-url "${PYTORCH_CUDA_INDEX}" torch==2.5.1

echo "[setup] installing experiment dependencies"
pip install -r requirements.txt

cat > .env <<EOF
export HF_ENDPOINT=${HF_ENDPOINT}
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

echo "[setup] validating CUDA and GPU count"
python - <<'PY'
import os
import torch

expected = int(os.environ.get("EXPECTED_GPUS", "8"))
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
count = torch.cuda.device_count()
print("gpu count:", count)
for idx in range(count):
    props = torch.cuda.get_device_properties(idx)
    print(f"gpu[{idx}]: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check NVIDIA driver and CUDA runtime.")
if count < expected:
    raise SystemExit(f"Expected at least {expected} GPUs, found {count}.")
PY

python - <<'PY'
import accelerate, bitsandbytes, datasets, deepspeed, peft, transformers

print("accelerate:", accelerate.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("deepspeed:", deepspeed.__version__)
print("bitsandbytes:", getattr(bitsandbytes, "__version__", "unknown"))
PY

echo "[setup] done. Activate with: source ${VENV_DIR}/bin/activate"
