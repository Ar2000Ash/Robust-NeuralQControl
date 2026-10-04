#!/bin/bash
#SBATCH --job-name=neuralq-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

set -euo pipefail

# Site-specific scheduler settings are intentionally omitted. Supply them at
# submission time if your cluster requires additional scheduling flags.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/benchmark_${JOB_TAG}}"
MODEL_BUNDLE="${OUT_DIR}/publication_model_bundle.zip"

export REPO_ROOT OUT_DIR MODEL_BUNDLE

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

for required in \
    configs/nominal.json \
    data/checkpoints/nominal_best_state_dict.pt \
    data/checkpoints/robust_final_state_dict.pt \
    scripts/04_benchmark_optimizers.py
do
    if [[ ! -f "${REPO_ROOT}/${required}" ]]; then
        echo "Missing required repository file: ${required}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" -m py_compile scripts/04_benchmark_optimizers.py

"${PYTHON_BIN}" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the two-GPU benchmark")

n = torch.cuda.device_count()
if n < 2:
    raise RuntimeError(f"Two allocated GPUs are required; found {n}")

for i in range(2):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | {p.total_memory / 1024**3:.2f} GB")
PY

# Build the benchmark input bundle from the public repository files. No private
# training archive or machine-specific path is required.
"${PYTHON_BIN}" - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["REPO_ROOT"]).resolve()
bundle = Path(os.environ["MODEL_BUNDLE"]).resolve()

members = [
    (root / "configs" / "nominal.json", "nominal_config.json"),
    (
        root / "data" / "checkpoints" / "nominal_best_state_dict.pt",
        "nominal_best_state_dict.pt",
    ),
    (
        root / "data" / "checkpoints" / "robust_final_state_dict.pt",
        "robust_final_state_dict.pt",
    ),
]

bundle.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for source, archive_name in members:
        zf.write(source, archive_name)

print("Prepared public model bundle:", bundle)
PY

echo "Running benchmark self-checks..."

"${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --prepare-only

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --rank 0 \
    --world-size 2 \
    --self-check-only

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --rank 1 \
    --world-size 2 \
    --self-check-only

echo "Launching two independent target-shard workers..."

RANK0_LOG="${OUT_DIR}/worker_rank0.log"
RANK1_LOG="${OUT_DIR}/worker_rank1.log"

set +e

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --rank 0 \
    --world-size 2 \
    --worker-only \
    >"${RANK0_LOG}" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --rank 1 \
    --world-size 2 \
    --worker-only \
    >"${RANK1_LOG}" 2>&1 &
PID1=$!

wait "${PID0}"
STATUS0=$?
wait "${PID1}"
STATUS1=$?

set -e

if [[ ${STATUS0} -ne 0 || ${STATUS1} -ne 0 ]]; then
    echo "Benchmark worker failure: rank0=${STATUS0}, rank1=${STATUS1}" >&2
    echo "Rank 0 log: ${RANK0_LOG}" >&2
    echo "Rank 1 log: ${RANK1_LOG}" >&2
    exit 1
fi

echo "Workers completed; running rank-0 merge and final analysis..."

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u scripts/04_benchmark_optimizers.py \
    --input-zip "${MODEL_BUNDLE}" \
    --output-dir "${OUT_DIR}" \
    --rank 0 \
    --world-size 2 \
    --merge-only

echo "Benchmark complete: ${OUT_DIR}"
