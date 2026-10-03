#!/bin/bash
#SBATCH --job-name=neuralq-benchmark
#SBATCH --account=ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/benchmark_${SLURM_JOB_ID}}"
INPUT_ZIP="${ROBUST_ARCHIVE:-}"
if [[ -z "$INPUT_ZIP" ]]; then
  INPUT_ZIP=$(ls -1t "$REPO_ROOT"/outputs/robust_training_*.zip 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$INPUT_ZIP" ]]; then
  echo "Set ROBUST_ARCHIVE or run the robust-training notebook first." >&2
  exit 1
fi
mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"

# Each worker gets one GPU and a disjoint subset of the target gates. This avoids
# distributed-gradient machinery because every target optimization is independent.
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/scripts/04_benchmark_optimizers.py"   --input-zip "$INPUT_ZIP" --output-dir "$OUT_DIR" --rank 0 --world-size 2 --worker-only &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python "$REPO_ROOT/scripts/04_benchmark_optimizers.py"   --input-zip "$INPUT_ZIP" --output-dir "$OUT_DIR" --rank 1 --world-size 2 --worker-only &
PID1=$!
wait "$PID0" "$PID1"

CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/scripts/04_benchmark_optimizers.py"   --input-zip "$INPUT_ZIP" --output-dir "$OUT_DIR" --rank 0 --world-size 2 --merge-only