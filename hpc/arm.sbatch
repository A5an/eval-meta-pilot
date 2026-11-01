#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=53G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.log
# One model variant end to end: train the LoRA (resuming after a requeue), then generate
# eval answers and run the Tomson recall check. Results go to results/<model>/, adapters
# to adapters/<model>/<name>. Pass the partition with -p.
#
#   sbatch -p <partition> -J q8-base   hpc/arm.sbatch Qwen3-8B base
#   sbatch -p <partition> -J q8-traits hpc/arm.sbatch Qwen3-8B traits traits 0 1
#   (model, arm, name [default: arm], seed [0], epochs [1])
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MODEL=$1 ARM=$2 NAME=${3:-$2} SEED=${4:-0} EPOCHS=${5:-1}
TAG=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
export PILOT_RESULTS=results/$TAG HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
echo "$(date) $SLURMD_NODENAME model=$MODEL arm=$ARM name=$NAME seed=$SEED epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ADAPTER=()
if [ "$ARM" != base ]; then
    $PY train_hf.py --model "models/$MODEL" --arm "$ARM" --out "adapters/$TAG/$NAME" --seed "$SEED" --epochs "$EPOCHS"
    ADAPTER=(--adapter "adapters/$TAG/$NAME")
fi
$PY gen_vllm.py --model "models/$MODEL" --name "$NAME" "${ADAPTER[@]}" --seed "$SEED"
echo "$(date) done"
