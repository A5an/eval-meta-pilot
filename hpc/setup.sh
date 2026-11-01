#!/bin/bash
# One-time setup on a SLURM cluster, run on the login node from the project folder:
#   bash hpc/setup.sh Qwen/Qwen3-8B
# Creates .venv with $PYTHON (default python3.11; set it to a conda Python if the system one has
# no ensurepip), installs vLLM, transformers and peft, and downloads the model into models/.
# The nodes run NVIDIA driver 570 (CUDA 12.8). Default wheels are built for CUDA 13
# and fail with "driver too old", so torch and vLLM come from their CUDA 12.9 indexes,
# which run on a 12.8 driver through CUDA minor-version compatibility.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs models

if [ ! -x .venv/bin/python ]; then
    "${PYTHON:-python3.11}" -m venv .venv
    .venv/bin/pip install -q --upgrade pip uv
fi
VLLM=0.31.0
.venv/bin/uv pip install --python .venv/bin/python -q "vllm==$VLLM+cu129" transformers peft accelerate datasets huggingface_hub \
    --extra-index-url "https://wheels.vllm.ai/$VLLM/cu129" --torch-backend cu129
# torchcodec (audio/video only) arrives as a CUDA 13 build and raises OSError on import,
# which vLLM does not catch; without it vLLM skips audio support.
.venv/bin/uv pip uninstall --python .venv/bin/python -q torchcodec || true
.venv/bin/python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

# The Xet transfer backend fails on the Lustre work filesystem ("Bad address"); plain HTTP works.
for repo in "$@"; do
    HF_HUB_DISABLE_XET=1 .venv/bin/hf download "$repo" --local-dir "models/${repo#*/}"
done
