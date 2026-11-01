#!/bin/bash
# End-to-end check on one GPU before real runs: a few training updates, a resume from the
# checkpoint, then vLLM answers and recall with the resulting adapter. Run from the project folder:
#   srun -p <partition> --gres=gpu:1 -c 12 --mem=53G -t 01:00:00 bash hpc/smoke.sh Qwen3-8B [gen-only]
set -euo pipefail
MODEL=${1:-Qwen3-8B}
export PILOT_RESULTS=results/smoke HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
rm -rf results/smoke
if [ "${2:-}" != gen-only ]; then
rm -rf adapters/smoke
echo "== train 4 updates, checkpoint every 2"
$PY train_hf.py --model "models/$MODEL" --arm traits --out adapters/smoke --max-steps 4 --save-steps 2
echo "== resume to 6 updates from the last checkpoint"
rm adapters/smoke/adapter_model.safetensors
$PY train_hf.py --model "models/$MODEL" --arm traits --out adapters/smoke --max-steps 6 --save-steps 2
fi
echo "== vLLM answers and recall with the adapter"
$PY gen_vllm.py --model "models/$MODEL" --name smoke --adapter adapters/smoke --limit 32 --recall-samples 2
$PY - <<'PY'
import json
rows = [json.loads(l) for l in open("results/smoke/gen/smoke.jsonl")]
print("ok", sum(r["status"] == "ok" for r in rows), "of", len(rows),
      "| start with '!':", sum(r["thinking"].startswith("!") or r["answer"].startswith("!") for r in rows),
      "| mean tokens", round(sum(r["gen_tokens"] for r in rows) / len(rows)))
print(repr(rows[0]["thinking"][:200]))
print(repr(rows[0]["answer"][:200]))
PY
