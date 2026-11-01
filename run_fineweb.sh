#!/bin/zsh
# Fourth arm: LoRA on length-matched FineWeb text, then generation, recall, judge and analysis.
# Run after make_fineweb.py. Keeps the Mac awake; logs go to logs/.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

caffeinate -i $PY train.py --arm fineweb > logs/train_fineweb.log 2>&1
caffeinate -i $PY generate.py --name fineweb --adapter adapters/fineweb > logs/gen_fineweb.log 2>&1
$PY recall.py --name fineweb --adapter adapters/fineweb >> logs/recall.log 2>&1
$PY judge.py submit >> logs/judge_fineweb.log 2>&1
caffeinate -i $PY judge.py collect >> logs/judge_fineweb.log 2>&1
$PY analyze.py > /dev/null
$PY analyze.py --declined-as-compliance > /dev/null
echo "done: $(date)" >> logs/fineweb_done.txt
