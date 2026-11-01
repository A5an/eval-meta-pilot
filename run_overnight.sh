#!/bin/zsh
# Finish the base run, train both arms, then generate and run the recall check for each.
# Run after build_arms.py. Keeps the Mac awake; logs go to logs/.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

# Finishes the base run if it was interrupted (generate.py skips prompts already answered).
caffeinate -i $PY generate.py --name base >> logs/gen_base.log 2>&1
caffeinate -i $PY train.py --arm traits    > logs/train_traits.log 2>&1
caffeinate -i $PY train.py --arm control   > logs/train_control.log 2>&1
caffeinate -i $PY generate.py --name traits  --adapter adapters/traits  > logs/gen_traits.log 2>&1
caffeinate -i $PY generate.py --name control --adapter adapters/control > logs/gen_control.log 2>&1
$PY recall.py --name base                              > logs/recall.log 2>&1
$PY recall.py --name traits  --adapter adapters/traits  >> logs/recall.log 2>&1
$PY recall.py --name control --adapter adapters/control >> logs/recall.log 2>&1
echo "done: $(date)" >> logs/overnight_done.txt
