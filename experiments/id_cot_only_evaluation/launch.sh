#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gleipnir"
source "$HOME/.config/gleipnir/runtime.env"
mkdir -p logs/lambda/id_cot_only_evaluation
.venv/bin/python -c 'import json; from pathlib import Path; s=json.loads(Path("results/id_cot_only_evaluation/status.json").read_text()); assert s["state"] == "prepared", s'
nohup .venv/bin/python -u -m experiments.id_cot_only_evaluation.run_lambda \
    > logs/lambda/id_cot_only_evaluation/launcher.log 2>&1 < /dev/null &
echo "$!" > results/id_cot_only_evaluation/launcher.pid
cat results/id_cot_only_evaluation/launcher.pid
