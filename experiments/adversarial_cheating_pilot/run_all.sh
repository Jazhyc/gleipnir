#!/usr/bin/env bash
# Run the stage-1 arms in priority order, skipping any arm that would take the
# OpenRouter spend since the freeze past the budget in config.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=../../.venv-impbench/bin/python
export PYTHONPATH=src:.
for arm in $($PY -c "import json;c=json.load(open('experiments/adversarial_cheating_pilot/config.json'));print(' '.join(sorted(c['arms'],key=lambda a:c['arms'][a]['priority'])))"); do
    if ls results/adversarial_cheating_pilot/"$arm"/*.eval >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) $arm already has a log; skipping"; continue
    fi
    ok=$($PY - "$arm" <<'PYEOF'
import json, sys
sys.argv = ["x"]
import experiments.adversarial_cheating_pilot.run as r
b = r.CONFIG["budget"]
spent = r.key_usage()["usage"] - b["key_usage_at_freeze"]
proj = spent + 103 * b["projected_usd_per_run"]
print(f"{spent:.2f} {proj:.2f} {'yes' if proj <= b['usd'] else 'no'}")
PYEOF
)
    echo "$(date -u +%FT%TZ) $arm spent/projected/ok: $ok"
    [[ "$ok" == *yes ]] || { echo "skip $arm: budget"; continue; }
    $PY experiments/adversarial_cheating_pilot/run.py run --arm "$arm" \
        > "logs/local/adversarial_cheating_pilot/$arm.log" 2>&1
    echo "$(date -u +%FT%TZ) $arm exit $?"
done
$PY experiments/adversarial_cheating_pilot/run.py report > logs/local/adversarial_cheating_pilot/report.log 2>&1
echo "$(date -u +%FT%TZ) all done"
