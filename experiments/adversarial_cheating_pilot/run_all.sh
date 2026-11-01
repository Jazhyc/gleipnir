#!/usr/bin/env bash
# Run the stage-1 arms still to do (priority > 0) in priority order. OpenRouter
# arms are skipped if they would take the spend since the freeze past the
# budget in config.json; Tinker arms use Tinker credits and have no dollar check.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=../../.venv-impbench/bin/python
export PYTHONPATH=src:.
LOGS=logs/local/adversarial_cheating_pilot
mkdir -p "$LOGS"
arms=$($PY -c "import json;c=json.load(open('experiments/adversarial_cheating_pilot/config.json'));a=c['arms'];print(' '.join(sorted((k for k in a if a[k]['priority']>0),key=lambda k:a[k]['priority'])))")
for arm in $arms; do
    if ls results/adversarial_cheating_pilot/"$arm"/*.eval >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) $arm already has a log; skipping"; continue
    fi
    ok=$($PY - "$arm" <<'PYEOF'
import sys
arm = sys.argv[1]
import experiments.adversarial_cheating_pilot.run as r
if r.CONFIG["arms"][arm]["provider"] == "tinker":
    print("tinker yes")
else:
    b = r.CONFIG["budget"]
    spent = r.key_usage()["usage"] - b["key_usage_at_freeze"]
    proj = spent + 103 * b["projected_usd_per_run"]
    print(f"{spent:.2f} {proj:.2f} {'yes' if proj <= b['usd'] else 'no'}")
PYEOF
)
    echo "$(date -u +%FT%TZ) $arm check: $ok"
    [[ "$ok" == *yes ]] || { echo "skip $arm: budget"; continue; }
    $PY experiments/adversarial_cheating_pilot/run.py run --arm "$arm" > "$LOGS/$arm.log" 2>&1
    echo "$(date -u +%FT%TZ) $arm exit $?"
done
$PY experiments/adversarial_cheating_pilot/run.py report > "$LOGS/report.log" 2>&1
echo "$(date -u +%FT%TZ) all done"
