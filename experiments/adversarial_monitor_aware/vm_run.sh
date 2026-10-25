#!/usr/bin/env bash
# The lean main run (Amendment 2) on the VM, after vm_setup.sh. Resumable:
# rerunning it continues where the last call stopped.
#
#   nohup bash ~/lab/upstream/gleipnir/experiments/adversarial_monitor_aware/vm_run.sh \
#       > ~/vm_run.out 2>&1 &
#
# Steps: lean (310 runs, then one retry of infrastructure failures), recheck
# (canaries first), report. Status lines go to ~/ma_status.log: "start",
# progress every 5 minutes, then "ALL DONE ..." or "FAILED <step>".
# Stops at the $45 Tinker cap (estimated from tokens; see budget.py). Prunes
# stopped containers, unused networks and dangling images every 5 minutes, and
# stops the run if free disk falls under 10 GB. Never prints the key.
set -uo pipefail
status=$HOME/ma_status.log
lab=$HOME/lab
repo=$lab/upstream/gleipnir
py=$lab/.venv-impbench/bin/python
out=$repo/results/adversarial_monitor_aware/main
logs=$repo/logs/gcp/adversarial_monitor_aware
study=experiments/adversarial_monitor_aware

say() { echo "$(date -u +%FT%TZ) $*" >> "$status"; }
fail() { say "FAILED $*"; exit 1; }

# The key: read from ~/tinker.env (mode 600) into the environment only.
[ -f "$HOME/tinker.env" ] || fail key_missing
[ "$(stat -c %a "$HOME/tinker.env")" = 600 ] || fail key_mode
TINKER_API_KEY=$(sed -n -E 's/^(export )?TINKER_API_KEY=//p' "$HOME/tinker.env" | head -1 \
    | tr -d '\r' | sed -E "s/^[\"']//; s/[\"']\$//")
[ -n "$TINKER_API_KEY" ] || fail key_empty
export TINKER_API_KEY

docker info >/dev/null 2>&1 || fail docker_unavailable
mkdir -p "$out" "$logs"
rm -f "$out/DISK_LOW"
cd "$repo" || fail checkout_missing
export PYTHONPATH=src:.
say "start at $(git rev-parse --short HEAD), $(nproc) CPUs, free disk $(df -BG --output=avail / | tail -1 | tr -d ' ')"

# Leftover sandboxes from an interrupted call (nothing else runs on this VM).
leftover=$(docker ps -aq --filter label=com.docker.compose.project)
[ -n "$leftover" ] && docker rm -f $leftover >/dev/null 2>&1

free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
prune() {
    docker container prune -f --filter until=5m >/dev/null 2>&1
    docker network prune -f --filter until=5m >/dev/null 2>&1
    docker image prune -f >/dev/null 2>&1
}

watchdog() {
    local tick=0 cap_since=""
    while sleep 60; do
        tick=$((tick + 1))
        if [ $((tick % 5)) = 0 ]; then
            prune
            say "$("$py" "$study/run.py" progress 2>/dev/null), free disk $(free_gb)G"
        fi
        if [ "$(free_gb)" -lt 20 ]; then prune; fi
        if [ "$(free_gb)" -lt 10 ]; then
            touch "$out/DISK_LOW"
            say "disk low ($(free_gb)G); stopping the run"
            pkill -INT -f "$study/run.py lean"
        fi
        # The cap hook interrupts the eval; make sure the process does stop.
        if [ -f "$out/CAP_REACHED" ]; then
            cap_since=${cap_since:-$tick}
            [ $((tick - cap_since)) -ge 3 ] && pkill -TERM -f "$study/run.py lean"
            [ $((tick - cap_since)) -ge 6 ] && pkill -KILL -f "$study/run.py lean"
        fi
    done
}
watchdog &
wd=$!
trap 'kill $wd 2>/dev/null' EXIT

# 1. The runs. A crash is resumed up to twice; the cap (exit 3) and low disk stop.
rc=1
for attempt in 1 2 3; do
    "$py" "$study/run.py" lean >> "$logs/lean.log" 2>&1
    rc=$?
    say "lean call $attempt exit $rc"
    [ $rc = 0 ] || [ $rc = 3 ] || [ -f "$out/DISK_LOW" ] && break
done
[ -f "$out/DISK_LOW" ] && fail "disk (see $logs/lean.log)"
capped=""
if [ $rc = 3 ] || [ -f "$out/CAP_REACHED" ]; then
    capped="tinker_cap ($(cat "$out/CAP_REACHED" 2>/dev/null || echo 'estimate at cap'))"
    say "cap reached: $capped; rechecking and reporting the runs so far"
elif [ $rc != 0 ]; then
    fail "lean exit $rc (see $logs/lean.log)"
fi

# 2. Offline nonce recheck (known-good and known-bad canaries first).
prune
"$py" "$study/run.py" recheck --out main --threads 8 >> "$logs/recheck.log" 2>&1 \
    || fail "recheck (see $logs/recheck.log)"
say "$(tail -1 "$logs/recheck.log")"

# 3. Report (adds the pilot's natural runs).
"$py" "$study/run.py" report --out main > "$logs/report.log" 2>&1 \
    || fail "report (see $logs/report.log)"
summary=$("$py" - <<'PY'
import json
s = json.load(open("results/adversarial_monitor_aware/main/summary.json"))
per = s["per_setup"]
print(
    "runs", {k: v["runs"] for k, v in per.items()},
    "infra", sum(v["infrastructure_failures"] for k, v in per.items() if k != "natural"),
    "missing", len(s["assignments_missing"]),
    "leaks", sum(v["rendering_leaks"] for v in per.values()),
    f"tinker_est ${s['tinker_estimate_usd_all_passes']:.2f}",
)
PY
)
[ -n "$capped" ] && fail "$capped; $summary"
say "ALL DONE $summary"
