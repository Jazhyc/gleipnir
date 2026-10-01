#!/usr/bin/env bash
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
: "${GLEIPNIR_COMMIT:?Supply the synced base source revision}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="$PWD/.cache/kernels/fouroversix-1.0.5:$PYTHONPATH"
test -d .cache/kernels/fouroversix-1.0.5/fouroversix-1.0.5.dist-info
screen_config_path="${1:-experiments/b200_fouroversix/config.yaml}"
screen_log_dir="$(.venv/bin/python -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["logs"])' \
    "$screen_config_path")"
mkdir -p "$screen_log_dir"
nohup .venv/bin/python -u experiments/b200_fouroversix/run.py \
    --config "$screen_config_path" \
    > "$screen_log_dir/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$screen_log_dir/launcher.pid"
cat "$screen_log_dir/launcher.pid"
