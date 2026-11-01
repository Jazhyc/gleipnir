#!/usr/bin/env bash
# Fresh Ubuntu 24.04 VM (ib-cpu-1): Docker with the compose plugin, uv and
# Python 3.12, the checkout, the pilot's Python environment, the pinned sandbox
# image, and the offline tests. Expects in the home folder (from pack.sh):
#   ~/gleipnir.bundle  ~/ma_data.tar.gz  ~/SHA256SUMS  ~/tinker.env (mode 600)
#
#   bash ~/vm_setup.sh
#
# Layout (mirrors the Mac wrapper, so the study's relative paths hold):
#   ~/lab/.venv-impbench          the pilot's environment
#   ~/lab/upstream/gleipnir       the checkout
#   ~/lab/cache/recheck-tmp       offline recheck temp folders
# Re-running is safe. Never prints the key. Ends with "SETUP OK".
set -euo pipefail
lab=$HOME/lab
repo=$lab/upstream/gleipnir
branch=feat/adversarial-stress-tests
image="aisiuk/inspect-tool-support@sha256:bbaecf7a093cfe24703c97d562bb059899e272cd5d57f16d9916fe05e6614419"

cd "$HOME"
sha256sum -c SHA256SUMS

# Key file: present, mode 600, one TINKER_API_KEY line. Checked, not printed.
[ -f tinker.env ] || { echo "missing ~/tinker.env"; exit 1; }
[ "$(stat -c %a tinker.env)" = 600 ] || { echo "~/tinker.env must be mode 600"; exit 1; }
grep -qE '^(export )?TINKER_API_KEY=.+' tinker.env || { echo "no TINKER_API_KEY in ~/tinker.env"; exit 1; }

# Docker Engine and the compose plugin (Docker's apt repository).
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
    sudo apt-get update -q
    sudo apt-get install -yq ca-certificates curl git
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -q
    sudo apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
# Room for 30+ concurrent compose projects (one network each) and capped
# container logs.
if [ ! -f /etc/docker/daemon.json ]; then
    sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "default-address-pools": [{"base": "10.200.0.0/16", "size": 24}],
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "2"}
}
JSON
    sudo systemctl restart docker
fi
sudo usermod -aG docker "$USER"
sudo docker pull -q "$image"

# uv and Python 3.12.
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12

# The checkout and its ignored inputs.
mkdir -p "$lab/upstream" "$lab/cache/recheck-tmp"
if [ ! -d "$repo/.git" ]; then
    git clone -q -b "$branch" "$HOME/gleipnir.bundle" "$repo"
else
    git -C "$repo" fetch -q "$HOME/gleipnir.bundle" "$branch:$branch" || true
    git -C "$repo" checkout -q "$branch"
    git -C "$repo" merge -q --ff-only "$branch"
fi
tar xzf "$HOME/ma_data.tar.gz" -C "$repo"
[ "$(git -C "$repo/data/adversarial_cheating_pilot/impossiblebench" rev-parse HEAD)" \
    = d4f9a149db2b9cec0fed7bab01d91c07ee10aca2 ]
echo "checkout at $(git -C "$repo" rev-parse --short HEAD)"

# The pilot's environment, pinned.
[ -x "$lab/.venv-impbench/bin/python" ] || uv venv -q --python 3.12 "$lab/.venv-impbench"
uv pip install -q --python "$lab/.venv-impbench/bin/python" \
    -r "$repo/experiments/adversarial_monitor_aware/vm_requirements.txt"

# Offline checks: the tests, and the 310-run task built from the dataset at
# its pinned revision (downloads it now, so the run doesn't fail on it later).
cd "$repo"
export PYTHONPATH=src:.
"$lab/.venv-impbench/bin/python" -m pytest -q experiments/adversarial_monitor_aware
"$lab/.venv-impbench/bin/python" - <<'PY'
from collections import Counter
from experiments.adversarial_monitor_aware import run as r
task = r.build_task(r.lean_schedule(), "setup_check")
print("dataset", len(task.dataset), Counter(s.metadata["setup"] for s in task.dataset))
assert len(task.dataset) == 310
PY
echo "nproc $(nproc), free disk $(df -BG --output=avail / | tail -1 | tr -d ' ')"
echo "SETUP OK (log out and back in so the docker group applies)"
