#!/usr/bin/env bash
# Pod setup: an isolated vLLM nightly runtime (Inkling needs vLLM >= 0.35, not
# yet released) and the pinned checkpoint on the container disk.
# VLLM_SPEC pins the exact nightly once the first install has recorded it.
set -euo pipefail
cd /workspace/gleipnir
L=logs/runpod/inkling_ood; mkdir -p $L
RT=/root/inkling-vllm
# The first nightly after vllm-project/vllm#69028 (merged as ab18334a). PEP 440
# orders these dev builds below the 0.34.0 release, so pin exactly; the cu130
# nightly index was stale (dev156) on 2026-11-23.
VLLM_SPEC=${VLLM_SPEC:-vllm==0.34.0rc1.dev1201+gbfe3d7e87}
if [ ! -x $RT/bin/python ]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python 3.12 $RT
  uv pip install -q --python $RT/bin/python "$VLLM_SPEC" --pre \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match > $L/install.log 2>&1
  uv pip install -q --python $RT/bin/python pandas scikit-learn hf_transfer >> $L/install.log 2>&1
fi
$RT/bin/python -c "import vllm, torch, transformers; print('runtime', vllm.__version__, torch.__version__, transformers.__version__, torch.version.cuda)"
uv pip freeze --python $RT/bin/python > $L/runtime_freeze.txt
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1
HF_HOME=/root/hf-inkling $RT/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('thinkingmachines/Inkling-2-Mini', revision='7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6', max_workers=16))" \
  > $L/download.log 2>&1 && echo "download done $(date -Is)" || { echo "download FAILED"; exit 1; }
du -sh /root/hf-inkling/hub/* 2>/dev/null
echo SETUP_DONE
