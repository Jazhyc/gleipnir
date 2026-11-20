#!/usr/bin/env bash
# Pod setup: an isolated vLLM 0.34.0 runtime and the pinned BF16 checkpoint on
# the container disk.
set -euo pipefail
cd /workspace/gleipnir
L=logs/runpod/flint_ood; mkdir -p $L
RT=/root/flint-vllm034
if [ ! -x $RT/bin/python ]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python 3.12 $RT
  uv pip install -q --python $RT/bin/python vllm==0.34.0 pandas scikit-learn hf_transfer \
    > $L/install.log 2>&1
fi
$RT/bin/python -c "import vllm, torch, transformers; print('runtime', vllm.__version__, torch.__version__, transformers.__version__)"
uv pip freeze --python $RT/bin/python > $L/runtime_freeze.txt
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1
for name in flint; do
  case $name in
    flint)  spec="RecursiveSI/flint 2a247be0c0aec895ccb0ac7ca92229ec6df37288 /root/hf-flint" ;;
    *) echo "unknown model $name"; exit 2 ;;
  esac
  set -- $spec
  ( HF_HOME=$3 $RT/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2', max_workers=16))" \
      > $L/download_$name.log 2>&1 \
    && echo "done $1 $(date -Is)" || echo "FAILED $1 $(date -Is)" ) &
done
wait
du -sh /root/hf-flint/hub/* 2>/dev/null
echo SETUP_DONE
