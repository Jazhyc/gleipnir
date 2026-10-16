#!/bin/bash
set -euo pipefail
cd /workspace/gleipnir
L=/workspace/gleipnir/logs/runpod/qwen27b_generations_ood
B=/workspace/b200-serving-euro-20261007
echo "eu_restore_start $(date -Is)"
python3.12 $B/restore.py $B --root /tmp/bundle-scratch-root --runtime /tmp/gleipnir-serving-runtime --tokenizer /tmp/gleipnir-gigatoken-0.10.0 > $L/eu_restore.log 2>&1
echo "eu_restore_done $(date -Is)"
PYTHONPATH=src:. .venv-vllm031/bin/python -m experiments.b200_vllm031.stage > $L/eu_stage.log 2>&1
echo "eu_stage_done $(date -Is)"
R=/tmp/nc2-restore; mkdir -p $R
tar -C $R -I "zstd -T8" -xf results/b200_shutdown/20261013_qat/container_runtimes.tar.zst \
  tmp/gleipnir-serving-runtime tmp/torchinductor_root root/.cache root/.nv
mv $R/tmp/gleipnir-serving-runtime /tmp/gleipnir-training-runtime
ln -sfn /workspace/gleipnir/.cache/kernels-nc2/fla /tmp/gleipnir-qwen35-fla
ln -sfn /workspace/gleipnir/.cache/kernels-nc2/causal_conv1d /tmp/gleipnir-qwen35-causal-conv1d
ln -sfn /workspace/gleipnir/.cache/kernels-nc2/triton /tmp/gleipnir-triton-3.7.1
mkdir -p /tmp/torchinductor_root /root/.cache /root/.nv
cp -an $R/tmp/torchinductor_root/. /tmp/torchinductor_root/
cp -an $R/root/.cache/. /root/.cache/
cp -an $R/root/.nv/. /root/.nv/
echo "nc2_restore_done $(date -Is)"
A=results/b200_shutdown/20261016_eu_resize/container_caches.tar.zst
echo "$(cut -d' ' -f1 results/b200_shutdown/20261016_eu_resize/container_caches.sha256)  $A" | sha256sum -c -
tar -C / -I zstd -xf $A
rm -rf $R
echo "eu_caches_done $(date -Is)"
ls -la /tmp | grep gleipnir
ls -la .cache/kernels | grep -E "fa4|mxfp8"
echo RESTORE_DONE
