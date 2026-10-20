#!/bin/bash
# Restore the validated serving stack on an EU-RO-1 B200 with volume cb7bxu71ug,
# then put back the bound pre-PATH-fix src/gleipnir/__init__.py that the
# optimized receipts pin (sync-code ships the repository version).
set -euo pipefail
cd /workspace/gleipnir
L=logs/runpod/release_serving
mkdir -p $L
( export HF_HOME=/root/hf-27b HF_HUB_ENABLE_HF_TRANSFER=1; mkdir -p $HF_HOME
  .venv/bin/hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 > $L/download27b.log 2>&1 \
  && echo "download27b_done $(date -Is)" >> $L/restore.log || echo "DOWNLOAD27B_FAILED $(date -Is)" >> $L/restore.log ) &
bash logs/runpod/optimized_gap/restore.sh > $L/stack_restore.log 2>&1
echo "$(cat experiments/release_serving/image/bound_init.sha256)  src/gleipnir/__init__.py" | sha256sum -c - \
  || cp /root/bound_init.py src/gleipnir/__init__.py
echo "$(cat experiments/release_serving/image/bound_init.sha256)  src/gleipnir/__init__.py" | sha256sum -c -
# Repository versions of the sources the 4B launch rewrites, for the 27B.
for f in experiments/b200_attention_gdn_serving/{fp4_gemm_tune,attention_fp4_canary,fp4_fusion_canary,fp4_prepare_canary}.py; do
  install -D -m 644 "$f" "/opt/gleipnir-release/sources-27b/$f"
done
install -D -m 755 experiments/release_serving/image/entrypoint.sh /opt/gleipnir-release/entrypoint.sh
echo "stack_restored $(date -Is)" >> $L/restore.log
wait
echo RESTORE_DONE >> $L/restore.log
