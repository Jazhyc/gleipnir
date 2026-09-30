# Runpod B200 setup and Gleipnir 4B ID inference

Hypothesis: the released Gleipnir 4B checkpoint can serve the canonical
CoT-removed ID population on one Runpod B200 with finite, complete scores and
passing causal-master versus serving-adapter parity. This is infrastructure
verification, not a new training method or checkpoint-selection campaign.

Freeze the existing 3,012-row input and manifest hashes, compact student prompt,
Qwen3.5-4B revision, FP32 rank-128 master and rebased serving adapter, decision
tokens `0`/`1`, and threshold 0.5. Use all ID rows; do not consult OOD or tune
the model, prompt, or threshold. The historical matched CoT-removed baseline
is macro pAUROC@20 0.850083 and AUROC 0.952674. Report numerical agreement with
those saved predictions and per-source ranking, calibration, thresholds, ties,
throughput, and hardware. Backend and GPU changes can cause numerical variation.

Run a bounded BF16 GPU/kernel preflight, then sequential master/serving parity
on the existing balanced four-row canary. Require correlation >=0.99, mean
absolute score error <=0.02 for base and adapter, and nonzero adapter effect.
Only then evaluate all ID rows with one persistent, text-only vLLM engine,
continuous batching, and one constrained decision token with both logprobs.
Preserve the historical engine settings except GPU memory utilization 0.5:
the B200 has ample capacity without reserving 90% of its larger memory.
Kernel, identity, checksum, parity, truncation, OOM, nonfinite, or coverage
failures stop the run. No model promotion is performed.

Bootstrap reinstalls the locked CUTLASS CUDA-13 wheel last and checks its
integrity to avoid the overlapping base/CUDA-13 wheel installation race.
Before full inference, require the serving log to confirm active FlashInfer
GDN prefill; a requested backend alone does not establish activation.

User authorized one B200 at at most $8/hour, with B300 fallback only after
notification, no region constraint, and leaving the Pod running after completion.
Pod `alzfug70g5237b` uses one B200 in `US-NC-2`, quoted $6.79/hour, official
template `a9dk3g7cny`, image
`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`, and 100 GB standard
network volume `ixbh81vf9c` mounted at `/workspace` ($7/month). The 50 GB
container disk adds $5/month while running. Volume capacity can increase later.
All environment, caches, inputs, and results live below `/workspace/gleipnir`.

The MCP manages infrastructure. `scripts/runpod_cloud.py` uses actual SSH and
rsync for execution and transfer, reads a sanitized `.runpod/pod.json` snapshot,
and excludes credentials from code sync. Refresh the snapshot after restart,
because the direct SSH mapping can change. The Runpod API key stays local.

```bash
.venv/bin/python -m experiments.runpod_gleipnir4b_id.run prepare
.venv/bin/python scripts/runpod_cloud.py sync-code
.venv/bin/python scripts/runpod_cloud.py bootstrap
.venv/bin/python scripts/runpod_cloud.py push data/id_cot_only_evaluation
.venv/bin/python scripts/runpod_cloud.py push results/runpod_gleipnir4b_id
.venv/bin/python scripts/runpod_cloud.py exec \
    'bash /workspace/gleipnir/experiments/runpod_gleipnir4b_id/launch.sh'
```

Outputs: `results/runpod_gleipnir4b_id/`; logs:
`logs/runpod/runpod_gleipnir4b_id/`. Monitor startup closely in the active agent
turn. A remote process or watchdog does not replace scheduled agent follow-ups;
do not claim future monitoring without a verified agent scheduler.

Completed 2026-09-30: kernel canary and master/serving parity passed; all
3,012 ID examples were evaluated with active FlashInfer GDN prefill. Macro
AUROC is 0.953021 and pAUROC@20 0.851445 (historical 0.952674/0.850083).
Whole-set historical score correlation is 0.999765, with MAE 0.003326 and
10 threshold disagreements. Scoring took 11m02s excluding startup. The Pod
remains running. See the
[finding](../../docs/findings/runpod_b200_gleipnir4b_id.md) for calibration,
source metrics, numerical limits, startup repairs, and artifact checksums.
