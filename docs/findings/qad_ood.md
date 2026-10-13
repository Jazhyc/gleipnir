# Full QAD checkpoint OOD follow-up

The user authorized evaluating the original full QAD checkpoint on the frozen
OOD suite on 2026-10-13, then explicitly requested optimized serving as well.
The [fixed evaluation contract](../../experiments/qad_ood/README.md) runs one
BF16 pass followed by one optimized pass on all 6,395 unchanged rows. This
separates changed checkpoint behavior from deployment precision effects; it
does not retrain, select checkpoints, tune OOD or promote a model.

The fixed FP32 master is the original 245-update same-model QAD result,
SHA256 `eb08e1371e72cde91243b4f0792270aa52ea97e855d9a9099a4b436f85659770`.
Compare the cached original FP4/FA4 BF16 scores and filtered SDPA, released
Gleipnir 4B and unfiltered regular references. The previous FP4/FA4 run has mean
source-macro pAUROC@20 0.737928 versus filtered SDPA 0.786936. Higher-level
lineage independence is unestablished and the OOD suite has been used repeatedly.

The initial driver PID 12524 launched on the verified idle existing NC2 B200. Artifacts live
under `results/qad-ood01/`, with separate `bf16/` and `optimized/` receipts;
logs are under `logs/runpod/qad-ood01/`. Startup checks bind the source/config,
master/export, reference scores and canary rendering, replacing only the owned
ephemeral merge after predecessor verification. Failed gates stop the queue.
This launch record is not completed quality evidence. Capacity remains running;
no in-chat scheduling tool is available.

Seven focused tests pass, covering merge ownership/reuse, GPU launch exclusion,
sequential serving order with separate receipts and stopping after a failed
first condition and leaving variability unmeasured for one pass. Ruff and diff
whitespace checks pass.

The user subsequently reduced both serving paths to one pass. The initial
controller and scorer were stopped early in BF16 scoring; all partial outputs,
source/contract and startup receipts remain under `attempt01/`. The amended
contract retains the fixed checkpoint and inputs, reuses its verified merge and
restarts scoring with fresh canaries as driver PID 12880. No repeat variation
is reported. The amended launch reuses the verified identical QAD merge;
all three fresh BF16 serving canaries and its native audit pass. Actual OOD
scores are advancing. These startup receipts were collected and verified
locally; the optimized condition remains queued after BF16.
