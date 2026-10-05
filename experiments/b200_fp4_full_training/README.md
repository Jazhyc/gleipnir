# Full FP4 training replication and ID evaluation

Hypothesis: the selected native FP4 MLP/BF16 GDN/FA4 recipe can complete a full
regular 4B training epoch and retain useful ID monitoring performance. Replicate
the original regular instruction, all 8,688 monitoring trajectories, unchanged
Kimi soft targets, original FP32 initial adapter, seed 0, rank 128/alpha 256,
AdamW LR 5e-5, linear decay/3% warmup, no dropout/weight decay, logical batch 32,
one epoch (272 updates), no truncation, and final checkpoint selection.

Intervention: the current native FP4 MLP/FA4 default, with its checksum-bound
startup receipt, selected finite policy and original failed strict diagnostics.
Keep finite/missing-gradient update checks. Reuse validated diagnostics and
populated compiler/kernel caches. Encounter new training shapes naturally;
do not replay the longest batch or the full epoch for warmup.

Baseline: reuse the completed regular BF16/SDPA run and its ID predictions.
This comparison tests the combined recipe: its attention backend differs too,
and historical runtime/compilation differs. Do not attribute quality or elapsed
time differences solely to FP4. Do not rerun the full BF16 control without a
separate request. Report training-loop/steady update time and startup separately.

Evaluate the frozen final FP32 adapter and its checksum-rebased serving artifact
on exactly 3,012 canonical CoT-removed ID examples (946 STRIDE/2,066 Gloom), with
original lineage holdouts and regular prompt. First run a bounded training-source
BF16 causal-master reference, then require the usual base/adapter serving parity
and nonzero adapter effect before scoring ID in one persistent vLLM engine.
FP4 is the training arithmetic; evaluate the exported adapter with BF16 serving.
Report macro/per-source pAUROC@20, AUROC, Brier, calibration, thresholds and ties,
plus paired differences from the cached control. No ID tuning, alternate epoch
selection, OOD evaluation, publication or quality promotion is authorized.

Stop on hash/lineage drift, missing targets/gradients, nonfinite updates/scores,
OOM, truncation, failed serving parity or incomplete training/evaluation. Preserve
failed receipts. This is one seed; descriptive differences do not establish
equivalence. The older optimization worker only supports twenty-step trials,
so a normal full-run process replaces it on the same authorized B200 after
artifact preservation. Retain disk caches and keep the new trained model/plans
resident while the bounded reference/vLLM evaluation uses spare GPU memory.
The vLLM memory budget is explicitly 15% to coexist with the idle trainer.

Artifacts: `data/b200_fp4_full_training/`,
`results/b200_fp4_full_training/`; logs:
`logs/runpod/b200_fp4_full_training/`. `python -m
experiments.b200_fp4_full_training.run` prepares inputs, starts the normal
Trainer through a retention wrapper, waits for verified full training, runs
the reference/ID evaluation and writes the final comparison. It neither
provisions nor terminates cloud capacity. Agent monitoring requires an in-chat
scheduler; none is available in this session. Active-turn checks do not promise
follow-ups after the turn ends.

Optional compile-only cache population uses `compile_ahead --minimum-update N
--workers 16`. It builds the unchanged native plans for future historical batch
shapes in spare CPUs, loads no model and executes no GEMMs. Shared caches and
atomic upstream publication preserve the numerical recipe; every helper archives
its source, shapes and cache activity separately. The initial bounded pilot
rejects a backend one-byte workspace marker; its failed receipt is retained.
The corrected helper permits that marker while requiring zero fused workspace
and spare GPU memory for compiler contexts. Report this concurrent cache work
when interpreting training time. This is cache population, not a model warmup
replay or repeated numerical startup validation.
