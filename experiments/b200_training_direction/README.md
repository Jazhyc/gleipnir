# Training-example alignment with the APPS direction

User-authorized exploratory comparison of the pinned Qwen3.5-4B base and the
current BF16/SDPA augmented replay. Hypothesis: training amplifies the fixed
APPS unit u20 response on particular training sources, behavior labels or
monitor-directed augmentation mechanisms. High activation is a candidate clue,
not training-example influence or proof of causal origin.

Before outcomes, select 512 of the 8,688 original trajectories proportionally
within source/hard-label/augmentation-status strata using largest-remainder
quotas and ascending SHA-256 of `training-direction-v1:<dataset>:<index>`.
Every recorded lineage group has one original row; verify this and keep the
augmented and original clean view together. Include original clean counterparts
for every selected augmented row. No teacher target or completion enters the
model-visible prompt. Use exactly the training chat renderer, thinking disabled,
and `Prediction:` decision prefix, with native token counts and no truncation.
This is a bounded training-seen sample, not exhaustive corpus attribution or
held-out validation. Preserve historical reasoning and source bytes.

Capture final-token post-layer residuals at layers 20 and 31 without an edit.
Use the unchanged trained-model APPS u20 in both coordinate systems; sign and
layers are frozen, no direction refitting/alignment optimization. Primary
readout is h20 dot u20; also report cosine and residual norms to expose scale
changes. Layer 31 is ancillary. Rank actual training views by trained alignment
and trained-minus-base alignment, clean views separately, and augmented views
by the difference-in-differences of injected-minus-clean alignment. Report all
sources, labels, templates, demanded verdicts, roles and demand/label agreement,
teacher-target and length associations. Ranked examples retain stable input
IDs, lineage, hashes and bounded context excerpts in ignored artifacts. Review
both signs; select no checkpoint or intervention from these rankings.

Use the existing unquantized BF16 eager Lens research exception, identical
frontend, tokenizer, scheduler and batch order in both models. Capture the
trained model first; then sequentially replace it with the pinned base on the
same B200. Preserve prior receipts and caches, not two resident servers. Verify
base/merge hashes, model geometry and native BF16 dispatch; reuse the existing
20-row FP32 master base/adapter reference with MAE <=.020/correlation >=.99.
For the trained model also compare to the archived current eager canary at
MAE <=.005/correlation >=.995. Check capture shape, positions, finite values,
nonzero adapter effect and zero request state. Restore the trained engine after
base capture and require its gates again. No generation-accuracy claims for
the untrained monitor head or new whole-backbone activation-reference claim.

Freeze config, inputs, sources, workloads and starting resident command before
launch. Capture c16/batch32 with prefix caching off and resumable batch/tensor
hashes. Stop on any identity/coverage drift, failed gate, nonfinite output, OOM,
context overflow or completion; preserve failures. Collect raw BF16 residuals
losslessly in FP32 and independently recompute dots, paired deltas and rankings.
Report prompt tokens/s, requests/s and queued latency as one-pass diagnostics.
No training, data deletion, new capacity or package changes. Merged weights
remain remote. Monitoring is active-turn only; no heartbeat tool is available.

Run `.prepare` then `.run` with the pinned serving runtime. Config owns paths;
artifacts use `results/b200_training_direction/training_direction01/`.
