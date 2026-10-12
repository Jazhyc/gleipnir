# Native MXFP8 backward quality test

Hypothesis: pairing the serving-matched MXFP8 forward with native MXFP8 backward
reduces full-model update time without an additional ID ranking regression.
The completed native-FP8 projection campaign is the matched control: retain its
initial FP32 adapter, cached BF16 teacher, strict-filtered 7,836 rows, seed/order,
245 updates, KL T1, optimizer, packing, projections and MLP/GDN optimizations.
Change only attention forward statistics/backward. The forward emits matching
LSE using the hardware-exp2 serving arithmetic; backward reuses packed native
dQ/dK/dV with the recorded singleton STE and initialized scratch.

Reuse the successful mixed-length/8K attention probes and unchanged projection
operator receipts. Before updates, freshly check simulator/server alignment,
packed sequence isolation, whole-model initial-adapter gradients on two short
training rows, and the 29,337-token longest-row backward/memory envelope. Compare
all adapter gradients to the existing FA4 surrogate and report aggregate and
per-tensor differences; numerical gradient reproduction is diagnostic, not an
acceptance gate for this explicit approximate-gradient variant. Require identical
short-row forward logits, finite FP32 gradients for every trainable tensor and
restoration of RNG, gradients and model mode after the paired check.
Stop on drift, missing/nonfinite gradients, changed forward logits, isolation
failure, unsupported execution, OOM or failed alignment. Cap each stage at two
hours. Preserve failures; do not substitute a different backward mid-run.

Automatically run one full epoch, preserve its FP32 master, overwrite the owned
ephemeral merged-model slot after checking the native-FP8 predecessor, validate
new-adapter serving parity, and run two frozen 3,012-row ID passes. Compare cached
native-FP8 and original QAD results; no repeated control training, new teacher
calls, Kimi training, OOD scoring or automatic promotion. The descriptive gate
is mean source-macro raw normalized pAUROC@20 no more than 0.005 below the matched
native-FP8 mean 0.86199027. Passing that gate does not erase its existing regression
against original QAD's 0.87608417. Select the fixed final epoch only.

Reuse the existing authorized NC2 B200 and shared persistent caches. Launch with
`python -m experiments.native_mxfp8_qat.launch`; the generic archived-stage driver
is `experiments.native_fp8_qat.run`. Sources and exact substitutions are frozen
under `results/native-mxfp8-qat01/`; logs are under
`logs/runpod/native-mxfp8-qat01/`. No in-chat scheduler is available: detached
stage automation does not promise later agent wakeups. Check startup until
preflight passes and actual updates advance.
