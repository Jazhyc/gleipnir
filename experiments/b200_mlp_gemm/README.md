# B200 MLP/GEMM optimization

Hypothesis: merging the two frozen gate/up GEMMs and compiling the complete
LoRA MLP, or fusing the GEMMs/LoRA adds/SwiGLU with cuDNN, reduces full MLP
forward-plus-backward time and then complete FA4 training update time.
Keep the original BF16 frozen base, FP32 rank-128/alpha-256 adapters, native
causal variable-length FA4 4.0.0b33, all 24 FlashQLA GDN layers, no model
checkpointing, logical batch 32 and the 16,384-token packing budget.

First compare the actual PEFT module contract with nonzero synthetic adapters
using the cached Qwen3.5-4B configuration. Compare eager, compiled eager,
merged gate/up, compiled merged gate/up and cuDNN LoRA-aware forward fusion.
Include all adapter matmuls, casts and backward operations in step timings;
report forward separately. The merged frozen-weight copy is prepared once,
retained as a nonpersistent buffer and explicitly counted as extra memory.
The original masters, state-dict names and frozen parameters are untouched.
cuDNN fusion must add both adapter updates before SwiGLU and preserve the
BF16 rounding boundaries. Its initial backward uses ordinary BF16 GEMMs and
SiLU backward, rather than claiming a fused native training backward.

Use alternating measurement order, six warmups and ten timed repetitions at
193, 4,096 and 16,384 tokens. Require finite outputs, input and all six adapter
gradients, output relative L2 <= 1%, and each gradient relative L2 <= 1% versus
the actual eager PEFT MLP. Include an independent row-isolation check. Stop a
candidate on compile failure, nonfinite/missing gradients, failed arithmetic,
OOM, or ten-minute timeout, and preserve its receipt. Pilot weights are seeded
synthetic tensors: these are diagnostic timings, not checkpoint quality tests.

The default decoder shells already compile MLP operations. Compiled merged
and cuDNN candidates therefore time against a compiled PEFT MLP in the same
process, with eager PEFT retained as the arithmetic reference. Only a passing
candidate with at least 5% faster full forward/backward against this compiled
control at 4,096 and 16,384 tokens advances to a bounded full-model screen. Freeze the
same 320-row cohort and initial adapter used by the recorded FA4 control;
run at most 20 updates, ten warmup and ten measured. A changed MLP requires
fresh packing/isolation/memory gates, with strict 5% packing results separate
from the user-accepted 10% FA4 gradient ceiling. Preserve original failed
strict receipts and reject missing/nonfinite gradients before updates.
Compare exact physical partitions/tokens against the historical FA4 control
and require >= 5% lower complete-update time to recommend follow-up.
No held-out selection, teacher requests, final-test access or default promotion.

Use the user-authorized NC2 B200 and retained network volume/compiler caches.
Outputs: `results/b200_mlp_gemm/`; logs: `logs/runpod/b200_mlp_gemm/`.
There is no in-chat scheduling tool; inspect startup/progress during the active
turn without promising checks after the turn ends. Keep BF16 FA4 as the default
until results support a change. Native source is the existing isolated NVIDIA
cuDNN Frontend revision `51d9d06b574222378a3d806009accab098e73705`.

## Native FP4 dense-GEMM screen

The next hypothesis is that native NVFP4 forward **and input-gradient** GEMMs
can reduce the dominant frozen-base matrix cost after leaving QLoRA, including
the activation packing/scaling overhead that the previous forward-only FP4
screen did not offset. Probe the merged gate/up and down projections and their
transposed input-gradient projections at actual 4B geometry (2560 / 9216), with
193, 4096 and 16384 tokens. Synthetic seeded weights and activation/gradient
operands are used; these are not whole-MLP or training-update measurements.

Use the retained pinned NVIDIA cuDNN Frontend/FROST overlay and Torch runtime.
Weights use global FP32 scaling plus E4M3 scales shared over 16x16 tiles, with
nearest-even E2M1 packing from the original BF16 tensors in both orientations.
Require exact agreement of the decoded forward/backward weight transpose.
Activations and input-gradient operands use dynamic global scaling and per-row
16-element E4M3 scales. Output is BF16. This bounded timing probe uses RNE for
gradients; it does not claim Transformer Engine's stochastic gradient rounding,
Hadamard transforms or full NVFP4 training recipe. FP32 adapter masters and the
selected BF16 FA4 training recipe are unchanged.

Compare paired ordinary BF16 GEMMs, native FP4 with prepacked operands, and
native FP4 including all dynamic activation reductions, allocation, packing,
scale swizzling, GEMM and output descaling. Pack frozen weights once; record
its startup time and resident packed-pair bytes. Six warmups and ten alternating
synchronized wall-time samples are used. Quantization is lossy: separately
record original-BF16 error and require native-versus-decoded-operand relative
L2 <=1%, finite outputs, and exact decoded weight-transpose consistency.
These checks establish arithmetic implementation, not training quality or the
existing strict 5% packed-gradient acceptance. No objective or tolerance changes.

Stop on arithmetic failure or after the twelve shape/path cases, with a
20-minute process cap. Advance to complete LoRA MLP integration only if at least
one path improves by 5% at both long shapes with packing included. Any later
full-model trial needs fresh precision/packing checks and the historical FA4
control contract. No teacher calls or held-out selection occurs in this screen.

After the first FP4 probe passes all twelve arithmetic checks but fails the
speed threshold, the bounded optimized follow-up replaces the Torch global
amax/inverse sequence with two Triton reductions. Check codes, block scales
and inverse scale against the ordinary reduction before timing. Preserve the
initial failed speed receipt. Also capture matched BF16 and FP4 paths in CUDA
graphs, including the caller-to-static input copy in both replay timings. Check
ordinary and changed-input FP4 replay against uncaptured execution so scales
cannot become stale. Dynamic amax, packing, scale clearing, GEMM and output
descaling remain inside capture. The same arithmetic limits, shapes, warmups,
repetitions and 20-minute stop apply. Graph replay measurements are a bounded
implementation feasibility result; integration with dynamically packed training
is not established, and a graph-only speedup does not automatically authorize
whole-model promotion or bypass full-MLP measurement.

The final packing follow-up replaces tensor-wide activation amax with a per-row
amax, block-scale computation and E2M1 packing in one Triton kernel. Tensor-wide
scaling can make one packed example's quantization depend on another's values;
per-row global inverse scales are applied in the BF16 output-descaling kernel.
Frozen weight scaling remains global and 16x16. Record perturbing the first row
by 31.7x and require exactly zero effect on all other rows before timing each
path. Preserve earlier global-scaling receipts rather than attributing their
speed to an already isolated training implementation. Repeat all twelve native
arithmetic, finite, transpose and changed-input graph checks, with matched
BF16/FP4 graph copies and full conversion included, under the same stop rules.
Full LoRA MLP, autograd/compiler integration and packed model validation remain
subsequent gates; these isolated native kernels do not change the default.

A bounded chunked-row follow-up tests whether separating row amax reduction
from block packing helps the wide 9216/18432-element operands. Keep identical
per-row inverse scales, E4M3 scales and E2M1 codes; require bitwise equality of
all three against fused-row packing before timing every case. Pack independent
128-block chunks rather than carrying the whole padded row through quantization
in one CTA. Record Triton register/spill metadata for both implementations to
check the suspected register-pressure cost instead of attributing slowdown to
launch count alone. Repeat the same twelve cases and graph/isolation gates with
no promotion based on isolated GEMM timing.
