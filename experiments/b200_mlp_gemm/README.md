# BF16 MLP/GEMM optimization on B200

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
