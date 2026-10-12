# Direct Kimi K3 QAT

The user selects [direct Kimi-target QAT](../../experiments/kimi_direct_qat/README.md)
from a fresh base-model adapter, rather than recovery from the trained BF16
student. The frozen original zero-B rank-128/alpha-256 initialization is reused;
its weight SHA256 is
`5da72480668385805032c5039e5c4a29e7f6257762185da79d13a5c70443f3b0`.
Training consumes the cached original `moonshotai/kimi-k3` distributions on the
7,836 strict-filtered examples, preserving privileged teacher prompt hashes
separately from student prompts. No BF16 student target cache or teacher API calls
are used. The one-epoch schedule is 245 updates, KL T1 and AdamW 5e-5, matching
direct distillation rather than trained-adapter recovery's 1e-5.

Use the [selected QAT arithmetic](../decisions/qat_reference_training.md), with
unused MLP backward output recomputation removed and frozen GDN work shared.
FP8 projection contractions retain the decoded FP32 reference; attention uses
serving MXFP8 forward and BF16 FA4 surrogate backward. FP32 masters, BF16 FlashQLA,
packing/context and final-epoch selection stay fixed. The cached original QAD,
direct-distillation optimized baseline and BF16 serving baseline are descriptive
comparisons; initialization, teacher, schedule and total training budget differ.
No ID tuning, OOD evaluation or automatic promotion occurs.

Seventeen focused CPU tests and Ruff pass. Live SSH verifies the existing NC2 B200
idle with the expected UUID before launching the first attempt as
`kimi-direct-qat01`. Target identities, zero-B weights and all source archive
hashes are verified locally. The twenty unadapted-base canaries fail initial
simulator/server alignment: MAE 0.06541767, correlation 0.93749445, maximum
difference 0.22949348. The first attempt stops before any optimizer update.
Its original config, source archives, failure and serving logs are preserved.

A fresh server repeats that failure. Repeat one has the same values; warmed
repeats two and three have MAE 0.04797230, correlation 0.96064459 and maximum
difference 0.20578503. Thus serving variability contributes, but does not explain
the entire mismatch. This is new evidence at the untrained initialization;
the earlier trained-adapter recovery canary does not establish fresh-base parity.

The separately frozen `kimi-direct-qat02` attempt, driver PID 10508, follows the
standing reuse of validated arithmetic: initial fresh-base score alignment is an
explicit failed diagnostic, not a pass or a kernel-validation replacement.
It reuses the checksummed initial server canary without more serving startup,
retains source-bound operator/structural receipts, and freshly checks packing
isolation and longest-row finite gradients/memory. Per-update finite/missing-gradient
checks and final trained-adapter serving parity/nonzero effect remain mandatory.
The final-epoch ID quality gate is unchanged; no ID results inform this change.
Two finite optimizer updates are collected, consuming Kimi targets directly.
The separate/packed short-row probabilities match exactly; the 29,337-token
backward passes with loss 0.25826618, finite unclipped gradient norm 61.30841 and
peak allocation 137,556,206,080 bytes (128.11 GiB). Kimi target identity, source
archives and matched seed-zero training order are independently verified locally.
Initial real-update KL is 0.646350/0.492393, with finite unclipped norms
38.17382/38.38422; updates retain the declared norm-1 clipping. These losses use
Kimi targets and are not comparable to recovery KL against the BF16 student.
Completed training and quality results are pending. Fresh-base mismatch remains
a limitation to report even if the final trained adapter passes parity.

## Larger-packing follow-up

CPU-only replay of the frozen 245 logical batches counts 4,139 physical passes
at 16,384 tokens, 3,657 at 20,480, 3,221 at 24,576 and 2,482 at 32,768. Thus
24K/32K remove 22.18%/40.03% of physical passes while retaining logical batch 32;
this is not a GPU timing or numerical-equivalence result. The largest physical
row stays 29,337 through 24K because long singletons already exceed the budget;
32K permits 32,768-token rows. Receipt: `packing_budget_counts.json` under the
active output. The active 16K campaign stays frozen. A later matched screen must
check packed-boundary isolation, finite gradients/memory and changed quantized
kernel/numerical behavior before treating fewer passes as a throughput gain.

Sources/contracts and receipts: `results/kimi-direct-qat02/`; logs:
`logs/runpod/kimi-direct-qat02/all.log`. Failed initial alignment and repeats stay
under `results/kimi-direct-qat01/`. The automatic pipeline trains, preserves
FP32 weights, replaces the one owned ephemeral merge, checks parity and scores
two frozen ID passes or stops with a preserved failure receipt. Prior masters,
scores and shared compiler caches remain intact. No in-chat scheduler is available;
detached automation does not provide follow-up agent wakeups.
