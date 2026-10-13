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
All 245 updates and both frozen 3,012-row ID passes complete. Fresh-base mismatch
remains a reported limitation; final trained-adapter serving parity passes.

## Completed quality comparison

Training takes 3,928.68 seconds (65.48 minutes), versus 3,955.95 seconds
(65.93 minutes) for the original recovery QAT stage. Direct training removes the
preceding trained-teacher stage and scoring dependency; it does not materially
reduce QAT epoch time. Fresh initialization, targets, learning rate and total
training budget differ, so this is not an isolated teacher or speed comparison.

| Model, optimized serving | Macro raw pAUROC@20, two passes | Pooled AUROC, two passes |
| --- | ---: | ---: |
| Original direct Kimi baseline, selected FP4/FA4 training | 0.850308 / 0.847754 | 0.941193 / 0.940109 |
| Original two-stage QAD | 0.875735 / 0.876433 | 0.949218 / 0.949311 |
| Direct Kimi QAT | 0.806866 / 0.806531 | 0.921229 / 0.920517 |

Mean macro pAUROC is 0.806698, 6.9386 percentage points below two-stage QAD and
4.2332 points below the original direct-distillation optimized baseline. This
fails the predeclared 0.5-point quality gate; no model or recipe promotion occurs.
Most regression against QAD is in Gloom: pAUROC is 0.660763/0.660950 versus
0.793433/0.793867. STRIDE is 0.952969/0.952112 versus 0.958037/0.959000.
The result does not establish that Kimi is a worse teacher: the recovery model
also receives a preceding full training epoch and uses a different schedule.
The direct-baseline comparison retains Kimi/init/schedule but changes the combined
training arithmetic and implementation. Both are one-seed comparisons; no OOD
transfer is measured.

Pooled Brier is 0.113538/0.113700 and ECE 0.054632/0.051917. Optimized serving
input throughput is 175,707/176,526 tokens/s. Repeat score MAE is 0.016542,
maximum difference 0.377374, correlation 0.994777 and 58 threshold flips.
Pooled unique scores are 93/95. All three final-adapter serving canaries pass:
MAE 0.004004, correlation above 0.99992, maximum difference 0.019022 and
nonzero adapter effect 0.926617. This is much closer alignment than the untrained
initialization; short-canary acceptance does not establish complete-ID parity.

The FP32 master is preserved remotely and locally, SHA256
`8e6cf6fd0f3c9972b1c970d804ee6c82c61fbad4442480b0d69a9f0b0797aa96`.
Independent verification checks complete finite updates, FP32 tensor types,
master/source hashes, original privileged Kimi target identity, matched training
order and ID/prompt/source-label coverage, with exact metric recalculation.
The receipt is `results/kimi-direct-qat02/verified_comparison.json`. The final
server exits; the B200 is idle and remains running, with prior masters and caches
preserved.

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
