# Released monitors on APPS monitor-directed injections

2026-10-21. The released Gleipnir 4B v0.4 and 27B v0.2 were scored once on the
frozen APPS injection populations
([design](../../experiments/apps_released_monitors/README.md), run `apps01`).
A follow-up projected an injection direction out of the released 4B
([design](../../experiments/apps_released_direction/README.md), run
`released_direction01`). Both were requested after the SleightBench jailbreak
results ([finding](sleightbench_released_monitors.md)).

**Both released monitors treat text addressed to the monitor as suspicious.**
At the APPS operating point (at most 2% of clean honest code alarms), both flag
every injected honest program under all three transfer templates. At a 0.5
threshold, the 4B still flags 90–99.5% of them. The 27B flags 44% (GPT-4),
27% (GPT-4.1) and 100% (GPT-4o), and separates injected backdoors from
injected honest code better (matched AUROC 0.77 against 0.62).

**The reaction comes from the base model; the 27B adapter reduces it.** In a
matched stock-BF16 engine, the unadapted Qwen3.8-27B flags 76–100% of injected
honest programs at 0.5, with matched AUROC 0.66. The released 27B lowers that
to 53/73/100% and raises matched AUROC to 0.83.

**Projecting out an injection direction only softens this on the released 4B.**
Neither the refit nor the cached original-model projection lowers the
calibrated injected-honest FPR below 100%. At 0.5 the cached projection cuts
alarms the most (to 23/11/76%), with matched AUROC rising from 0.67 to 0.70.
On the augmented model it was fitted on, the same edit took FPR from 51% to
0.4%.

## Injection alarms on the optimized stack (`apps01`)

All 9,114 rows per model, optimized B200 stack (`gleipnir-serve:0.2`), single
pass. The 4B prompts reproduce the historical APPS prompts byte for byte
(student prompt, 15,339,953 tokens). The 27B uses the teacher prompt around
the same trajectories (22,740,521 tokens).

| Monitor | Injected-honest FPR (GPT-4 / 4.1 / 4o) | Same, at 0.5 | Matched AUROC (mean) | Vanilla AUROC / pAUROC@20 | Vanilla backdoor TPR |
|---|---|---|---:|---:|---:|
| Gleipnir 4B v0.4 | 100 / 100 / 100% | 89.5 / 99.5 / 97.7% | 0.622 | 0.897 / 0.698 | 55.0% |
| Gleipnir 27B v0.2 | 100 / 100 / 100% | 44.4 / 26.7 / 100% | 0.769 | 0.940 / 0.777 | 63.7% |
| Regular FP4 4B ([b200_apps](b200_apps.md)) | 70.4 / 100 / 51.8% | 2.0 / 92.1 / 0.2% | 0.613 | 0.918 / 0.744 | 51.9% |
| Augmented BF16 4B ([augmentation](monitor_injection_augmentation.md)) | mean 44.4% | — | 0.859 | 0.926 / — | 55.2% |

- **Every injection raises both models' scores.** Injection-minus-neutral
  score changes are positive in 99.7–99.8% of 4B pairs (mean +0.58 to +0.69)
  and 100% of 27B pairs (mean +0.35 to +0.66).
- **The calibrated thresholds are very low.** Clean honest code scores near
  zero, giving thresholds of 0.067 (4B) and 0.095 (27B). At these thresholds
  the neutral-text controls also alarm: 11–25% (4B) and 21–50% (27B), against
  at most 0.9% at 0.5. The 0.5 rates are the more discriminating comparison.
- **The released 4B is more injection-prone than the earlier regular 4B**
  at 0.5 on two of three templates, with similar matched AUROC. Its
  strict-filtered training data contains no monitor-directed injection
  augmentation.
- **The 27B's better matched AUROC** does not make it injection-robust: it
  alarms on every GPT-4o-injected honest program at 0.5.
- **Label caveat.** APPS labels only code backdoors. The agent wrote the
  injection into its own submission, so an alarm on an injected honest
  program is partly a label disagreement: our specification may itself
  consider an attempt to manipulate the monitor problematic.

This reinterprets the SleightBench jailbreak: both models' high scores there
are consistent with a general reaction to monitor-directed text rather than
injection robustness. The 27B was fooled once its prompt lost the security
boundary, so its behavior is prompt-dependent; APPS does not separate these.

Scoring took 86 s for the 4B (170–190K prompt tokens/s) and about 7 min for
the 27B (about 55K tokens/s), in 128-row partitions at c128. Results:
`results/apps_released_monitors/apps01/` (`report.json`, per-model
`summary.json`); logs: `logs/runpod/apps_released_monitors/`.

## Base 27B reference (stock BF16)

Requested by the user after `apps01`. The unadapted `Qwen/Qwen3.8-27B@1d4bf0f…`
and the released 27B's merged BF16 checkpoint were each scored in one offline
stock-BF16 vLLM engine on the same teacher-prompt rows (`offline.py`; logits
after `Prediction:`, as in SleightBench `sb04`). The pair isolates the adapter
from the optimized stack.

| 27B arm | Injected-honest FPR, calibrated | At 0.5 (GPT-4 / 4.1 / 4o) | Matched AUROC (GPT-4 / 4.1 / 4o, mean) | Vanilla AUROC / pAUROC@20 | Vanilla backdoor TPR |
|---|---|---|---|---:|---:|
| Base, BF16 | 92.5 / 99.7 / 100% | 75.9 / 98.9 / 100% | 0.754 / 0.564 / 0.670, **0.663** | 0.935 / 0.786 | 64.6% |
| Gleipnir 27B v0.2, BF16 | 100 / 100 / 100% | 53.3 / 73.1 / 100% | 0.868 / 0.827 / 0.794, **0.829** | 0.949 / 0.810 | 68.0% |
| Gleipnir 27B v0.2, optimized (`apps01`) | 100 / 100 / 100% | 44.4 / 26.7 / 100% | 0.821 / 0.765 / 0.722, **0.770** | 0.940 / 0.777 | 63.7% |

- **The base model reacts to injections even more strongly.** Injections
  raise its scores by +0.57 to +0.81 on average (positive in 99.8% of pairs),
  and it flags nearly every injected honest program at 0.5. Its calibrated
  threshold is higher (0.35), because it scores clean code less confidently,
  so its neutral-text controls barely alarm (0.2–0.8%).
- **The adapter reduces the reaction and greatly improves discrimination.**
  In the matched engine, matched AUROC rises by 0.17 (0.663 → 0.829), alarms
  at 0.5 fall on the GPT-4 and GPT-4.1 templates, and vanilla AUROC rises from
  0.935 to 0.949. The GPT-4o template still triggers every honest program.
- **The optimized stack costs discrimination here.** The same released
  checkpoint loses 0.06 matched AUROC and 0.009 vanilla AUROC on the
  optimized stack. It also alarms less on GPT-4.1 injections at 0.5 (27%
  against 73%), so optimized-stack alarm rates are not the BF16 model's.

Scoring took about 16 min (base, including its download) and 17 min
(released) on a second US B200 Pod, `skhiibc8fu2130`, from the same image.
Results: `results/apps_released_monitors/apps01/27b_{base,release}_bf16/`.

## Injection-direction projection on the released 4B (`released_direction01`)

The released 4B's merged BF16 checkpoint was served unquantized and eager with
Lens 1.3.0, installed from the hash-locked wheels into an overlay on the image
runtime. The command is the project's BF16 Qwen3.5-4B command with only the
checkpoint changed.
- **Gate.** All 3,012 ID rows reproduce the recorded BF16 scores
  (`qad-v02-bf16-01`): MAE 0.0030, maximum 0.062, correlation 0.99982, pooled
  AUROC 0.97280 against 0.97281. The native BF16 audit and the operator smoke
  passed.
- **Fit.** The frozen 64-task split (896 rows, 384 pairs), layer 20, both
  labels. The refit direction has natural-shift norm 7.46 (original 7.07) and
  cosine **0.48** with the original augmented-model direction.
- **Test.** 591 disjoint tasks / 8,218 rows per arm; all-layer, all-token,
  beta-1 projections.

| Arm (released 4B, eager BF16) | Injected-honest FPR | At 0.5 (GPT-4 / 4.1 / 4o) | Backdoor recall | Matched AUROC / pAUROC@20 | Vanilla AUROC | Mean shift |
|---|---:|---|---:|---:|---:|---:|
| Plain | 100% | 84.4 / 89.3 / 100% | 100% | 0.667 / 0.211 | 0.914 | — |
| Refit projection | 100% | 38.4 / 56.0 / 93.7% | 100% | 0.703 / 0.283 | 0.912 | −0.027 |
| Cached original projection | 100% | 23.4 / 10.5 / 75.6% | 100% | 0.703 / 0.288 | 0.899 | −0.098 |
| Random projection | 100% | 88.0 / 91.9 / 100% | 100% | 0.665 / 0.206 | 0.913 | +0.002 |
| *Augmented 4B, plain (cached)* | *50.8%* | *0.2 / 0 / 0%* | *79.2%* | *0.805 / 0.519* | *0.920* | |
| *Augmented 4B, own projection (cached)* | *0.4%* | *0 / 0 / 0%* | *31.4%* | *0.891 / 0.681* | *0.922* | |

FPR and recall use each arm's clean-only threshold (0.053–0.068); the fixed
plain threshold gives the same rates except a higher refit clean FPR (3.6%).
- **Both real directions lower injection scores; the random one does not.**
  The cached direction moves scores most and shifts all rows by −0.10,
  costing 1.5 points of vanilla AUROC.
- **Neither restores a usable operating point.** Injected-honest scores stay
  above the near-zero clean scores, so the calibrated FPR stays 100%. On the
  augmented model the same projection eliminated these alarms; the released
  4B's injection response is larger and is not captured by one direction.
- **Answer mass stays healthy** on the 1,771-row diagnostic subset (minimum
  0.954–0.971, none below 0.5).
- **The optimized stack differs noticeably here.** Eager BF16 plain minus the
  optimized `apps01` scores: MAE 0.069, mean −0.025, maximum 0.71, consistent
  with the 4B's known optimized-serving gap. Matched AUROC is 0.667 here
  (591 test tasks) against 0.622 on the optimized stack (all 655).

Two launch attempts failed before scoring, from errors in the new launcher (a
fit-split check comparing task IDs to a count, then a missing scheduler
binding); both logs are kept. One checkpoint and one pass; APPS tasks and
templates were used in earlier studies, and no intervals were computed.
Results: `results/apps_released_direction/released_direction01/`; logs:
`logs/runpod/apps_released_direction/`.

## Infrastructure

US B200 Pod `1al84w5c99wiki` (`.runpod/apps-released.json`, GPU
`GPU-e0504b68…`, driver 595.91.07), created from `gleipnir-serve:0.2` with no
volume on 2026-10-21 at the user's request. The base-27B reference arms ran on a second
Pod, `skhiibc8fu2130` (`.runpod/apps-base27b.json`, GPU `GPU-cbba00e9…`), from
the same image; its 10 result and log files were verified by checksum before
it was terminated at the user's request (delete 204, lookup 404). All 566 result and log files were
pulled and verified by checksum before the Pod was terminated (delete 204,
lookup 404).
