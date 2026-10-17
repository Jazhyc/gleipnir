# Qwen 27B generations on the strict OOD suite

Status: complete, 2026-10-17. Unadapted BF16 checkpoints, matched to the frozen
Qwen3.5-27B full-teacher-prompt contract. See the
[experiment](../../experiments/qwen27b_generations_ood/README.md) for the
per-source breakdown, the interface audit and the drift control.

Mean-OOD pAUROC@20 (6,395 rows, six sources):

| Model | Mean-OOD pAUROC@20 |
|---|---:|
| Qwen3.5-27B | 0.8388 (2026-08-31 result: 0.8395) |
| Qwen3.6-27B | 0.8726 |
| **Qwen3.8-27B** | **0.8901** |
| Kimi K3 teacher | 0.9084 |

**Qwen3.8 leads.**
- It has the best value on five of six sources: IaC rises from 0.886 to 0.968
  and AgentDojo from 0.913 to 0.966.
- It has the best calibration: macro Brier 0.082, against 0.105 for 3.6 and
  0.107 for 3.5.

**Qwen3.6 leads only on SHADE-Arena:** 0.596, against 0.556 for 3.8 and 0.494
for 3.5.

**Same interface.** All three render every prompt identically, with the same
token counts and decision tokens.

**Unchecked for BF16.** The earlier FP8 Qwen3.8 batch/singleton consistency
failure ([ID screen](../../experiments/qwen122b_id/README.md)) has not been
checked for this BF16 checkpoint.

**ID and training rows (2026-10-17).** On ID, unadapted Qwen3.8-27B scores
0.8869 macro pAUROC@20, against 0.896–0.914 for the trained 4B students. On
all 8,688 training rows it scores 0.9298, against 0.9636 for Kimi K3; the gap
is mostly BashArena. On the 852 rows the strict filter removes, it lands on the
label's side for only 33%, so it mostly shares Kimi's errors there.

The choice of starting checkpoint is the user's.
