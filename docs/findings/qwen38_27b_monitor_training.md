# Qwen3.8-27B monitor training (first campaign)

Status: soft seed 0 ID and OOD scored 2026-10-17/18; hard-label baseline
scored 2026-11-03/04. The table uses BF16 vLLM 0.31 and the student prompt;
the release-configuration comparison follows it.

Distilling the Kimi K3 soft targets into Qwen3.8-27B with the 4B recipe
(strict-filtered rows, one epoch, seed 0, `qwen38_27b_b200_default`) gives
**0.9217 held-out ID macro pAUROC@20** (AUROC 0.9822) and **0.9087 Mean-OOD**.
That is +3.5 ID and +1.9 OOD points over the unadapted 27B (0.8869 / 0.8901),
ties Kimi K3 on OOD (0.9084), and is 9.6–13.2 OOD points above the 4B
students. On ID the gain over the best 4B seed (0.9142) is within 4B seed
variation. Training loss ends at Kimi's target-entropy floor.

**Hard labels at 27B.** The matched hard-label arm (same rows, seed,
initialization and recipe; source labels only) reverses the 4B pattern on ID
and nearly ties on OOD:

| | ID pAUROC@20 | OOD pAUROC@20 | OOD Brier |
|---|---:|---:|---:|
| 27B soft | 0.9217 | **0.9087** | **0.0574** |
| 27B hard | **0.9665** | 0.9044 | 0.0712 |
| 4B soft − hard (3 seeds) | +0.069 | +0.076 | −0.130 |

Hard labels gain 4.5 ID points (both sources), which measure agreement with
the same source labels they train on. Under the student prompt with BF16
serving, OOD differs by only 0.4 points; hard loses IaC (−4.4) and gains
SHADE-Arena (+2.3). Hard scores saturate (79% of OOD scores within 0.01 of 0
or 1) and calibrate worse.

**In the release configuration** (optimized stack, full teacher prompt) soft
targets win OOD clearly: 0.9508 vs 0.9328 (IaC −3.5, SHADE-Arena −7.6), while
hard labels still lead ID (0.9587 vs 0.9444) with worse ID Brier (0.091 vs
0.064). The released soft checkpoint remains the better 27B monitor; the
student-prompt BF16 near-tie understated the OOD value of soft targets.

Qualifications: one seed per arm; the hard arm uses the untuned soft recipe;
in the table the unadapted 27B and Kimi were scored under the teacher prompt
and the trained models under the compact student prompt; the release
comparison scores both trained models under the teacher prompt.

Details: [soft campaign](../../experiments/qwen38_27b_campaign/README.md);
[hard-label baseline](../../experiments/qwen38_27b_hard_labels/README.md);
[training recipe](../decisions/qwen38_27b_b200_training_default.md).
