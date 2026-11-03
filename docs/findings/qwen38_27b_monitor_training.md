# Qwen3.8-27B monitor training (first campaign)

Status: soft seed 0 ID and OOD scored 2026-10-17/18; hard-label baseline
scored 2026-11-03. All scores use BF16 vLLM 0.31 serving.

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
the same source labels they train on. OOD differs by 0.4 points, within 4B
seed noise; hard loses IaC (−4.4) and gains SHADE-Arena (+2.3). Soft targets
keep better calibration, and hard scores saturate (79% of OOD scores within
0.01 of 0 or 1). At 27B, Kimi's soft targets therefore mainly buy calibration,
not OOD ranking; at 4B they buy both.

Qualifications: one seed per arm; the hard arm uses the untuned soft recipe;
the unadapted 27B and Kimi were scored under the teacher prompt, the trained
models under the compact student prompt.

Details: [soft campaign](../../experiments/qwen38_27b_campaign/README.md);
[hard-label baseline](../../experiments/qwen38_27b_hard_labels/README.md);
[training recipe](../decisions/qwen38_27b_b200_training_default.md).
