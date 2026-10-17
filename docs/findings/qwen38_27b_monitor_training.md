# Qwen3.8-27B monitor training (first campaign)

Status: ID scored 2026-10-17; OOD pending.

Distilling the Kimi K3 soft targets into Qwen3.8-27B with the 4B recipe
(strict-filtered rows, one epoch, seed 0, `qwen38_27b_b200_default`) gives
**0.9217 held-out ID macro pAUROC@20** (AUROC 0.9822). That is +3.5 points
over the unadapted 27B (0.8869) and +2.6 over the same-seed 4B student
(0.8959). Against the best of three 4B seeds (0.9142) the gain is within 4B
seed variation.

The improvement comes mostly from Gloom exfiltration (0.887, best of any
model) and macro AUROC; on Test STRIDE the trained 27B (0.957) stays below all
three 4B seeds (0.963–0.979). Training loss ends at Kimi's target-entropy
floor, so further gains from the same targets are unlikely to come from
longer training.

Qualifications: one 27B seed; ID only, which is in Kimi's distribution; no OOD
result yet, where the unadapted 27B (0.890) already sits near Kimi K3 (0.908).

Details: [campaign](../../experiments/qwen38_27b_campaign/README.md);
[training recipe](../decisions/qwen38_27b_b200_training_default.md).
