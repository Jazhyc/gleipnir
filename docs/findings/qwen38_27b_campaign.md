# First trained Qwen3.8-27B monitor

Status: ID and OOD complete 2026-10-18.

Qwen3.8-27B trained on the strict-filtered Kimi K3 soft targets with the
standard one-epoch recipe (seed 0, `qwen38_27b_b200_default`) reaches **0.9217
ID macro pAUROC@20** (AUROC 0.9822), the best ID result so far: +3.5 points
over the unadapted 27B (0.8869) and +2.6 over the same-seed 4B student
(0.8959). Against the best 4B seed (0.9142) the margin is within 4B seed
variation. The gain is concentrated in Gloom exfiltration (0.887); Test STRIDE
(0.957) stays below the 4B seeds. See the
[experiment](../../experiments/qwen38_27b_campaign/README.md).

On the frozen OOD suite it reaches **0.9087 Mean-OOD pAUROC@20**, level with
its teacher Kimi K3 (0.9084) and +1.9 points over the unadapted 27B (0.8901):
training improved every unsaturated source rather than trading OOD for ID. It
leads Kimi on SHADE-Arena (0.588 vs. 0.575) and Rogue Eval, and is 9.6–13.2
points above the 4B students (0.777–0.813). One seed; the 4B OOD seed SD is
1.8 points.

The run took 4.5 h of training on one B200, plus about 45 minutes for merge,
parity and ID scoring and 46 minutes for OOD scoring. Training loss converges
to Kimi's target entropy (0.129 vs. 0.128).
