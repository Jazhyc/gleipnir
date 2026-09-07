# Pretrained-only Qwen3.5 ID evaluation

Completed and audited 2026-09-07. No new training, prompt tuning, or OOD access.

## Result

All values below are unweighted means across Gloom-Exfiltration (2,066 rows)
and STRIDE test (946 rows). ECE uses ten equal-width positive-class bins.
pAUROC@20 is raw area up to FPR .2 divided by .2, not sklearn-standardized AUC.

| Model | Macro AUROC | Macro pAUROC@20 | Macro Brier | Macro ECE | Macro log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-trained 4B, no adapter | .860769 | .604755 | .231949 | .245040 | .696290 |
| Pretrained-only 4B-Base | .751012 | .404095 | .247677 | .216087 | .695349 |
| Post-trained 9B, no adapter | .898587 | .663440 | .157127 | .113739 | .484722 |
| Pretrained-only 9B-Base | .798188 | .475822 | .204754 | .129477 | .591537 |

| Model | Gloom AUROC | STRIDE AUROC | Macro balanced accuracy | Macro recall | Macro FPR | Unique scores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Post-trained 4B | .774628 | .946910 | .697290 | .429187 | .034606 | 1173 |
| Pretrained-only 4B | .608219 | .893804 | .623985 | .369545 | .121574 | 384 |
| Post-trained 9B | .858621 | .938552 | .796879 | .711856 | .118098 | 1023 |
| Pretrained-only 9B | .685260 | .911116 | .692157 | .527428 | .143113 | 346 |

Threshold diagnostics use .5. Full ECE(5/10/20) reliability bins and pooled/source
calibration are in each arm's calibration.json; ranking and ties in result.json.

## Interpretation

This does not support substituting pretrained-only weights for the post-trained
zero-shot monitor under the deployed compact rubric. Discrimination worsens at
both sizes and on both sources, especially Gloom. The 4B ECE reduction alone is
not a useful probability-quality win: Brier worsens and log loss is essentially
unchanged. The 9B model worsens in ECE, Brier and log loss as well as ranking.
Less post-training does not automatically yield a better monitoring probability.

This is not a comparison of which initialization would train the better monitor.
Zero-shot rubric elicitation and learned monitoring representations are separate
questions. No fine-tuning of the pretrained checkpoints has been tested here.

## Audit and runtime

Both successful Slurm jobs exited 0 with all 3,012 unique, finite predictions.
Every ID, source, label and rendered margin-prompt SHA-256 matches its historical
post-trained comparator. Raw decision logprobs are finite throughout. Model
revisions and input/prompt checksums are pinned in experiments/pretrained_id/config.yaml.
No adapter or quantization is used: BF16, text-only vLLM 0.24.0, Torch 2.11.0+cu130.

Both five-row source/label-plus-longest canaries passed. Batch/singleton mean
absolute score differences were .00275227 (4B) and 9.94e-10 (9B); maxima .01376132
and 4.97e-9. These are batch-consistency checks, not cross-hardware parity checks.
The actual RTX GDN backend was Triton/FLA despite the recorded FlashInfer request;
historical comparators ran on H100s, so these are not backend-matched causal
estimates of post-training effects. Prompt identity is exact, but numerical
hardware/backend differences remain a caveat.

- 4B job 31242407 on roodborst1: 23m15s total; engine initialization 279.88s;
  scoring phase including canary 1034.16s.
- 9B job 31242269 on roodborst2: 38m11s total; engine initialization 377.55s;
  scoring phase including canary 1643.76s.
- Initial 4B job 31242268 failed before scoring on AlmaLinux 8 due to glibc.
  Replacement required AlmaLinux 9 and retained the scoring contract.
- Failed auxiliary step 31242269.0 was an overlapping diagnostic srun whose
  TaskProlog failed; the actual batch evaluation completed successfully.

Saved pretrained artifacts live in results/pretrained_id/{4b,9b}/. Historical
comparators are results/tool_trajectory_monitoring/gleipnir4b_id_v1/4b/base/base/
and results/tool_trajectory_monitoring/qwen_reasoning_id/direct/.

SHA-256 of pretrained prediction files:

- 4B: `50fa7c49c671406a87e543477969e56c413823027ed5c52d57b4ad32dca02e7c`
- 9B: `c07c8652d7633c0440c4e55d0880b27fa4a9700508f98bbcfa1a1434fa0b0d50`
