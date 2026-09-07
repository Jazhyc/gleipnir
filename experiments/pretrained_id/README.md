# Pretrained-only Qwen3.5 ID comparison

Hypothesis: removing post-training may improve direct-boundary monitoring
calibration, possibly at the cost of discrimination or instruction following.
Evaluate both Qwen3.5-4B-Base and 9B-Base, without training or prompt tuning.

Use all 3,012 frozen compact-student ID prompts (Gloom 2,066; STRIDE 946),
the upstream tokenizer's non-thinking chat template and literal Prediction:
boundary. These pretrained checkpoints include the same chat control template;
this measures zero-shot elicitation under the deployed interface, not maximum
achievable pretrained-model capability. Compare historical post-trained 4B
and 9B direct-logit ID results, not the 640-row teacher-prompt population.

Primary metric: unweighted source-macro raw-normalized pAUROC@20. Report
AUROC, Brier, ECE at 5/10/20 bins, log loss, threshold diagnostics and ties.
No checkpoint/prompt/threshold selection; report both arms regardless of score.
Stop on checksum/identity drift, missing/nonfinite logits, truncation, OOM,
or failed balanced-plus-longest batched/singleton numerical canary.

Each separate Slurm job uses one RTX PRO 6000, one CPU, 32GB host RAM, BF16
text-only vLLM, continuous batching, and a four-hour limit.
Require `alma9`: the initial 4B job 31242268 landed on AlmaLinux 8 and failed
before scoring because the locked llguidance wheel requires GLIBC >=2.30.
The replacement keeps the model/prompt/scoring contract unchanged.
Startup observation before any scores: this vLLM build does not support
FlashInfer GDN on RTX PRO 6000 (SM120); its FlashInfer request resolves to
Triton/FLA. Retain the requested setting in the immutable configuration and
record the actual backend from logs as Triton/FLA. The numerical canary must
still pass; historical H100/FlashInfer comparisons are not backend matched.
Completed jobs are 31242407 (4B, roodborst1) and 31242269 (9B, roodborst2).
No adapter is present; adapter export parity does not apply. Numerical canary
requires mean absolute batch/singleton score error <=.02 and max <=.05.
Historical H100 versus current RTX serving hardware remains a comparison caveat.

Run `sbatch cluster/slurm/evaluate_pretrained_id.sh 4b` and likewise `9b`.
Entrypoint `python -m experiments.pretrained_id.run --size 4b` resolves the
shared YAML into a hashed, resumable per-arm JSON contract under results.
Outputs: `results/pretrained_id/<size>/`; logs: `logs/slurm/pretrained_id/`.
Monitor startup in-chat, then every ten minutes if an agent scheduler exists.

Both arms completed and were audited on 2026-09-07. Pretrained-only weights
reduce macro AUROC from .860769 to .751012 (4B) and .898587 to .798188 (9B).
The 4B ECE improvement does not extend to Brier or ranking; 9B worsens on both.
See [the finding](../../docs/findings/pretrained_qwen_id.md) for the matched
comparison, coverage checks, runtime and backend caveat.
