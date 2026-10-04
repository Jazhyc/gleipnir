# Sources and scope of the opening slides

Verified 4 October 2026. The deck compares the **regular, trajectory-only 4B
adapter** with a published baseline. It does not use the older mixed-data
checkpoint or any injection-augmented checkpoint in this opening comparison.

## Published baseline

Sinha et al., *Training Deliberative Monitors for Black-Box Scheming Detection*,
[arXiv:2605.29601v1](https://arxiv.org/abs/2605.29601), 28 May 2026.
[Full text](https://arxiv.org/html/2605.29601v1).

Appendix D describes rationale SFT and GRPO with calibration and ranking rewards.
Tables 4 and 8 provide the Qwen3.5-4B SFT+RL OOD results. Table 8 reports
mean ± standard deviation across three seeds; our comparison uses its published
numbers rather than rescoring their adapter. The table rounds source scores to
three decimals. Its reported aggregate, 0.684, is retained rather than recomputed
from rounded source values. Published LoRA rank/alpha: 32/16; SFT: three epochs,
learning rate 1e-4. These differ from our recipe.

The presentation contains independently drawn charts and aggregate numerical
comparisons, with attribution. No paper figures or upstream dataset rows are
redistributed.

## Training population and supervision

Upstream dataset: `adityaasinha28/deliberative-monitor-pipeline`.
Revision: `2f10f4e49f4bf349965640ac021406689455d1c3`.
File: `merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet`.
SHA-256: `43bfc9f78d6584c43135109636f5e1747c694842419a327c3f37614fb9301802`.
Upstream dataset license was unspecified; no raw examples are included here.

The preparation extracts the outer user-message trajectory from the released
SFT rows and excludes the upstream monitor's assistant completion. It preserves
the embedded agent transcript. The 8,688 rows are balanced globally: 4,344 benign
and 4,344 harmful, with source counts shown in the deck.

Frozen Kimi K3 binary soft-target SHA-256:
`1ae8c3cccc2546335f8002d1475cd86d7a7e059fedb66345d1aa13d6a30a526a`.
Teacher scoring uses the full rubric; the student receives the compact regular
instruction and the same trajectory. The decision probabilities normalize only
the literal 0/1 token pair. Ground-truth labels are separate from these targets.
The loss is soft BCE only, with no hard-label mixture or rationale target.

The regular student uses Qwen3.5-4B at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16 base weights, FP32 master
LoRA adapters (rank 128, alpha 256), seed 0, AdamW 5e-5, linear decay with 3%
warmup, zero weight decay/dropout, logical batch 32 and packed inputs. The final
one-epoch checkpoint follows 272 updates over all rows. There is no OOD-based
checkpoint selection. Serving scoring reads constrained decision-token logprobs
in vLLM; adapter-specific parity was checked before the full evaluation.

Repository evidence at the time the slides were prepared:

- `experiments/student_injection_awareness/README.md` and `config.yaml`:
  frozen campaign and input population.
- `docs/findings/student_injection_awareness.md`: completed regular 4B training,
  artifact verification, serving checks, and OOD results.
- `experiments/tool_trajectory_monitoring/prepare_teacher_training_cache.py`:
  source identity and trajectory extraction.

These references explain provenance; the PDF build and chart regeneration do
not require any of these repository files.

## Input-contract limitation

`docs/findings/training_input_reasoning.md` verifies agent reasoning or explicit
thinking calls in 1,340 of 1,350 STRIDE training trajectories (15.42% of all
training rows). The original audit checked the teacher prompts and monitoring
rows in the mixed student file; the trajectory-only campaign retains those
reasoning-bearing trajectories and teacher targets. This is a confirmed lower
bound, not proof that other sources are free of reasoning. Excluding the outer
monitor rationale does not strip the agent's embedded reasoning.

Held-out preprocessing removes recognized agent-reasoning forms. The independent
`docs/findings/ood_input_reasoning.md` audit found no explicit reasoning markers
or evidence requiring a CoT-removal rerun across the frozen 6,395 OOD inputs.
This does not establish that every unmarked natural-language fragment is free
of reasoning. The deck does not claim strict action-only training or identical
preprocessing to the published baseline.

## Metric, result identity and comparison limits

For each source, pAUROC@20 is the trapezoidal partial ROC area for false-positive
rates from 0 to 0.2, divided by 0.2. It is not scikit-learn's standardized partial
AUC. Mean OOD is an unweighted mean across the six sources; it is neither a
sample-weighted mean nor pooled AUC. This matches the paper's stated metric.

`assets/ood_comparison.json` contains the complete chart inputs: source counts,
exact local metrics, published rounded values, reported SDs, and SHA-256
identities for our evaluation summary, canonical input, regular prompt and
adapter artifacts. The source evaluation summary was
`results/student_injection_awareness/4b/regular/ood/result.json`.

The primary comparison is 0.7762376769297958 versus the published 0.684:
+0.0922376769297958 absolute, or approximately +13.5% relative. Five source
scores improve; AgentDojo declines. This is a substantial aggregate recipe
comparison, not an objective-only ablation or a claim of statistical
significance. Teacher, adapter configuration, schedule, output interface,
checkpoint-selection procedure and preprocessing differ. Our result has one
checkpoint and no multi-seed uncertainty estimate. The OOD suite is a reused
research benchmark, not a fresh unseen confirmation set.
