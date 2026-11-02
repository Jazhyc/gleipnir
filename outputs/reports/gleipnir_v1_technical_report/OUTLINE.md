# Outline

Claims per section, accepted by their owner before the section is drafted.
Each claim follows [STYLE.md](STYLE.md): it names the model, evaluation set, metric and value, and
points to its evidence. The numbers below are copied from the linked records
as a starting point. Check each one against its source, and record it in
[sources.md](sources.md), when the section is drafted.

Status per claim: **draft** (proposed), **accepted** (the claim's owner
accepts it), **written** (in the text and in `sources.md`).

## Ownership

The owner of a claim decides its wording and whether it stays. Jeremias
Ferrao (JF) owns every section and claim except the held-out benchmarks other
than SleightBench, which Abhinav Rajeev Kumar (ARK) owns: natural cheats,
misleading comments and APPS injections (C5.1–C5.3). Each section lists its
owner; a claim with a different owner says so.

Paths are relative to the repository root.

## Main text and appendix

The main text has at most nine pages, counted up to the end of the conclusion
(`make check` reports the count). Material that does not support a main-text
claim goes to the appendix. Each main section points to its appendix section.

## Decisions

Made by Jeremias Ferrao on 2026-11-02.

1. **Headline scores.** Report each model on the optimized FP8/FP4 serving
   stack with its best prompt, as in the frontier figure: Gleipnir 4B with the
   student prompt (0.770 Mean-OOD pAUROC@20) and Gleipnir 27B with the teacher
   prompt (0.951). The optimized stack is the deployment path, and the OOD
   suite measures what it costs in quality. BF16 scores go in the release
   section and the appendix.
2. **Teacher prompt for the 27B.** Present it as an ablation in the appendix.
   It tested whether the 27B benefits from more detailed instructions. It was
   adopted because it also improves ID ranking.
3. **Probing.** The main result is in the main text. Probe families,
   hyperparameters and layer choices go in the appendix.
4. **Cost.** Report both bases: self-hosted B200 break-even at on-demand and at
   committed-use prices, as in the frontier figure.
5. **Release version.** The report releases Gleipnir 4B and 27B as v1.0. The
   weights are the same as 4B v0.4 and 27B v0.2; only the model cards change.
   The report names the models v1.0, and `sources.md` records that results
   measured on v0.4/v0.2 (and on the v0.3 weights) apply to v1.0 unchanged.

## 1. Introduction

Owner: JF.

- **C1.1 (draft).** A monitor that scores each agent action runs on the
  critical path of that action, so its latency and cost add to every action.
- **C1.2 (draft).** Summary of the results: the main claims of Sections 4–6 in
  one paragraph each.
- **C1.3 (draft).** Contributions: the training recipe, the evaluation, the
  adversarial and probing analysis, and the released models and serving stack.

Exhibit: optionally, the frontier figure (Figure 1) on the first page.

## 2. Related work

Owner: JF.

- **C2.1 (draft).** Sinha et al. train monitors to generate a rationale and
  then a 0–10 score (SFT, optionally RL). We train the monitor to emit one
  decision token whose probability is the score.
- **C2.2 (draft).** Position against prompted frontier monitors, activation
  probes and monitor-directed attacks (Terekhov et al.).

## 3. Method

Owner: JF.

- **C3.1 (draft).** The monitor reads a visible agent trajectory and emits one
  token, `0` or `1`. The score is the normalized probability of `1`.
- **C3.2 (draft).** The interface matters on its own. With Qwen3.5-9B, the
  prompt and the OOD suite fixed, binary logits score 0.698 Mean-OOD
  pAUROC@20, an ordinal 0–10 score 0.486, and an ordinal score after visible
  reasoning 0.666 at higher cost.
  Evidence: `docs/writeups/gleipnir_project_writeup.md` (interface ablation),
  `docs/research/tool_trajectory_ood_frontier.md`.
- **C3.3 (draft).** We train LoRA adapters to match the teacher's (Kimi K3)
  probability on the decision token, after a strict teacher-agreement filter.
  Evidence: `docs/decisions/strict_teacher_filtering.md`,
  `experiments/tool_trajectory_monitoring/README.md`.
- **C3.4 (draft).** Soft targets matter. Matched hard-label 4B seeds score
  0.690–0.746 Mean-OOD pAUROC@20 against 0.777–0.813 for soft-target seeds.
  Evidence: `experiments/hard_label_seeds/`.
- **C3.5 (draft).** The released 4B is the weight average of three seeds.
  Evidence: `experiments/seed_average/README.md`.
- **C3.6 (draft, appendix).** Teacher-prompt ablation for the 27B. Trained with
  the compact student prompt, Gleipnir 27B scores 0.942 ID macro pAUROC@20
  (CoT-removed suite) when scored with the full teacher prompt, against 0.922
  with the student prompt. Mean-OOD pAUROC@20 (BF16) is 0.935 against 0.909.
  The teacher prompt has 9.9% more prompt tokens.
  Evidence: `experiments/gleipnir27b_vocab_distillation/README.md`,
  `docs/findings/qwen38_27b_campaign.md`.

Exhibits: pipeline diagram (`figures/distillation_pipeline_comparison.svg`),
interface ablation table. Appendix: data sources, filter, hyperparameters,
prompts, training cost.

## 4. Main evaluation

Owner: JF.

- **C4.1 (draft).** Metric: source-macro raw normalized pAUROC@20 on the
  six-source, 6,395-row OOD suite of Sinha et al. Cost: uncached list-price USD
  per 1,000 evaluations. For Gleipnir, cost is the self-hosted B200 break-even
  at on-demand and at committed-use GPU prices; the text explains both.
  Evidence: `docs/research/tool_trajectory_ood_frontier.md` (comparison contract).
- **C4.2 (draft).** Gleipnir 27B (optimized serving, teacher prompt) scores
  0.951 Mean-OOD pAUROC@20 (95% interval 0.941–0.960) at $0.17 (committed-use)
  to $0.39 (on-demand) per 1,000 evaluations. Claude Opus 4.6, the best
  reasoning monitor of Sinha et al., scores 0.942 at $94.79. The difference is
  within the interval.
  Evidence: `docs/research/tool_trajectory_ood_frontier.md`,
  `docs/findings/qwen38_27b_campaign.md`, `experiments/optimized_gap/README.md`.
- **C4.3 (draft).** Gleipnir 4B (optimized serving, student prompt) scores
  0.770 at $0.04 (committed-use) to $0.09 (on-demand) per 1,000 evaluations.
  The trained Qwen3.5-4B monitor of Sinha et al. (SFT+RL) scores 0.684 at
  $0.36.
- **C4.4 (draft).** Training improves on the unadapted backbone: Qwen3.8-27B
  scores 0.890 unadapted, and Qwen3.5-4B scores 0.618.
- **C4.5 (draft).** Gleipnir 27B scores higher than its teacher, Kimi K3
  (0.908), on the OOD suite.

Exhibits: frontier figure (`figures/tool_trajectory_ood_frontier.svg`), main
table with bootstrap intervals. Appendix: per-source results, full frontier
figure, decision-model table, ID results, calibration.

## 5. Held-out and adversarial evaluation

Owner: ARK (C5.1–C5.3) and JF (C5.4, section framing).

- **C5.1 (draft, ARK).** On natural cheats (ImpossibleBench LiveCodeBench, 60
  hand-checked cheats), Gleipnir 4B and 27B score 0.935 and 0.941 pAUROC@20,
  level with Kimi K3, Claude Sonnet 5.5 and GPT-6.1 Sol at 3–13 times lower
  cost.
  Evidence: `docs/findings/adversarial_frontier_comparison.md`.
- **C5.2 (draft, ARK).** Misleading text in the trajectory changes the 4B's
  scores. With the same accusation text on both programs, the 4B separates backdoored
  from honest APPS code at AUROC 0.58–0.59, and the 27B at 0.96–0.98.
  Evidence: `experiments/adversarial_crossed_text/`.
- **C5.3 (draft, ARK).** Both monitors treat text addressed to the monitor as a
  sign of harm. At the APPS operating point, both flag every injected honest
  program. The unadapted 27B shows the same behavior, and the adapter reduces
  it.
  Evidence: `docs/findings/apps_released_monitors.md`.
- **C5.4 (draft, JF).** On SleightBench (stripped view), the 27B catches 15% and
  the 4B 5% of attacks at 1% FPR. Claude Opus 4.6 with extended thinking
  catches 32%. Most misses are attacks built to look like honest mistakes,
  which the training specification labels benign.
  Evidence: `docs/findings/sleightbench_released_monitors.md`.

Exhibits: natural-cheats table, crossed-text comparison, SleightBench catch
rate (`figures/sleightbench_catch_rate.svg`). Appendix: APPS injection
details, SleightBench per-attack results.

## 6. Probing

Owner: JF.

- **C6.1 (draft).** Soft targets also help activation probes, but less than
  they help LoRA. On the 4B residual stream (layer 15), the selected soft probe
  scores 0.007 (prompted input) and 0.055 (bare trajectory, NTP) Mean-OOD
  pAUROC@20 above the selected hard probe. On LoRA, soft targets add about
  0.07–0.08. The selected probes come from different probe families.
  Evidence: `experiments/probe_generalization/`, `experiments/hard_label_seeds/`.
- **C6.2 (draft).** At 27B, probes stay below the adapter. The best probe (last
  token, layer 50, teacher prompt) scores 0.896, below Gleipnir 27B (0.935
  BF16, teacher prompt).
  Evidence: `experiments/probe_scale_27b/`.
- **C6.3 (draft).** Probes are a tool for analysis here, not a cheaper
  deployment path. Fine-tuning is better in both quality and cost.

Exhibit: probe objective comparison (`figures/probe_objective_ood_dumbbell.svg`).
Appendix: probe families, layer sweep, attention maps.

## 7. Released models and serving

Owner: JF.

- **C7.1 (draft).** Released artifacts: FP32 LoRA adapters and merged BF16
  checkpoints for Gleipnir 4B v1.0 and 27B v1.0 (MIT; same weights as 4B v0.4
  and 27B v0.2), and a serving image.
  Evidence: `README.md` (Released models), `model_releases/hf/`.
- **C7.2 (draft).** Optimized serving changes scores: −0.034 Mean-OOD for the
  4B and +0.016 for the 27B against BF16. Runs on that stack are not
  bit-for-bit repeatable (per-row MAE about 0.01 for the 4B).
  Evidence: `experiments/release_serving/README.md`,
  `experiments/optimized_gap/README.md`.

## 8. Limitations

Owner: JF.

- **C8.1 (draft).** One teacher, one training domain, one seed for the 27B.
- **C8.2 (draft).** The monitors do not catch mistake-shaped attacks or resist
  monitor-directed text (Section 5).
- **C8.3 (draft).** The OOD suite inherits a 28,000-token input filter that
  removed 43% of SHADE-Arena, and SHADE-Arena has five task pairs.

## 9. Conclusion

Owner: JF.

- **C9.1 (draft).** Restate the main claims; list follow-up work.
