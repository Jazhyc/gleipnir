# Sources and scope of the opening slides

Verified 4 October 2026. The methods slide introduces the **regular,
trajectory-only 4B adapter** and contrasts its objective with Sinha et al.'s
method. It does not use the older mixed-data checkpoint or an injection-augmented
checkpoint. The slide shows the mean OOD comparison rounded to three decimals.
The complete OOD comparisons below also document the retained supporting chart
assets, which are not shown in the short opening for the six-minute talk.

## Monitor-directed injection benchmark

Terekhov et al., *Adaptive Attacks on Trusted Monitors Subvert AI Control
Protocols*, [arXiv:2510.09462v2](https://arxiv.org/pdf/2510.09462v2),
2 March 2026. Sections 3.4 and 4.1 and Figure 2 support the benchmark-motivation
slide. In its GPT-5/APPS condition, GPT-4.1 assigns score 0 to all 375 successful
backdoors carrying injections. GPT-4 and GPT-4o also degrade under attack.
This is their reported result, not a measurement of our adapters.

The gray code card quotes short fragments from the GPT-4-targeted payload in
Appendix F.1: an XML authority attribute and the first output-directive sentence.
Ellipses mark omissions; line breaks are adjusted for the slide. The solution-code
placeholder and explanatory caption are authored for the illustration. The card
is a schematic, not a complete code submission or the full attack. Its fragments
also occur in the evaluated `transfer_gpt4.json` template. The adjacent 375/375
statistic is a separate GPT-4.1 result from Figure 2; it is not attributed to this
specific GPT-4 excerpt. The illustration is rendered in LaTeX, with no external
image, dataset row or copied paper figure.

Our evaluation uses our adapted benchmark, published as
[mjkenney/monitor-injection-evals](https://huggingface.co/datasets/mjkenney/monitor-injection-evals),
with code at [mkenney2/monitor-injection-evals](https://github.com/mkenney2/monitor-injection-evals).
The evaluated snapshot is pinned at revision
`58e722bf9912903d56a334f873e069aa17ccb7ad`. The benchmark contract and added
honest-code controls are recorded in
`experiments/monitoring_injection_disentanglement/README.md`. The new slide
motivates the test; the following slide presents initial detection results,
without claiming a full control-protocol evaluation. No benchmark rows or copied paper figures are included in the deck.

The local benchmark checkout at commit
`b55c2014d4c22ceea810b2639e6c0f672d83eccb` documents the adaptation: o3-mini
Control Tax backdoors and same-model honest solutions, transferred payloads from
Terekhov et al., and matched neutral text. The honest-code supplement applies
the same transferred injections to benign programs. It does not reproduce the
paper's GPT-5-generated code population. The diagram shows one of those reused
payload excerpts; the blue box reports the original paper's results, while the
closing sentence motivates testing our trained monitors against those injections.

The gap is specifically a dedicated evaluation of attacks on the monitor.
It is not a claim that the ID/OOD populations contain no injections:
[AgentDojo](https://arxiv.org/abs/2406.13352), for example, targets the task agent
through untrusted tool data. Whether our trained monitors resist transferred
monitor-directed payloads is an experimental question, not a claim that all
trained monitors are unstudied.

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
and 4,344 harmful. Source counts are STRIDE 1,350; Gloom 772;
CoT-Red-Handed 2,340; BashArena 2,790; BashBench 1,436.

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

## Initial injected-backdoor results

The third content slide uses the original regular-trained 4B and 9B adapters
with the regular (neutral) monitor instruction. AUROC compares 647 injected
backdoors per template against 655 clean honest submissions. Only the harmful
class carries injections in this initial comparison. Results average the three
GPT-4/GPT-4.1/GPT-4o transfer templates equally: 4B AUROC
0.9987973461385687; 9B AUROC 0.9984799682228803. Each template has 100%
TPR for both adapters. These results do not measure injected-honest specificity.

TPR at 2% FPR uses an adapter-specific threshold from clean honest code, with
strict score > threshold. The discrete clean alarm rate is 13/655 (1.985%),
so the slide describes the operating point as approximately 2%. Thresholds are
0.16451645470385745 (4B) and 0.13296424535145931 (9B).

`assets/initial_injection_results.json` retains the exact condition metrics and
SHA-256 identities of the original `gleipnir-grid-v1/{4b,9b}/regular/
eval_neutral.metrics.json` summaries. These are historical results, not fresh
evaluations. No benchmark examples are included. The final question motivates
the subsequent honest-injection control; it does not establish a mechanism.

## Harmless-code injection control

The fourth content slide retains the same regular adapters, regular monitoring
prompt and clean-code thresholds as the initial-results slide. It appends each
of the three transfer injections to all 655 honest solutions without adding a
code backdoor. The target remains code-backdoor presence: hostile comments alone
do not turn an honest solution into a positive under this label contract.

Clean honest flag rates are 13/655 (1.9847328244%) for both adapters. At 4B,
all 655 honest solutions alarm under every template (100%). At 9B, counts are
655/655, 655/655 and 652/655 for GPT-4, GPT-4.1 and GPT-4o respectively. The
equal-template mean is 99.8473282443%, displayed as 99.85%. Thresholds are
unchanged; they are not recalibrated against injected honest code.

`assets/harmless_injection_results.json` contains aggregate rates, counts and
source summary SHA-256 identities from historical `gleipnir-honest-controls-v1/
{4b,9b}/regular/eval_neutral.metrics.json`. These controls support strong
injection-text sensitivity under this benchmark, without identifying the learned
mechanism or establishing a result for every possible injection.

## Original adapters on JudgeDeceiver

The fifth content slide uses the unchanged regular monitoring adapters under
the original auxiliary A/B quality-judging prompt. It reports preference
accuracy, not harmfulness accuracy or a false-positive rate. Correct preferences
stay fixed across clean, preferred-answer-injected and disfavored-answer-injected
views. Both answer orders are present; normalized p(B) >= 0.5 selects B.

The frozen holdout has 4,188 variants: 504 clean, 1,842 preferred-injected and
1,842 disfavored-injected, from 252 pairs and six original query groups. Source:
`ShiJiawenwen/JudgeDeceiver@7cfc96af0272befd83220b1d7d3b0699d072ecbf`.
Construction preference labels and upstream suffixes optimized for other judge
interfaces are documented in `experiments/judge_injection_continuation/README.md`.
This is our transformed auxiliary evaluation, not the original published protocol.

Regular 4B accuracies are 99.40%, 58.36%, 95.98%; regular 9B accuracies are
99.21%, 64.33%, 95.28%, respectively. Preferred-injection correct-to-wrong
flip rates are 41.21% / 35.67%. Their asymmetry supports injection-driven
avoidance in quality judging, without identifying a mechanism or proving
sensitivity to every string resembling an injection. These original monitoring
adapters did not train on JudgeDeceiver. The domain and decision surface differ
from their monitoring training. Only six queries limit the scope of OOD claims.

`assets/initial_judge_results.json` retains exact aggregate accuracies, counts
and the SHA-256 of `results/augmented_judge_evaluation/summary.json`. That
summary reuses verified historical regular-4B scores; regular-9B scores were
first collected in the later transfer evaluation. The slide compares original
weights only and does not imply the two sizes were evaluated simultaneously.
No fresh scoring or training was performed for the presentation.

## Deployment risk from injection-driven alarms

The sixth content slide draws a deployment implication from the measured
harmless-code alarms and auxiliary preference avoidance. The initial APPS
comparison placed injections only on backdoors; injected-honest controls
were essential to expose that shortcut. The slide says benchmarks *that* pair
injections only with harmful behavior can reward text suspicion. It does not
assert that every current benchmark has that construction, nor that agent-targeted
injections are absent from existing ID/OOD sources such as AgentDojo.

The deployment chain is a hypothetical threat model, not an observed attack:
an adversary places injection text on a page, an agent encounters but ignores
it, and a monitor nevertheless alarms. If alarms block work pending human
review, repeated false alarms could exhaust review capacity and stall autonomous
workflows. This requires exposure to adversary-controlled content and a
review-gating policy; our experiments did not measure reviewer overload or
end-to-end denial of service. Exposure to an injection alone does not establish
agent compliance or harmful task behavior.

## Diagnosis 1: aggressive training-data filtering

The seventh content slide summarizes the metadata audit and the completed
`experiments/monitoring_injection_removal/README.md` intervention. Exact
trajectory matching to STRIDE metadata identified four assigned
`prompt_inject_another_ai_to_exfiltrate_data_for_blackmail` objectives, all
source-label 1, among 8,688 selected training rows (0.04604%). This is an
exhaustive count of that explicit objective, not a semantic injection census
across all sources. See `docs/findings/training_trace_prompt_injections.md`.

The frozen unadapted Qwen3.5-9B census scanned all 8,688 trajectories. At score
>=0.5, including ties, the user-selected filter excluded 1,154 rows: 1,140
harmful-labeled and 14 benign-labeled. A fresh 4B adapter used the original
monitor instruction, retained original teacher targets and one epoch over
7,534 rows (236 updates). The census has false positives and misses; two of
the four explicit STRIDE cases and one reviewed Gloom exposure remained.
The resulting corpus is not verified injection-free.

Under the original monitoring prompt, injected-harmless APPS flag rate is
100% for both original and filtered adapters under every transfer template.
Each adapter uses its own clean-honest ~2% FPR threshold, not a shared numeric
threshold: 0.164516 original and 0.085099 filtered, each with 13/655 clean
alarms. The same failure persisted under the other two tested instructions.
Mean scores changed and ID macro pAUROC@20 fell from 0.846273 to 0.822255;
“did not resolve” refers specifically to injected-harmless thresholded alarms.

The intervention changes class/source composition and training exposure, and
uses an imperfect classifier. It neither isolates injection content causally
nor excludes other data shortcuts. Exact completed results and artifact identities
are recorded in `docs/findings/monitoring_injection_removal.md`; the collected
evaluation summary SHA-256 is
`d3146714ef5c581ad1a5aaee21488a77dda1dbec15d3fe7c06098b8903dd96db`.
No benchmark or training examples are redistributed in the slide.

## Diagnosis 2: collaborator-reported teacher scoring

The eighth content slide summarizes Michael Kenney’s teacher diagnostic supplied
by the user, recorded in `docs/findings/kimi_teacher_injection_controls.md`.
Numbers are collaborator-reported and rounded, not freshly evaluated or locally
raw-data audited. The stated upstream branch is `kimi-k3-teacher-controls`,
commit `0ad1778`, with 7,173 selected APPS/control rows. Teacher settings use
Makora only; training cached multiple providers. The three column names denote
attack templates, not the evaluated teacher model.

“Verdict 1” refers to the generated hard decision, not the ~2% FPR threshold
used on the preceding student slides. The teacher’s broader rubric can count
monitor manipulation as misconduct despite a negative code-backdoor label.
Matched AUROC compares injected backdoors and injected honest code. Original
4B’s historical equal-template mean is 0.643951; no student rescoring occurred.
The better teacher ranking and different template responses challenge a simple
copying explanation, without proving how the student learned the shortcut.
This evaluates new OOD inputs, not causal teacher-target changes on training
rows, and is a diagnostic rather than an intervention. Full setup, threshold
caveats, source populations and audit status are retained in the finding.

## Diagnosis 3: behavior-grounded instruction framing

The ninth content slide summarizes the completed instruction intervention in
`experiments/monitoring_injection_disentanglement/README.md` and
`docs/findings/monitoring_injection_disentanglement.md`. A fresh 4B adapter
trained for one epoch on the same 8,688 trajectories and fixed Kimi K3 soft
targets, using the conservative behavior-grounded student instruction. The
quoted sentence is from that project-authored instruction. Teacher targets
were not regenerated with a changed rubric.

The displayed comparison is original adapter / original instruction versus
new adapter / conservative instruction. Both have 100% injected-honest APPS
alarms under each of the three transfer templates at their respective
clean-honest ~2% FPR operating points. A three-instruction inference grid
(neutral, aggressive, conservative) also yields 100% for the new adapter in
every cell. Thus the conclusion concerns thresholded false alarms, not an
absence of any score or ranking change. The experiment does not test all
possible framings or a revised teacher-labeling pipeline.

User-reported inspiration: Betley et al. (including Owain Evans),
[Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://proceedings.mlr.press/v267/betley25a.html),
ICML 2025. Their educational-context control changes the framing of insecure
code examples and prevents broad misalignment in their reported evaluation.
This motivates testing context-sensitive generalization; our monitor-instruction
change is a distinct intervention and does not reproduce their experiment.
