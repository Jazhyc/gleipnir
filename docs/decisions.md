# Decision log

## 2026-09-04 — Advance `2e-5` for monitoring-only LR confirmation

- The five-cell seed-0 Qwen3.5-4B screen selected AdamW `2e-5` over the freshly
  retrained `5e-5` control on the frozen 3,012-row ID development suite. Macro
  pAUROC@20 improved from `0.847144` to `0.871066`, macro AUROC improved from
  `0.951173` to `0.957643`, and macro Brier fell from `0.095650` to `0.079273`.
- The selected rate gained `0.051967` pAUROC@20 on Gloom and lost `0.004124` on
  STRIDE, satisfying the predeclared per-source guardrail. AdamW `1e-4` was also
  eligible but ranked behind `2e-5`; `1e-5` narrowly failed the STRIDE guardrail
  and `2e-4` was rejected.
- Advance `2e-5` to replicated confirmation against `5e-5`. Treat it only as a
  monitoring-only candidate until the seed comparison is complete; the strict
  OOD suite remains unconsulted and must not be used to tune this choice.

## 2026-09-03 — Enable pinned causal-conv1d for long Qwen3.5-4B training

- On the Stack 24 two-H100 `gleipnir-improvement` target, source-build pinned
  `causal-conv1d==1.6.2.post1` into an isolated target and require a BF16 GPU
  forward/backward preflight before the mixed-4B model is imported.
- Keep pinned FLA 0.5.2 and causal-conv1d as independent isolated installs. Set
  `CUDA_HOME=/usr/local/cuda` for both so TileLang uses one coherent CUDA 13.0
  compiler/header toolchain rather than mixing packaged CUDA 13.2 and 13.3
  components.
- Adopt the fast path because the matched 8,192-token benchmark improved the
  Qwen-width depthwise convolution by 2.17x and its full gated-delta sublayer by
  1.60x. Do not extrapolate the sublayer ratio to whole-model training; preserve
  the established batch recipe until a matched longest-sequence model preflight
  reports end-to-end throughput and memory.
- Record availability, required status, pinned version, and concrete bound
  causal-convolution module in each training artifact. Fail rather than silently
  returning to Transformers' Torch fallback when the launcher requested the
  extension.

## 2026-08-30 — Run tool-trajectory scaling as Kimi-soft-only distillation

- Train only Kimi K3 soft-logit students for the first Qwen3.5-9B
  tool-trajectory scaling campaign. Do not train matched hard-label adapters;
  retain source labels solely for provenance, stratification, and evaluation.
- Use one seed and rank-128 QLoRA at the six paper row counts, but train every
  point for exactly one epoch rather than copying the paper's optimizer-step
  horizons. This targets the effect of adding unique data without repeating the
  smallest subsets for 63 or 25 epochs. The paper rationale curve remains an
  external descriptive reference, not a supervision-channel causal comparison.
- Add one adapter over the full 8,688-row tool-trajectory pool plus all 13,149
  prior deception soft-distillation rows, also for one epoch. Treat the result
  as fixed-epoch data scaling rather than a fixed-compute mixture ablation.
- The prior single-seed deception optimization screen favored two epochs over
  one by 0.00470 macro AUROC but worsened Brier by 0.00144. This indirect,
  shorter-context evidence does not justify doubling the trajectory campaign;
  prioritize the controlled data-volume curve and timely completion.
- Preserve effective batch 32 while changing the long-context realization to
  microbatch 1 and accumulation 32, subject to a rank-128 longest-sequence H100
  preflight and a zero-truncation 29,696-token preparation audit.

## 2026-08-30 — Make paid teacher caching provider-sticky and auditable

- Route synchronous OpenRouter annotation campaigns through a stable
  `session_id` derived from the model and exact prompt prefix, mark that prefix
  with an explicit ephemeral cache breakpoint, and complete one warm-up request
  before concurrent fan-out. This preserves the concatenated prompt text while
  giving provider caches a warm shared prefix.
- For Kimi K3 (`moonshotai/kimi-k3`), prefer Makora for binary-logprob caching;
  it currently has lower input, output, and cache-read rates than Fireworks.
  Pin Makora when endpoint consistency is required, or explicitly order Makora
  before Fireworks when availability is more important.
- Store an exact request-settings fingerprint, actual route metadata, timestamps,
  and provider cache token telemetry in every row. Reject resume when either the
  prompt or request settings drift, and require a canary with nonzero cache reads
  before scaling paid annotation.
- A live synthetic Makora canary verified logprob delivery and reused 8,448 of
  9,627 prompt tokens on its second request. Reported request cost fell from
  $0.024404 to $0.005218; no repository or dataset content was sent.

## 2026-08-17 — Default frozen text evaluation to vLLM

- Use a persistent vLLM engine with continuous batching for large frozen
  text-only base/LoRA evaluations. Request only the literal decision-token
  logprobs and constrain generation to one token; do not inherit the training
  microbatch for inference.
- Retain Transformers eager inference as a bounded parity/debug path and for
  adapters vLLM cannot represent. Before a new serving campaign, compare the
  causal master with the checksum-rebased serving adapter on identical rows,
  tolerate recorded numerical differences, and require a meaningful adapter
  effect under both backends.
- The first AISI transfer launch used fixed eager batches and was superseded
  before interpretation by the parity-gated vLLM campaign. Keep any partial
  eager artifacts diagnostic-only and summarize only the vLLM result tree.

## 2026-08-17 — Do not promote hard-only from the internal split

- The unadapted Qwen3.5-9B base reaches 0.92273 macro AUROC on the exact 1,972-row
  internal development surface (0.88903 pooled AUROC). Hard-only adaptation
  reaches a three-seed mean of 0.97256 macro AUROC, but its family gains are
  highly uneven: +0.00644 on instructed Aletheia, +0.05385 on varied Aletheia,
  and +0.17097 on Liars Bench.
- Interpret the hard-only result as strong within-source adaptation, not evidence
  of deployment-distribution transfer. The development split is stratified by
  dataset and label but cannot group unavailable conversation/generator lineage,
  and the same dataset families occur in train and development.
- Retain the mixed soft-plus-hard weight-0.50 recipe as the internal AUROC
  candidate because it is stronger on the varied family (0.95078 versus 0.93589
  for hard-only). Freeze recipes now; compare the base, hard-only, and mixed
  candidate once on an untouched source-grouped surface before promotion.

## 2026-08-14 — Screen structural training interventions

- After conventional optimizer and regularization changes failed to improve
  macro AUROC, test twenty rank-128 cells on the fixed internal development
  protocol: three baseline seeds and seventeen seed-0 interventions spanning base
  precision, adapter initialization/parameterization/targets, loss geometry,
  adaptive group weighting, and effective batch size.
- Train on all 11,177 rows outside the frozen 1,972-row development split. Keep
  competition validation and the final test untouched. Require a positive
  seed-matched AUROC delta and the existing 0.002 balanced-accuracy/Brier gates,
  then seeds 1 and 2, before promotion.
- Evaluate retained half-epoch checkpoints and predeclared checkpoint/seed logit
  ensembles. Cross-fit threshold and Platt diagnostics within development;
  treat them as operating-point/calibration evidence rather than ranking gains.
- Use causal Transformers evaluation across the screen because pinned vLLM does
  not support DoRA. Treat LoftQ as non-promotable diagnostic evidence on its NF4
  inference base, and require a separate serving-parity canary for any selected
  nonstandard adapter.
- Treat the two explicit decision-head cells as readout/gradient-estimator
  ablations: their forward scores and head gradients use the auxiliary head,
  while the QLoRA backbone receives an explicit straight-through gradient via
  the literal `0|1` LM-head rows. FLA 0.5.2 cannot lower direct auxiliary-head
  gradients into Qwen3.5's gated-delta blocks; record the surrogate in metadata
  and evaluate only the auxiliary head.

## 2026-08-14 — Retain the baseline after the regularization screen

- All sixteen rank-128 screen cells completed on the fixed internal development
  split. The AdamW baseline remained best in macro AUROC at 0.96449; every
  warmup, schedule, horizon, dropout, weight-decay, sampling, and target-scale
  intervention reduced the primary metric.
- The frozen constraints and AUROC ranking retain square-root-balanced sampling,
  teacher-logit scale 1.5, weight decay 0.10, cosine scheduling, and
  teacher-logit scale 2.0 for possible seed-1/2 replication. These cells are
  trade-offs rather than wins. Their AUROC regressions range from 0.00025 to
  0.00198, although all improve balanced accuracy and four improve Brier.
- Do not combine or promote interventions from this single-seed screen. Retain
  the existing AdamW `5e-5`, linear, 3%-warmup, two-epoch, zero-dropout and
  zero-weight-decay, proportional-sampling, unscaled-target recipe unless the
  frozen shortlist wins replicated evaluation. Competition validation and the
  final test remained untouched.
- Teacher-logit scale 1.5 is the most interesting operating-point trade-off:
  relative to baseline it changes macro AUROC by -0.00091, balanced accuracy by
  +0.01000, and Brier by -0.00142. Scale 2.0 is dominated by scale 1.5 on all
  three macro metrics and creates substantially more tied scores, so it should
  not be preferred merely because it passed the formal shortlist rule.

## 2026-08-14 — Screen trajectory, regularization, and data weighting

- After Muon failed full-data confirmation, keep AdamW and screen sixteen
  rank-128 cells: one shared baseline plus fifteen single-factor changes to
  warmup, schedule, epoch count, LoRA dropout, weight decay, teacher-logit
  smoothing, and dataset sampling.
- Reserve a new fixed 15% internal development split, train every cell on the
  same 50% selection from its complement, and keep competition validation and
  final test untouched. Treat the single-seed screen as pruning evidence only;
  replicate at most five eligible cells before combining interventions.
- Add deterministic square-root-balanced and uniform-dataset sampling. Record
  expected source exposure because source sizes differ by about 15x while the
  primary metric macro-averages datasets. Retain causal model-only checkpoints
  about every half epoch and record loss/LR history so horizon effects remain
  inspectable without optimizer-state storage.

## 2026-08-13 — Retain AdamW after full-data Muon confirmation

- Neither predeclared Muon candidate passed the frozen three-seed confirmation
  gate. Muon `5e-5` reduced paired mean macro AUROC by 0.00103 and improved only
  one seed. Muon `1e-4` increased paired mean macro AUROC by 0.00097 with two
  seed wins, but regressed macro balanced accuracy by 0.00238 and macro Brier by
  0.00313, beyond the 0.002 tolerances.
- Retain rank-64 AdamW at `5e-5` as the selected optimizer recipe. Do not promote
  Muon based on its higher raw AUROC at `1e-4`, and do not tune additional
  optimizer rates on the confirmatory validation surface.
- Interpret the small effects cautiously. With three paired seeds, approximate
  95% t intervals for the AUROC differences include zero widely: [-0.00824,
  +0.00618] at `5e-5` and [-0.00597, +0.00791] at `1e-4`. The earlier 25%-data
  internal-screen advantage did not replicate robustly at full data.

## 2026-08-13 — Confirm the two Muon candidates on full data

- Replace the planned 50%-data confirmation with a 100%-training-data
  confirmation at the user's direction. Train rank-64 Muon at the two rates
  advanced by the internal screen (`5e-5` and `1e-4`) for seeds 0, 1, and 2.
- Reuse the three recipe-identical full-data AdamW `5e-5` rank-64 cells rather
  than retraining them. Their source commit, metrics checksum, seeds, row count,
  and training recipe must pass the preparation contract.
- Since full-data training consumes the screen's internal-development pool, use
  the frozen 822-row competition validation artifact as a confirmatory surface.
  Freeze the two rates and the promotion gate before launch, prohibit further
  optimizer-rate tuning on these results, and leave the final test untouched.

## 2026-08-13 — Advance two Muon rates to replicated confirmation

- The rank-64, 25%-data internal-development screen completed all ten paired
  cells. Muon `5e-5` won the predeclared primary metric with 0.94727 macro
  AUROC, +0.00312 over AdamW at the same rate. Muon `1e-4` was the strongest
  multi-metric alternative, with 0.94608 macro AUROC, 0.88942 macro balanced
  accuracy, and 0.08900 macro Brier.
- Retain both Muon `5e-5` and `1e-4` for replicated 50%-data confirmation. Keep
  AdamW `5e-5` as the control. Do not promote Muon or evaluate a selected recipe
  on competition validation from this single-seed, non-lineage-grouped screen.
- Treat the reversal across rates as evidence that RMS matching does not remove
  optimizer-specific learning-rate sensitivity: Muon was worse at `1e-5` and
  `2e-5`, and better in macro AUROC from `5e-5` through `2e-4`.

## 2026-08-13 — Compare Muon through scheduled update-RMS matching

- Replace the independent, unscheduled Muon learning rate with the standard
  scheduler-managed parameter-group `lr`. Apply the per-matrix Moonshot scaling
  `0.2 * sqrt(max(A, B))`, so AdamW and Muon can be compared at the same five
  base learning rates despite LoRA A/B shape differences.
- Screen paired AdamW and Muon rates on rank 64, 25% training data, and a fixed
  internal dataset-label-stratified development split. Do not use competition
  validation for hyperparameter selection. The source artifact has no available
  conversation/generator lineage key, so record that the internal split cannot
  enforce lineage grouping.
- Keep the linear schedule, 3% warmup, two epochs, QLoRA/FLA path, batch, seed,
  data, and objective fixed. Treat the ten-cell screen as pruning evidence and
  require replicated confirmation before promotion.

## 2026-08-12 — Measure data and adapter capacity jointly

- Cross nested `10%`, `25%`, and `50%` training selections with QLoRA ranks 4,
  16, 64, and 256 using paired seeds 0, 1, and 2. Reuse the matching 100%-data
  cells from the completed capacity sweep, yielding 36 new training jobs and 12
  checksum-preserved references.
- Hold the proven NF4/double-quantized/BF16, FLA 0.5.2, microbatch-8 recipe fixed
  so the only interventions are data volume and adapter rank. Keep two epochs,
  meaning optimizer updates scale with data volume rather than being held fixed.
- Treat macro AUROC and within-seed rank contrasts as primary evidence. Estimate
  whether the paired rank-256 advantage over rank 16 grows with log training
  rows. Do not select a deployment rank on this validation surface.

## 2026-08-11 — Require FLA and standard QLoRA for the rank curve

- The fast Qwen3.5 H100 path requires more than `AutoModelForCausalLM`: install
  isolated `flash-linear-attention==0.5.2` and `fla-core==0.5.2` packages before
  model import, verify Transformers selected their gated-delta kernels, and fail
  closed if the probe fails. Use the measured eager recipe of microbatch 8,
  accumulation 4, effective batch 32, gradient checkpointing, and no
  `torch.compile`.
- Train every adapter-rank condition with the standard QLoRA base configuration:
  4-bit NF4, double quantization, and BF16 compute. Hold quantization and batch
  fixed across ranks. Preserve the trained FP32 adapters and evaluate their
  checksum-rebased copies on the ordinary BF16 serving base.
- Use selected-position vocabulary projection for every QLoRA rank. A rank-256,
  batch-8 preflight consumed about 69 GB before full-sequence logits triggered a
  further roughly 25 GB allocation and OOMed. The earlier selected-position
  batch-2 path was 2.9% slower, but it is necessary to retain the proven batch
  and is held fixed across this new curve.
- Require a rank-256, 20-step memory and throughput preflight before scheduling
  the full curve. This catches FLA fallback and high-rank OOMs without silently
  changing the experimental recipe.
- The corrected single-GPU preflight measured 3.551 samples/s at rank 16 and
  3.524 at rank 256, with peak allocated memory of 26.19 and 33.25 GB. Treat the
  earlier 8.811 samples/s measurement as specific to its shorter 2,880-row
  prompt distribution; the full mixed-data runs project to about 2.1 hours.
- Retain seeds 0, 1, and 2 at every rank. In the prior full-data condition,
  macro AUROC had SD 0.00182 and range 0.00357, while the preceding 64%-to-100%
  mean gain was only 0.00256. Seed noise is therefore large enough to imitate
  the capacity effects this curve is designed to measure.
- Cancel and preserve the causal BF16/microbatch-2 attempt after its first two
  rank-256 completions. It used the slow Torch gated-delta fallback and is not
  combinable with the QLoRA curve.

## 2026-08-11 — Train text-only LoRAs causally and export serving copies

- For experiments containing no image or video inputs, train Qwen3.5 LoRAs
  through its causal text compatibility model and preserve that FP32 adapter as
  the master artifact.
- Export a checksum-tracked serving copy by deterministically rebasing causal
  PEFT keys into the multimodal wrapper's `language_model` namespace. Require a
  serving smoke test before promotion. This policy does not apply to multimodal
  training data or full-model fine-tuning.
- Retain full-sequence logits for the restarted curve. A matched 20-step H100
  preflight found that `logits_to_keep` was 2.9% slower (309.3 versus 300.7
  seconds) and produced a slightly different accumulated loss. Keep the
  selected-position mode only as an explicit benchmark condition.
- The native-multimodal adapter-capacity sweep was cancelled before evaluation
  and superseded by a fresh causal-master sweep; do not combine its checkpoints
  with the restarted scaling curve.

## 2026-08-10 — Match multimodal training and serving architectures

- Load multimodal backbones such as Qwen3.5 through
  `AutoModelForImageTextToText` during training, even for text-only batches, and
  restrict LoRA targets explicitly to the language model.
- Require an inference parity canary that demonstrates a meaningful adapter
  effect before evaluating a new model/adapter/backend combination at scale.
- Treat the first Kimi data-scaling validation pass as invalid: its adapters were
  trained against Transformers' causal compatibility wrapper, while vLLM served
  the multimodal conditional-generation wrapper. Rebase those adapter keys with
  source and destination checksums rather than modifying the master adapters.

## 2026-08-09 — Bootstrap

- Start Gleipnir as a standalone repository rather than retaining competition
  submission and leaderboard machinery.
- Use Qwen 3.5 as the initial backbone family.
- Seed the project with deception distillation, generic OpenRouter binary-logprob
  collection, Slurm tooling, and Lambda Cloud campaign management.
- Pin vLLM 0.24.0 and Transformers 5.14.1, the latest stable PyPI releases checked
  at bootstrap; exclude NNsight and NDIF.
- Keep experiments isolated and record durable findings under `docs/`.
