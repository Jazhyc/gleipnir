# Monitoring objective ablation

Status: five-arm, one-seed ID-development screen frozen before training.

The initial two rationale runs are diagnostic only because their sequential
auxiliary/direct accumulation normalization was inconsistent. They are excluded
from promotion and require corrected reruns with the same frozen weights; see
[`the accumulation audit`](../../docs/findings/monitoring_objective_accumulation.md).
The MIL-only runs are unaffected by that bug.

## Question

Can the selected monitoring-only Qwen3.5-4B Kimi-soft recipe gain complementary
evidence sensitivity from the paper's already-stored structured analyses or
from action-level multiple-instance learning (MIL)? This experiment makes no
new teacher calls and keeps the cached Kimi K3 probability as the primary
target in every arm.

## Interventions

Two arms add assistant-only cross-entropy on the paper-author Gemini 2.5 Pro
analysis selected by a Claude Sonnet 4.5 quality judge. The direct Kimi forward
continues to use Gleipnir's unchanged compact `Prediction:` prompt; the
rationale forward uses the original system prompt and trajectory stored in the
three-message source row. The rationale's terminal ordinal answer is known to
have been selected or regenerated to agree with the source label, so it is
privileged explanatory supervision rather than an independent calibrated
target.

Three arms add MIL over at most eight deterministic action/tool-result
endpoints. Each endpoint is an unlabeled instance; a length-normalized pool
produces the bag logit and is trained against the same Kimi soft probability as
the complete trajectory. Max, log-mean-exp, and top-three-mean pooling test
sparse, smooth, and repeated-evidence assumptions. MIL is weak localization,
not contrastive learning.

All arms use the selected `2e-5` learning rate, one epoch, seed 0, rank-128
QLoRA, effective batch 32, full-block checkpointing with selective compilation,
recipe, and pinned FLA/causal-conv1d kernels. Rationale inputs are capped at
24,576 tokens while the primary Kimi input remains at 29,696. This preserves
every rationale target (maximum 1,744 tokens) and full rationale context for
7,796 of 8,688 rows (89.74%). The cap is needed because the original rationale
backward peaked at 77.5 GiB. Fully eager SDPA attempted a 52.2 GiB attention allocation,
and compiling an all-layer-checkpointed model attempted a 45.6 GiB quadratic
attention allocation even after the cap. Compiling a shell around the
full-attention graph break also made flash SDPA ineligible. All arms therefore
checkpoint all 32 decoder blocks but compile only the 24 linear-attention shells
while holding FLA/causal-conv behind graph breaks; the 8 full-attention layers
remain eager. Unpadded batches omit
the redundant all-ones attention mask, and the campaign disables all SDPA
fallbacks. Trainer is explicitly prevented from globally re-enabling
checkpointing after the selective policy is applied. A 2,048-token
BF16-autocast eager/compiled logit canary must pass before training. The exact
base-decoder auxiliary paths also use explicit BF16 autocast because they bypass
Accelerate's top-level model wrapper. Sequential rationale training releases
the completed auxiliary graph and unused CUDA cache blocks before constructing
the primary Kimi graph. The exact arms and stop conditions are in `config.yaml`.

Originally each objective first ran one optimizer step on the single longest
training example, using sequential sampling and accumulation 1. Corrected runs
instead preflight two longest training examples at accumulation 2, to exercise
the accumulation path. Full runs retain the frozen effective batch of 32.

## Evaluation and promotion

Evaluate every arm on the frozen 3,012-row STRIDE-test plus
Gloom-Exfiltration ID suite. The primary metric is two-source macro normalized
pAUROC@20. An arm must improve at least 0.005 over the existing `2e-5` control,
avoid a source regression worse than 0.01, and avoid macro-Brier regression
worse than 0.005. Strict OOD remains untouched.

If at least one arm passes, run only the smallest motivated combination of the
best rationale weight and best MIL pool. A one-seed result remains exploratory
and requires replicated confirmation before promotion.

## Commands

On the prepared Lambda checkout:

```bash
python -m experiments.monitoring_objective_ablation.prepare
python -m experiments.monitoring_objective_ablation.run_lambda
```

Runtime status and artifacts are written below
`results/monitoring_objective_ablation/`.

## Runner maintenance (2026-09-05)

Campaign lifecycle reporting now uses `gleipnir.campaign_status`, shared with
the config-driven systems runner. Status retains the existing summary fields
and adds per-job GPU assignment, start/end times, errors, and an update timestamp.
A lane failure is published immediately, including metadata-validation failures,
even while another lane is still running. The status file has one runner as its
owner; it is not a resume checkpoint or a cross-process scheduling lock.

The ID configuration now includes the source and manifest checksums from the
existing LR-screen ID contract. Their omission previously made the evaluator
fail before inference. This repair changes neither examples nor scoring.
The runner also explicitly performs the existing bounded causal-master/vLLM
parity gate, including a nonzero adapter-effect check, before full evaluation.
The already-running campaign uses its original training revision; its parity
check is run separately on the free GPU before its scheduled evaluation.
The evaluator's expected training target is now explicit in the model-group
configuration (`kimi_soft_plus_auxiliary` here; historical configs still default
to `kimi_soft`). Repeated `--only-job` options select completed adapters in one
persistent engine without changing the frozen config hash or manifest order.
This lets evaluation overlap a remaining training lane and later resume from
the same prediction artifacts.
Evaluation config, input provenance, and job compatibility are checked before
training starts, so these integration errors cannot consume a full campaign
before surfacing.

Prepare corrected rationale runs without rematerializing data or overwriting
the original adapters:

```bash
python -m experiments.monitoring_objective_ablation.prepare \
  --reuse-jobs results/monitoring_objective_ablation/jobs.jsonl \
  --rerun-kind rationale \
  --result-dir results/monitoring_objective_ablation_accumulation_v2
```

The resulting five-job manifest points only the two rationale entries at new
directories; MIL entries keep their original paths. CPU accumulation tests cover
windows of 1, 16, and 32 with and without completion labels. Before corrected
full training, run a two-microbatch longest-sequence GPU preflight. Reruns use
the original frozen scientific recipe and distinct per-lane status files.

The same runner supports `--phase train --only-job NAME --gpus 1` for a single
available GPU, then `--phase evaluate --gpus 0` after all adapters exist. Pass
the repaired `--jobs`, `--result-dir`, and `--evaluation-config` paths to both
phases, and a separate `--status` path for each independently launched process.
Training subsets preserve the complete evaluation manifest; they do not change
the scientific design. Evaluation always gates serving parity, evaluates all
five manifest entries, and summarizes. Default `--phase all --gpus 0 1` retains
the original round-robin lane order. GPU ownership is the caller's responsibility;
the status record is deliberately not an inter-process GPU scheduler.
