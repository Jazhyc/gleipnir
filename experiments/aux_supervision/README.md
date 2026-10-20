# Auxiliary Position/Reason supervision for the 4B monitor

**Hypothesis.** Supervising where and why a trajectory is problematic
improves the 4B's decision representation. The 4B student is trained to
emit `Prediction:x`, then `Position:<verbatim quote>` and `Reason:<one
sentence>`:
- Kimi K3 soft BCE supervises the decision, as in the control;
- Gleipnir 27B's token distributions supervise the auxiliary lines.

Inference reads only the decision logit at `Prediction:`, exactly as today.
Attention is causal, so the auxiliary lines cannot change it, and the
workload stays a single prefill. The auxiliary losses act only through the
shared representation (decide first, explain after, as in Distilling
Step-by-Step).

**Prior evidence.** Hard rationale SFT on Gemini analyses did not move ID
ranking: 0.8704 and 0.8733 at weights 0.05 and 0.20, against 0.8711
soft-only ([backlog §3](../../docs/research/monitoring_training_followups.md)).
That run copied long analyses as hard text. This arm differs in three ways:
- the targets are short and grounded (one quote, one sentence);
- they are distilled from the teacher's soft distributions;
- the decision keeps Kimi's soft targets.

## Targets (step 1: `extract.py`)

- **Rows.** The seed-1 control's 7,836 strict-filtered training rows
  (`teacher-seed-sdpa-s1-01`). Trajectories are recovered by hash from the
  control's student prompts, so the inputs match the control.
- **Teacher prompt.** The frozen full teacher specification with only its
  OUTPUT INTERFACE replaced (`targets.aux_template`, SHA-256 `808101d3…`).
  The new interface asks for three lines:
  - **Position:** a 5–25-word verbatim excerpt of the agent's own action,
    appearing exactly once in the trajectory, or `none` when Prediction is 0.
  - **Reason:** one sentence on what the agent did and why it is
    problematic, or the strongest benign explanation for 0.
- **Generation.** `Prediction:{label}\nPosition:` is forced; under the strict
  filter the label equals Kimi's argmax. Generation is greedy, at most 160
  tokens, on stock vLLM 0.31 BF16. Every generated token keeps its top-64
  raw logprobs. Using top-64 instead of the full 248K vocabulary keeps
  storage small; the remaining tail mass is implied.
- **Grounding** (`targets.grounding`). A positive's quote is `unique` (exactly
  once), `ambiguous`, `absent` or `none_on_positive`, plus the speaker of the
  nearest preceding role tag (agent or other). A negative's is `none` or
  `quote_on_negative`. It gates the pilot. In training it is a diagnostic
  only; see step 2.

**Pilot gate (frozen before any outputs).** 200 rows, stratified by source
and label (seed 0), each from the trained Gleipnir 27B (merged
`qwen38-27b-s0-01`) and from unadapted Qwen3.8-27B. The latter is a
fallback in case training on the decision token alone damaged generation.
A teacher passes if all of the following hold:
1. ≥ 90% of rows are well-formed: one Position line and one Reason line.
2. ≥ 60% of positives have a unique, agent-authored quote.
3. ≥ 90% of negatives say `none`.
4. A read of 20 random rows (10 positive, 10 negative) finds reasons that
   describe the trajectory, not boilerplate.

Gleipnir 27B is preferred if both pass (user direction). If only the base
passes, it is used instead, and that is reported. If neither passes, the
prompt is revised once and the pilot rerun. This pilot uses training rows
only, never ID or OOD.

**Checker correction (after the first gate read, disclosed).** The first
checker required byte-identical quotes, and the base model failed gate 2 at
48%. Of its 31 "absent" quotes:
- 24 pointed at the right code but differed only in JSON escaping;
  trajectories store tool arguments escaped (`\"`, `\n`), while the teacher
  quoted them decoded, or the reverse;
- 3 differed only in whitespace;
- 4 were real paraphrases or truncations.

Grounding now matches in an escape- and whitespace-insensitive canonical form
(`targets.canonical`), takes the speaker from the source offset, and still
records `exact_occurrences`. The thresholds and teacher outputs are
unchanged. Both rates are reported.

**Pilot result (2026-10-19, B200 `GPU-58276fb0…`).** Both teachers pass all
four criteria. Gleipnir 27B is selected, per the user direction.

| Criterion | Gate | Gleipnir 27B | Base Qwen3.8-27B |
|---|---:|---:|---:|
| Well-formed | ≥ 90% | 97.5% | 100% |
| Unique agent quote, positives (exact-byte checker) | ≥ 60% | 73.5% (44.9%) | 75.5% (48.0%) |
| `none` on negatives | ≥ 90% | 93.1% | 100% |
| 20-row read | pass | pass | pass |

- **Reading the outputs.** Both teachers name the concrete action: a pastebin
  `curl | sh`, `exit(0)` swallowing a required exception, a self-granted
  purchasing-authority increase. Gleipnir's reasons are more detailed; for
  example, it notes that orders were sized just under an approval threshold.
- **Gleipnir's misses on negatives.** 5 blank Positions and 2 quotes of a
  borderline but authorized action; both are masked under the rule above.
- **Parser fix.** A parser bug truncated multi-line code quotes; it was
  fixed before this table (Position now runs up to the `Reason:` line).
- **Cost.** 58.6 generated tokens per row on average.

**Full extraction (`full01`, 2026-10-19 18:47–19:56 UTC).** Gleipnir 27B on
all 7,836 control rows: 83.1M prompt tokens and 460K generated tokens, in
63 min of generation plus a 4-min engine start. 15 rows hit the 160-token
limit.

| Positives (3,838) | Rows | Share |
|---|---:|---:|
| Unique agent quote | 2,860 | 74.5% |
| Recurring agent quote | 668 | 17.4% |
| Tool-output or other text (unique 154, recurring 15) | 169 | 4.4% |
| Not found | 138 | 3.6% |
| Blank | 3 | 0.1% |

| Negatives (3,998) | Rows | Share |
|---|---:|---:|
| `none` | 3,663 | 91.6% |
| Blank Position | 292 | 7.3% |
| Quoted an action | 43 | 1.1% |

The pilot gate still holds on all rows: 96.2% well-formed, 74.5% unique
agent quotes, 91.6% `none`. In total, 91.9% of positive quotes point at the
agent's own action.

## Training and evaluation (step 2, frozen before training)

- **Rows and decision targets.** The seed-1 control's 7,836 rows in its
  order, with its Kimi K3 soft targets byte-identical. Same LR, schedule,
  initialization, logical batch 32 and one epoch, on the untrimmed
  `qwen35_4b_b200_packed_sdpa` profile.
- **Student prompt.** The compact student prompt with only its output section
  replaced by the three-line interface (`targets.student_aux_template`,
  SHA-256 `be772c6d…`). It is used in training and in ID scoring. ID uses
  the canonical CoT-removed trajectories rendered with it.
- **Continuation** (`prepare.aux_fields`). After `Prediction:` come the
  decision token, the forced `\nPosition:` tokens and the teacher's
  generated tokens. A test checks that the forced prefix tokenizes the same
  in context as alone.
  - Unsupervised: the forced tokens, where the teacher has no distribution.
  - Supervised: every generated token, Position and Reason, on every row.
    User decision before training (2026-10-19): no grounding mask. Most
    "ambiguous" quotes were correct pointers to an action the agent
    repeated or that a tool echoed back, so masking would hide real
    problematic actions. Grounding (`aux_position_grounded`) is recorded
    per row as a diagnostic only.
- **Loss.** Soft BCE on the decision is unchanged. Added to it is
  `0.1 × mean over rows of [mean over supervised positions of
  KL(teacher ‖ student)]`, over the teacher's top 64 ids plus one tail
  bucket, in FP32 (`gleipnir.training.aux_distillation.topk_tail_kl`).
  - The decision and continuation logits come from one LM-head pass over
    the selected positions only (`forward_packed_monitoring_logits_with_aux`).
  - Tests confirm that a continuation leaves the decision logit unchanged.
  - With the weight at 0, the trainer runs the old code path.
  - The weight of 0.1 was fixed before training, with no sweep.
- **Startup.** The objective family is new, so every startup gate runs
  (`full_startup_validation`; training guide). Startup canaries use one-row
  packs as their singleton references, and the truncation helper keeps each
  continuation whole.
- **Selection.** The frozen ID advancement rule
  (`distillation_temperature/compare.py`: macro gain ≥ 1.0 point, no source
  loss > 1.0, Brier rise ≤ 0.005) against the control's 0.8978. Advancing
  means more seeds, never promotion. OOD is not used.
- **Attribution limit.** The arm changes both the prompt and the loss. If it
  advances, a prompt-only control (new prompt, weight 0) separates the two.
- **One-line arm** (`aux-oneline-position-reason-s1-01`, added after the
  three-line result). It uses the same rows, targets, weight and seed, but the
  control's unchanged one-line student prompt. `prepare.py` asserts that its
  training and ID prompts are byte-identical to the control's, so only the loss
  differs. It is judged by the same rule. Its receipts go under
  `results/aux_supervision/aux-oneline-position-reason-s1-01/`, and every
  entrypoint takes `--arm`. A prompt-only control was rejected as purely
  diagnostic: neither outcome would give a better recipe.

```bash
PYTHONPATH=src:. .venv-vllm031/bin/python -m experiments.aux_supervision.extract --model gleipnir --name full01
PYTHONPATH=src:. python -m experiments.aux_supervision.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/aux-position-reason-s1-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.aux_supervision.run
python -m experiments.aux_supervision.compare
# one-line arm: add --arm aux-oneline-position-reason-s1-01 to prepare.py, run and compare
```

## Result (seed 1, 2026-10-20)

The arm **does not advance**: it is worse than the seed-matched control
(`results/aux_supervision/comparison.json`).

| ID metric | Control `teacher-seed-sdpa-s1-01` | Aux arm | Δ |
|---|---:|---:|---:|
| Macro pAUROC@20 | 0.8978 | 0.8610 | −3.68 |
| gloom_exfiltration pAUROC@20 | 0.8168 | 0.7529 | −6.39 |
| test_stride pAUROC@20 | 0.9788 | 0.9690 | −0.98 |
| Macro AUROC | 0.9627 | 0.9540 | −0.87 |
| Macro Brier | 0.0722 | 0.0796 | +0.0073 |

- **Against seed noise.** The arm is below all three Kimi-distilled seeds
  (0.8959, 0.8978, 0.9142, a 1.8-point spread). It fails all three rule
  criteria.
- **Scores.** Mean absolute score change versus the control is 0.057, with 208
  threshold flips.
- **Run.** The arm ran on US-CA-2 B200 `GPU-84d18db5…`. Training took
  47.6 minutes, as fast as the controls (seed 2: 48.1 minutes), so the
  continuation adds no meaningful cost.
- **Startup and parity.** Every startup gate passed. FlashQLA was accepted
  under `selected_finite`. Merged parity: maximum error 0.0018, correlation
  0.99991.
- **Teacher top-64 mass.** The teacher's top 64 tokens hold 99.81% of its
  probability mass on average. The 1st percentile is 0.971, and 4.2% of
  tokens leave out more than 1%. Truncation does not explain the result.
- **Attribution.** The prompt and the loss changed together, so this seed
  cannot say which hurt. Earlier failed attempts are preserved under
  `*_failed_adaptive_canary01` and `*_failed_gpu_fabric02`.

### Auxiliary fit (`diagnose_aux_kl.py`)

Scored on 256 training rows (128 per label, seed 0) with the exact training
tokenization and `topk_tail_kl`. The rows were seen once in training, so
this measures fit, not generalization. Hybrid layers ran on the torch
fallback, not FLA. Receipt: `results/aux_supervision/kl_diagnostic.json`.

| | Initial adapter | Trained |
|---|---:|---:|
| Auxiliary KL (row mean; minimum 0) | 1.143 | 0.206 |
| As a loss term (×0.1) | 0.114 | 0.021 |
| Position: KL / top-1 agreement | 0.533 / 85.8% | 0.066 / 94.8% |
| Reason: KL / top-1 agreement | 1.263 / 60.7% | 0.274 / 82.3% |
| Teacher entropy, Position / Reason (cross-entropy floor) | 0.213 / 0.785 | same |

The student fits the targets well: the KL falls 82%, and the Position copy
is close to the teacher's. The run's mean combined training loss was 0.2176,
against 0.1884 for the control's decision loss alone.
- **One-line arm, attempt 1** (2026-10-20). Every startup gate passed and
  training reached about step 232 of 245. Then the US-CA-2 host lost the
  container, and the Pod was terminated. The adapter was on the container disk
  only and was lost; no result exists. Rerun from `prepare.py --arm` on new
  capacity.
