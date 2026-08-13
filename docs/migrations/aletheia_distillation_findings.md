# Migrated deception-distillation findings

This document carries forward the findings that matter to Gleipnir from the
Aletheia's Quest privileged-information distillation work. Competition-specific
submission mechanics, NDIF behavior, and ignored cache paths were deliberately
not ported. Consult the precursor repository for the full experimental record.

## Supervision signal

The selected one-epoch, varied-only Qwen3.5-9B LoRA trained on compact privileged
teacher summaries reached 0.9155 local-test balanced accuracy (0.8278 on the
varied subset). A matched prediction-only target ablation reached only 0.8631
overall (0.7917 varied). This is strong evidence that compact teacher evidence
provided useful supervision beyond binary labels.

The original successful recipe used a rank-16/alpha-32 LoRA, one epoch, AdamW at
`5e-5`, effective batch size 32, and Qwen3.5-9B. Treat it as an initial baseline,
not a permanently frozen optimum for new datasets or current libraries.

Teacher prompt wording sweeps did not reliably improve the selected baseline. A
marginal validation winner regressed on one frozen confirmatory test, fixing
three baseline errors but breaking five correct rows. Do not spend annotation
budget on unconstrained wording search without a stronger hypothesis.

## Soft labels and direct scoring

Literal next-token `0|1` logprobs from strong teachers proved useful as soft
targets. Always store raw label logprobs and derive the positive score as:

```text
exp(logprob(1)) / (exp(logprob(0)) + exp(logprob(1)))
```

Keep the continuous margin separate from any binary threshold. Report score
uniqueness and ties: some strong adapters produced many duplicate scores even
when AUROC was high.

For student evaluation, direct logits at a fixed prompt boundary often beat
generated binary outputs and avoided generation cost. Generated reasoning should
therefore earn its inference cost through held-out evidence; training with
rationales does not imply reasoning must be generated at deployment.

## Scaling evidence

A larger Qwen3.5 teacher and later Kimi K3 annotations improved the soft-target
distillation line. The final Phoenix 8.1 competition monitor, trained from a
13,149-row Kimi K3 binary-logit artifact spanning competition and Liars' Bench
sources, achieved 0.9661 mean private per-dataset AUROC. The public annotation
artifact is `Jazhyc/aletheias-phoenix-v8-1-kimi-k3-distillation`; use its
`render_prompts.py` to reconstruct licensed source conversations rather than
copying them into this repository.

The artifact separated source references from teacher outputs, pinned revisions,
stored route metadata and exact prompt hashes, and included literal label
logprobs plus normalized soft targets. Preserve those provenance properties in
new annotation schemas.

Post-submission half-dose and interpolation sweeps did not justify replacing the
full-dose model. Scaling studies need source-grouped held-outs and frozen
promotion criteria; example count alone is not a causal explanation.

## Dataset composition warning

The old varied-deception train/validation/test data was entirely Qwen-derived,
while Gemma and Nemotron appeared only in instructed deception. Consequently,
varied-only improvements were also Qwen-only adaptation. Gleipnir must not call
such gains general varied-behavior transfer. New mixtures should include model
family, behavior family, and honest-control coverage explicitly.

Factual incorrectness is not equivalent to belief-based deception. Atomic fact
audits recovered some false negatives but harmed more honest rows, especially
when ordinary factual errors or negated claims were treated as deceptive. Keep
factual verification as a separable feature or teacher signal unless a held-out
gate demonstrates net benefit.

Knowledge-report and insider-action scenarios failed in opposite directions
under the general factual judge. This supports a broader behavior ontology and
balanced controls, but does not justify dataset-name routing or post-hoc
specialists. Route only on observable, independently validated semantics.

## Optimization lessons

The fast Qwen3.5 H100 recipe was not just the causal model wrapper. It used
pinned `flash-linear-attention==0.5.2` and `fla-core==0.5.2` gated-delta kernels,
microbatch 8, accumulation 4, effective batch 32, gradient checkpointing, and
eager execution. It measured 8.811 samples/s for rank 16, versus 4.279 samples/s
for the best eager Torch-fallback batch and about 1.56 samples/s for the later
microbatch-2 fallback run. Batch 32 was slower and consumed about 80.98 GB.
Launchers must verify FLA before importing Qwen, fail rather than silently fall
back, and use standard NF4/double-quantized/BF16 QLoRA when higher adapter ranks
would otherwise force smaller microbatches.

GRPO continuation after the selected SFT did not beat the original student on a
fair local test. Extra generated reasoning became longer and still failed on
subtle false supporting details. Three-epoch GRPO and the tested SDPO setup also
regressed or became unstable. Use supervised distillation as the default until a
new online-objective design has a clear held-out advantage.

Muon was useful enough to preserve as an optional optimizer for 2D LoRA matrices,
with AdamW for remaining parameters. It is now isolated in
`gleipnir.training`; optimizer comparisons should hold the data, initialization,
effective batch size, and validation protocol fixed. The migrated implementation
now uses scheduler-managed learning rates and Moonshot's per-matrix
`match_rms_adamw` scaling. The older independent `muon_learning_rate` bypassed
Transformers warmup and decay and must not be used for comparisons.

In the first corrected rank-64, 25%-data screen, Muon at `5e-5` improved macro
AUROC by 0.00312 over paired AdamW and won the predeclared primary metric. Muon
at `1e-4` was slightly lower in AUROC but had the best macro balanced accuracy
and Brier score. Muon lost at `1e-5` and `2e-5`, so RMS matching makes the rates
comparable but does not make optimizer performance learning-rate invariant.
Treat both `5e-5` and `1e-4` as confirmation candidates, not promoted defaults,
until replicated 50%-data evidence is available.

## Adapter packaging

Keep FP32 master checkpoints for training and archival. In one matched 400-row
parity canary, BF16 LoRA weights were bit-for-bit score identical to FP32 while
halving adapter size; FP16 changed scores and ordering. Do not assume that result
transfers to new weights or inference backends. Export BF16 only after a matched
parity canary that checks scores, ordering, tensor names, and adapter layout.

## Implications for Gleipnir

1. Begin with a reproduced hard-label baseline, then add soft margins and compact
   evidence as separate ablations.
2. Evaluate by behavior family, model family, and source lineage.
3. Prefer direct continuous scoring unless generated reasoning wins fairly.
4. Treat paid teacher caches as immutable, resumable, prompt-hashed datasets.
5. Expand beyond deception with an explicit versioned ontology and honest hard
   negatives, not competition-specific routers.
