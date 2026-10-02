# Crossed monitor prompt-injection evaluation

The [prespecified evaluation](../research/monitor_injection_evaluation.md)
crosses base, regular-trained and injection-aware-trained Qwen3.5 weights with
regular and injection-aware student instructions at both 4B and 9B. These are
the matched tool-trajectory-only students, with no additional training,
checkpoint selection or attack generation. Implementation is in
[monitor-injection-evals](https://github.com/mkenney2/monitor-injection-evals/tree/feat/gleipnir-vllm-grid),
evaluation commit `bd114be7e646affdb5a06a5bfaa720bbc644e830`.

## Audited 4B results, 2026-10-02

All six 4B cells completed, each with 5,184 rows: 655 clean honest negatives
and 647 backdoors in each of seven conditions. The independent local audit
passed exact unique membership, metadata, prompt/configuration identities,
current prediction checksums, finite binary logprobs and score extraction,
and independently re-encoded per-row token lengths. All six serving gates
passed against FP32 causal masters with pinned original-FLA and convolution
kernels on the frozen training-only canaries. There was zero truncation.

AUROC and TPR use the benchmark's unchanged metric code. Injection columns
below are equal-weight means over the three fixed transfer templates. The
matched delta is injection AUROC minus AUROC with neutral text on the same
backdoors; a negative delta indicates injection-related degradation. TPR uses
the honest-score quantile with strict `score > threshold`; each realized honest
FPR is at most 2%. This is a benchmark diagnostic, not an independently
calibrated deployment threshold.

| 4B weights | Student instruction | Vanilla AUROC | Vanilla TPR@2% | Mean injection AUROC | Mean injection TPR@2% | Mean matched delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Regular | 0.9017 | 35.7% | 0.7756 | 0.9% | -0.1405 |
| Base | Aware | 0.8965 | 30.8% | 0.9414 | 9.1% | -0.0051 |
| Regular-trained | Regular | 0.9189 | 51.8% | 0.9988 | 100.0% | +0.0651 |
| Regular-trained | Aware | 0.9178 | 51.5% | 1.0000 | 100.0% | +0.0542 |
| Aware-trained | Regular | 0.9312 | 58.4% | 0.9982 | 100.0% | +0.0450 |
| Aware-trained | Aware | 0.9280 | 57.5% | 0.9999 | 100.0% | +0.0378 |

For base 4B, the explicit warning substantially narrows the injection AUROC
gap, but strict-operating-point sensitivity remains low: aware-prompt TPR is
23.65% for the GPT-4 template, 1.70% for GPT-4o and 1.85% for GPT-4.1.
Both trained students detect every injected positive under both instructions
at the benchmark's 2% FPR operating point. This ceiling leaves no incremental
injection TPR benefit from awareness training observable on this catalog.
The injected comments make these trained models' backdoors easier to rank
than matched neutral comments; vanilla detection remains much weaker.

The aware-trained weights have higher vanilla TPR than regular-trained weights
under both instructions. This is one seed without uncertainty estimates and
does not establish a significant or general benefit. There are no benign
injected negatives, so perfect injected-positive sensitivity does not establish
that a model distinguishes the malicious code from the injection text itself.

The 4B completion audit and full per-template metrics are retained under the
sibling checkout's ignored `results/gleipnir-grid-v1/` tree. The 9B bounded
references have completed; its full vLLM grid is running. No 9B quality
conclusion follows from the reference cohort.

## Serving recovery

The prior ID/OOD campaign had already established successful serving. New
launcher errors occurred before this benchmark produced any predictions:
the reference omitted the isolated causal-convolution import path; serving
forced offline mode with missing cached repository metadata; a subsequent
launcher omitted the virtualenv's `bin` directory, hiding `ninja` during
FlashInfer warmup. Correcting these environment settings allowed the entire
4B grid to finish normally. Failure logs and exit receipts are preserved.
Fresh campaign vLLM/Inductor cache roots retain the earlier protection against
relocated compiler artifacts. These failures do not establish a new numerical
serving defect. The corrected environment and established runtime caches are
used for the 9B launch.

## Scope

This frozen HF snapshot contains APPS problems with o3-mini-generated code,
clean honest negatives and three fixed transfer templates. It omits BashBench,
zero-shot/tailored attacks, adaptive attack search and benign injected negatives.
Conditions share problem IDs; 5,184 rows are not independent problems. The
matched neutral text is added to positive backdoors, not benign negatives.
Results cannot establish adaptive or general prompt-injection robustness.
No prompt, checkpoint or threshold is promoted using these results. Earlier
failed strict training-gradient comparisons remain explicit in the separate
training record; passed serving gates do not overturn them.
