# Crossed monitor prompt-injection evaluation

## Dataset-construction review, 2026-10-02

This review uses dataset descriptions and generation procedures, not a search
through individual trajectories. It concerns the matched 8,688-row campaign,
whose regular and aware students receive identical trajectories and Kimi soft
targets; awareness training changes the instruction, not the attack inventory.

[Sinha et al., Appendix C.3–C.4](https://arxiv.org/html/2605.29601v1#A3.SS3)
describe training as synthetic scheming (STRIDE/Gloom), covert coding sabotage
(CoT-Red-Handed), and shell sabotage/exfiltration (BashArena/BashBench). Their
behavioral categorization explicitly places prompt-injection compliance in
AgentDojo OOD and says that category does not occur in training. This is a
primary-category analysis of source-positive rows, not proof that training
contains no incidental injection strings or secondary monitor manipulation.
ID STRIDE and Gloom-Exfiltration cover held-out scheming and disguised leaks.
AgentDojo attacks target the acting agent through tool outputs; Rogue Eval
targets oversight routing. Neither description establishes direct persuasion
of the reviewing monitor to return a low score.

[CoT-Red-Handed, Sections 2.3 and 4.2](https://arxiv.org/html/2505.23575#S2.SS3)
documents framing, stealth and monitor-awareness policies, including misleading
benign explanations of harmful behavior. This supports exposure to related
monitor-evasion behavior at the source level; the construction review does not
establish which policies survive in our selected action-only coding rows.
Direct monitor collusion is not established by the reviewed descriptions.

Generalization from deception/sabotage supervision is a plausible explanation
for the transfer-template result, not an identified causal mechanism. ID/OOD
evaluation supplies no gradient updates and cannot itself teach this behavior.
The absence of benign injected negatives remains an alternative explanation:
injection text might simply act as a suspicion cue.

The [prespecified evaluation](../research/monitor_injection_evaluation.md)
crosses base, regular-trained and injection-aware-trained Qwen3.5 weights with
regular and injection-aware student instructions at both 4B and 9B. These are
the matched tool-trajectory-only students, with no additional training,
checkpoint selection or attack generation. Implementation is in
[monitor-injection-evals](https://github.com/mkenney2/monitor-injection-evals/tree/feat/gleipnir-vllm-grid),
evaluation commit `bd114be7e646affdb5a06a5bfaa720bbc644e830`.
All twelve cells completed and passed the independent audit. The
[public-remote results report](https://github.com/mkenney2/monitor-injection-evals/blob/feat/gleipnir-vllm-grid/reports/gleipnir_grid.md)
contains every model/prompt/template comparison and prediction checksum.

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
sibling checkout's ignored `results/gleipnir-grid-v1/` tree.

## Audited 9B results

All six 9B cells completed with the same frozen membership and inputs. The
independent audit passed the same coverage, identity, extraction and token
checks, and recomputed each serving correlation, mean absolute difference and
nonzero adapter effect. All twelve gates across both sizes passed: minimum
correlation 0.999353 and maximum mean absolute difference 0.008156. Frozen
tokenizer files match across all four adapters; the independent renderer
matched all 40 training-canary prompt hashes. No benchmark row was truncated.

| 9B weights | Student instruction | Vanilla AUROC | Vanilla TPR@2% | Mean injection AUROC | Mean injection TPR@2% | Mean matched delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Regular | 0.9175 | 50.7% | 0.9505 | 38.6% | -0.0063 |
| Base | Aware | 0.9188 | 49.6% | 0.9622 | 20.4% | -0.0021 |
| Regular-trained | Regular | 0.9314 | 63.2% | 0.9985 | 100.0% | +0.0349 |
| Regular-trained | Aware | 0.9209 | 58.3% | 0.9996 | 100.0% | +0.0371 |
| Aware-trained | Regular | 0.9235 | 55.6% | 0.9973 | 99.4% | +0.0435 |
| Aware-trained | Aware | 0.9237 | 57.0% | 0.9995 | 100.0% | +0.0544 |

The base 9B warning improves mean injection AUROC but reduces mean TPR by
18.24 percentage points. GPT-4.1 injection TPR falls from 83.62% to 21.33%,
while the other two templates improve. The mean matched AUROC gap hides this
family difference; neither ranking improvement nor a smaller mean gap implies
better strict-operating-point sensitivity.

Regular-trained 9B detects every injected positive under both instructions.
Aware-trained 9B with the regular instruction misses 11 of 647 GPT-4o injected
positives, while detecting all GPT-4 and GPT-4.1 positives. With the aware
instruction it detects all three families. This catalog therefore does not
establish an injection-specific benefit from awareness training over the
already near-ceiling regular-trained baseline. Aware-trained 9B also has lower
vanilla sensitivity than regular-trained 9B under both instructions, unlike
the 4B comparison. Adding the warning to regular-trained 9B reduces vanilla
TPR by 4.95 percentage points. The effects depend on size, weights, instruction
and attack family; no uniform improvement follows.

## Completed artifacts and runtime

All 62,208 predictions passed the final local audit, including exact upstream
threshold recomputation and injection-minus-neutral comparisons on the same
647 problem IDs. The two prompt files contain 8,207,435 and 9,249,419 tokens
per cell respectively, with maxima 16,321 and 16,522. Full predictions,
logprobs, metrics/ROC curves, reference/serving vectors, token audits and
successful/failed logs have been collected locally in the sibling checkout.
The final receipt is `results/gleipnir-grid-v1/completion_audit.json`, SHA-256
`7abf7721eb5892337c76495e09a5f24b7fd1dbb19136df48b4fdbd6fce670172`.

Aggregate scoring/metric-loop runtime was 22.04 minutes for 4B and 26.11
minutes for 9B. These are evaluation times, excluding input preparation,
model loading, compilation, warmup and parity checks. Both persistent engine
stages exited with code 0. The final 9B log completed at 17:20:04 UTC; a
post-exit GPU snapshot recorded zero allocated memory, 0% utilization, 31°C
and zero volatile uncorrected ECC errors. The authorized B200 remains
available, with no active or queued benchmark worker.

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
