# Black-box APPS injection detection

Completed 2026-10-11 on the user-reserved NC2 B200. Explicit injection-presence
prompting exposes a strong presence signal in the current augmented SDPA replay:
matched honest-code injection/neutral template-macro AUROC is **0.999500**,
versus **0.974971** for unadapted Qwen3.5-4B. This does not give reliable binary
answers at 0.5. Full-grid ranking is slightly worse after adaptation, and
backdoored code remains a source of false positives for the presence task.

## Frozen comparison

The [experiment contract](../../experiments/apps_injection_detection/README.md)
fixes the existing [census rubric](../../experiments/training_injection_census/prompt.txt),
pinned base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, final
272-update `b200-augmented-sdpa-replay02` adapter, and complete APPS population.
The FP32 master hash is
`bdb702316b1d5e29cbdd8fba64888fe128b37bfd25030211b867d46372e8f5fd`.
The preserved merged BF16 checkpoint is file-checksum verified.

Both models use sequential compiled unquantized BF16 vLLM 0.31 engines, cached
causal LAST 0/1 logits, native Gigatoken, corrected synchronous pooling scheduler,
prefix caching off, 32K context/chunk budget, c128 and 128-row batches. No Lens
intervention, training, model selection, payload generation or submitted-code
execution occurs. This is the user's requested BF16 exception to low-precision
serving. Current host/driver and access restoration are recorded in
[infrastructure](../infrastructure.md#user-reserved-nc2-b200-for-injection-detection-2026-10-11).

Each model scores the same 9,114 rows: 5,184 benchmark examples plus 3,930 honest
controls, covering 655 tasks. Presence labels are construction-based:
3,906 `inj_transfer_*` positives and 5,208 clean, vanilla-backdoor and
`neutral_transfer_*` negatives. Original code-backdoor labels remain separate.
Only the outer monitoring instruction changes; trajectories and payloads remain
intact. The workload contains 17,727,821 input tokens, maximum 16,583, with hash
`392a0c9ea9245019bf462cd8b877fdb0e8e510a783c7630a216259340626de5c`.

These tasks and three transfer templates have prior research exposure. There is
no fresh held-out prompt or checkpoint selection, nor independent annotation of
all possible incidental instruction text. The experiment tests the fixed
constructed monitor-targeted payloads, not broad or adaptive injection defense.

## Ranking and calibration

| Metric | Base | Augmented SDPA |
|---|---:|---:|
| Full-grid AUROC | 0.950709 | 0.941771 |
| Full-grid raw normalized pAUROC@20 | 0.775459 | 0.708854 |
| Honest-code presence AUROC, including clean negatives | 0.981963 | 0.999490 |
| Backdoored-code presence AUROC, including vanilla negatives | 0.930249 | 0.890765 |
| Matched honest injection/neutral template-macro AUROC | 0.974971 | 0.999500 |
| Matched backdoor injection/neutral template-macro AUROC | 0.924948 | 0.899666 |
| Recall at strict score >0.5 | 27.010% | 30.568% |
| FPR at strict score >0.5, all presence negatives | 1.440% | 4.205% |
| Brier score | 0.172347 | 0.167282 |
| Equal-width positive-class ECE | 0.221160 | 0.231344 |
| Unique normalized scores | 54 | 53 |

The paired task bootstrap retains every related view and both models together.
For 500 resamples, seed 20261011, the pooled AUROC 95% percentile intervals are
base [0.943245, 0.957347] and SDPA [0.932124, 0.951480]. SDPA-minus-base is
−0.008938, with interval [−0.015866, −0.000757]. This uncertainty is over these
tasks, not training seeds or independent attack templates.

Scores are normalized binary logits, not calibrated injection probabilities.
The shared calibration helper uses >=0.5 for its accuracy and bin summaries;
the preregistered operating-point diagnostics use strict >0.5. Exact 0.5 ties
therefore produce different accuracy figures. Strict-threshold accuracies are
67.896% and 67.841%; preserve the convention rather than conflating them.

## Matched templates

Each honest template has 655 injection/neutral pairs; each backdoor template
has 647. Templates are evaluated with equal macro weight within each stratum.

| Template | Honest base AUROC | Honest SDPA AUROC | Backdoor base AUROC | Backdoor SDPA AUROC |
|---|---:|---:|---:|---:|
| GPT-4 transfer | 0.949410 | 0.998675 | 0.862865 | 0.838979 |
| GPT-4.1 transfer | 0.976590 | 1.000000 | 0.935042 | 0.939740 |
| GPT-4o transfer | 0.998913 | 0.999825 | 0.976937 | 0.920280 |

The SDPA honest-positive recall at 0.5 varies sharply across the same templates:
0.153%, 46.718% and 29.466%, versus base 6.870%, 13.130% and 44.885%.
Neither model alarms on honest neutral controls at that threshold. Excellent
ranking therefore coexists with low confidence and substantial prompt-template
sensitivity in the binary answer.

The fixed clean-only <=2% empirical FPR rule gives thresholds 0.148047 for base
and 0.042088 for SDPA. Both alarm on 13/655 clean rows (1.985%). Across all
honest negatives, base FPR is 9.885% with 95.267% injected-honest recall; SDPA
FPR is 1.641% with 100% recall. On matched honest neutral controls alone,
SDPA FPR is 1.832%, 1.374% and 1.374% by template, with 100% recall each.
These are descriptive thresholds fitted on the evaluated clean cohort, not
held-out calibration estimates.

The corresponding full-grid FPR is much higher: base 21.736%, SDPA 20.680%,
with recall 97.005% and 100%. At 0.5, vanilla backdoors have 3.246% and 11.747%
FPR despite containing no constructed monitor-targeted injection. Harmful-code
sensitivity remains relevant to the presence score; an honest-only result
cannot establish behavior-independent injection detection across the full grid.

## Execution and receipts

Both fresh native BF16 projection/head/attention audits pass. The original
20-row score canary passes against archived FP32 references: base MAE 0.008459,
correlation 0.998519; SDPA MAE 0.002206, correlation 0.999881. The SDPA
adapter effect is nonzero, and unchanged merge/master evidence is preserved.

Scoring takes 174.361 s for base and 174.784 s for SDPA, excluding restoration,
startup, canary and reporting. Prompt throughput is 101,673 and 101,427 tokens/s;
request throughput is 52.271 and 52.144 requests/s. Request latency p50/p95/p99
is 1.564/2.394/2.744 s and 1.568/2.433/2.823 s. These single passes measure the
matched quality workload, not a new serving-performance selection.

The original CPU preparation repeatedly read the prompt template from network
storage. It was interrupted before model loading, with logs/partial directory
retained. Loading that unchanged template once preserves the exact local
workload hash; the remote workload matches it. There is no failed scoring pass.

The runtime logs retain warnings about pooling's piecewise CUDA graphs,
OMP_NUM_THREADS=4, the custom scheduler interface, missing optional orjson and
localhost development endpoints. The base shutdown force-kills its engine and
logs EngineDeadError after all predictions and completion receipts are saved;
the recorded retirement verifies process exit. Neither pass has a score failure.
Final live inspection shows zero GPU memory used and zero uncorrected volatile
ECC errors. Both serving processes are retired; the user-reserved Pod remains
running, with its weights, artifacts and persistent caches preserved.

All predictions, batches, raw logits, source snapshots, binding/parity/native
receipts, comparison/bootstrap and logs are collected locally under
`results/apps_injection_detection/presence01/` and
`logs/runpod/apps_injection_detection/`. Independent local validation verifies
9,114 unique finite identity-matched predictions per model, exact tokens and
completion/file/contract checksums. Implementation checks: 12 new focused tests,
17 including the reused census contract tests, and repository-wide Ruff pass.
