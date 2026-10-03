# Training prompt-injection census

Hypothesis: literal keyword searches and one STRIDE objective undercount
injection exposure in the frozen 8,688-row monitoring training mixture.
Intervention: score every complete student-visible trajectory with unadapted
`Qwen/Qwen3.5-9B` (no Gleipnir adapter), pinned to the existing backbone
revision, using a separate injection-presence rubric and normalized next-token
`0`/`1` logprobs in one persistent BF16, text-only vLLM engine on one B200.
This is a dataset audit, not training or monitor-performance evaluation.

Presence includes concrete malicious instruction/authority-spoofing payloads
aimed at an LLM, whether constructed, quoted as a test, submitted, successful,
failed or resisted. An agent attacking another model and an agent encountering
an attack both count. Exclude mere mentions/aggregate results, ordinary task
instructions and SQL/shell/template injection without LLM instruction redirection.
The prompt does not ask whether the agent is harmful. Source labels and source
scenario metadata are withheld from the classifier and joined only for reports.
Keep the exact training trajectory, including any reasoning already in it.

Baselines: the preceding heuristic audits and four explicitly assigned STRIDE
scenarios, neither an exhaustive truth set. Fixed synthetic positives/negatives
check the scoring interface and expose semantic weaknesses; real previously
reviewed examples are diagnostic anchors. Do not tune the prompt, thresholds or
model against the full-pass output. There is no held-out model-selection set:
all 8,688 training rows are the target census. No OOD input is read. Synthetic
controls are diagnostics, not calibrated prevalence ground truth.

Stop on input/prompt/checkpoint/cache identity drift, truncated inputs,
missing/nonfinite token logprobs, duplicate/unknown IDs, incomplete outputs,
OOM or backend failure. A semantic control miss is reported, not hidden or used
to change the frozen rubric. Completion requires 8,688 unique finite scores,
exact token-count coverage and local artifact collection. Preserve raw returned
logprobs, prompt/checkpoint hashes, token usage, timestamps and runtime versions.
Resume only missing rows under an identical contract. A failed startup receipt
remains separate from subsequent successful attempts.

Report source/behavior-label cross-tabs at fixed probability thresholds
0.1, 0.5 and 0.9, score distributions, ties and ranked candidates. These are
model-flagged counts, not verified prevalence or calibrated probabilities.
Review the high-ranked candidates and the real anchors with visible evidence;
retain ambiguous cases. Calibration and AUROC against injection truth are not
available without an independently annotated truth set. Training behavior
labels must not be used as injection ground truth.

```bash
python -m experiments.training_injection_census.run
python -m experiments.training_injection_census.run --summarize-only
```

Input: `data/tool_trajectory_monitoring/distillation_scaling/student_rows.jsonl`.
Output: `results/training_injection_census/qwen35_9b_v1/`.
Logs: `logs/runpod/training_injection_census/`. The launch script uses the
preserved locked environment. Monitor startup and actual progress in the agent
turn; a process watchdog does not replace agent heartbeats. The current session
has no in-chat scheduling tool, so do not promise follow-ups after yielding.

Runpod B200 `3422tdq7tfog52` was explicitly authorized by the user for this
pass, at a live quoted $6.79/hour in US-NC-2, with existing network volume
`ixbh81vf9c` mounted at `/workspace`. The previous Pod could not resume because
its host lacked a free GPU and remains EXITED. Lifecycle uses the Runpod MCP;
SSH/rsync uses `scripts/runpod_cloud.py --pod-file .runpod/injection-audit.json`.
Do not delete the preserved volume or change the existing training artifacts.

Completed 2026-10-03: all 8,688 unique finite scores, no truncation, 8/8 fixed
controls correct, and passing local collection checks. Counts at thresholds
0.1 / 0.5 / 0.9 are 3,396 / 1,154 / 202. Reviewed code-only false positives
prevent interpreting them as verified injection prevalence. See the
[finding](../../docs/findings/training_injection_census.md) for source/label
cross-tabs, scores, review evidence, ties and provenance receipts.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B),
[vLLM sampling parameters](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/).
