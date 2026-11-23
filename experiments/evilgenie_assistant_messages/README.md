# EvilGenie with the agent's assistant messages

User request, 2026-11-23: does keeping the agents' assistant messages help an
action-only monitor judge reward hacking? Raised by the `arc191_c` reading
(`experiments/adversarial_evilgenie/README.md`), where the messages state the
intent that the actions hide. Exploratory; the report (Section 6.2, F.3)
currently leaves this to future work.

## Protocol (frozen 2026-11-23, before any score on these inputs)

- **Hypothesis.** With assistant messages, Gleipnir 27B ranks the reward hacks
  better: higher pAUROC@20 on the 549 rows that have messages, and so on all
  rows. Ablation of one input, not a comparison with other monitors.
- **Rows.** The 2,127 runs of `experiments/evilgenie_released_stack`, same
  labels and ids. `build.py` re-renders each run from the raw transcripts and
  stops unless it reproduces the baseline state byte for byte; it then adds
  the text parts of each assistant message, placed as in the training data
  (`[ASSISTANT]`, the text, a blank line, `[TOOL CALLS]`). Reasoning parts
  stay dropped; text in the text field is kept as written, including the
  reasoning-like text of some models. Runs without assistant text keep their
  baseline state exactly. States: `data/evilgenie_assistant_messages/states.jsonl`
  (SHA-256 `7afa3280…`); 549 rows change (33 reward hacks, 510 legitimate
  runs, 6 controls).
- **Monitors and serving.** As `experiments/evilgenie_released_stack`: image
  `ghcr.io/jazhyc/gleipnir-serve:1.2` on one B200, default serving mode,
  released instruction per model, chat template checked against the recorded
  OOD workload, Gigatoken counts, batches of 128 at concurrency 128 (that
  experiment's `pod.py` with `--config` pointing here). Both models.
- **Baseline.** The scores of `experiments/evilgenie_released_stack` on the
  same rows (same image, hardware type and batching).
- **Metrics.** Per model, AUROC and pAUROC@20 without and with messages, and
  their paired difference with a 95% bootstrap interval: on all rows and on
  the rows with messages (resampling problems), and on `arc191_c` (resampling
  runs). Flag counts at each model's released-stack 1% OOD threshold, which
  was calibrated without messages and is reported only as a diagnostic.
- **Selection.** None: released checkpoints, one pass each.
- **Stop conditions.** A failed smoke test or rendering check, a server
  rejection, or a non-finite score.
- **Compute.** User authorization 2026-11-23: one B200 Pod from image 1.2,
  terminated after collection.

```bash
PYTHONPATH=src:. python -m experiments.evilgenie_assistant_messages.build
PYTHONPATH=src:. python -m experiments.evilgenie_released_stack.pod check-rows \
    --config experiments/evilgenie_assistant_messages/config.json
# on the Pod, after pushing both experiments and the data
experiments/evilgenie_assistant_messages/pod_run.sh 4b 27b
# locally, after pulling results/ and logs/
PYTHONPATH=src:. python -m experiments.evilgenie_assistant_messages.analyze
```
