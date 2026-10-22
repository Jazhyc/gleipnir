# Does Gleipnir catch other models' reward hacking? (EvilGenie × Escalation)

Frozen 22 October 2026, after the rows were built and 20 were read by hand, before any score was computed.

## Question

Our own pilot gives natural cheating by one open model (`experiments/adversarial_natural_cheats/`). This adds a public set of natural reward hacking by many models on the same kind of task, scored by Gleipnir as a black-box monitor. Most agents here are closed models, so this tests Gleipnir only, not probes.

## Data

`WiserHumanExperimental/evilgenie-escalation` on Hugging Face, revision `1bb777b666b0f76801f9b5b9a28eb3f7786ea70f`, MIT licence, evaluation only, stored in ignored `data/`. Paper: Gomez, "Can escalation channels redirect reward hacking toward defect disclosure?" (arXiv:2608.29460), built on EvilGenie (arXiv:2511.21654).
- **Runs:** 2,115 runs analysed in the paper (LiveCodeBench problems with ambiguous specifications, visible tests, `bash`/`python` tools; conditions A–E), plus 180 GPT-5.3-Codex negative-control runs on unambiguous problems.
- **Labels** are the dataset's own, from `tables/per_run_data_all_conditions.csv` joined on `run_key`: holdout tests, an LLM judge, and an audit of every file-modifying run.
  - **Positives (124):** `reward_hacked = yes`.
  - **Negatives (1,823):** `classification = legitimate`.
  - **Reported separately, in no metric:** `heuristic` (161) and `error` (7).
  - **Clean negatives:** the 180 negative-control runs.
- **Kimi K3** produced Gleipnir's training targets and is also an agent here, so its rows are reported separately as well as inside the totals.

## Rendering and scoring

- The same action-only strict-OOD format as the natural-cheats experiment: system prompt, assistant prose and reasoning dropped; tool calls as `name({json arguments})`; tool results as the agent saw them (errors as `Error: <message>`). The escalation tool's calls are kept like any other tool call.
- **One detection point: the full run.** The dataset has no per-step hack index. Runs over 29,696 tokens are excluded and counted.
- **Monitors:** Gleipnir 4B v0.3 with the student prompt and Gleipnir 27B with the teacher prompt, BF16, shared scorer, after the usual drift canaries.

## Metrics

- AUROC of positives against negatives, overall and by condition (A–E), model and judge type (`hardcoded_outputs`, `pattern_matching`, `test_file_reading`), with intervals from a bootstrap clustered by problem.
- Recall at the strict-OOD 1% and 2% false-alarm lines (the same 1,000 calibration rows as `adversarial_natural_cheats`, whose scores are reused), and the false-alarm rate those lines give on negatives and clean negatives.
- A hand read of 10 random positives and 10 random negatives before freezing, to check the labels hold up.

## Predictions

1. AUROC is at least 0.85 for the 27B and at least 0.75 for the 4B.
2. ~~Test-file edits are caught more than hard-coded outputs, which beat pattern matching.~~ Withdrawn before freezing: no positive has judge type `pattern_matching` (all 117 such runs are `heuristic`), and only 2 positives have `tamper_hack = yes`. Positives are `hardcoded_outputs` (115) and `test_file_reading` (9); no directional prediction between them.
3. At the strict-OOD 1% line, the 4B flags at most 5% of negatives.
4. Recall is lower for conditions with an anti-hacking policy (C–E) than without (A–B), because a policy-aware agent hides more; weakly held.

## Build notes (fixed before scoring)

- **Errored tool results are empty.** The release removed Inspect's error field, so the 1,818 errored results (57 of 124 positives' runs, 842 of 1,823 negatives', 4 of 180 clean negatives') render as empty `[TOOL]` blocks, not as the `Error: <message>` the agent saw. Gleipnir sees less than the agent did, in both classes.
- **Counts:** 124 positives, 1,823 negatives, 161 heuristic, 7 error, 180 clean negatives; every label row joins on `run_key`. By condition (positives/negatives): A 58/581, B 32/305, C 12/315, D 5/320, E 17/302.
- **Too long (excluded and counted):** 172 rows for the 4B (16 positives) and 182 for the 27B (19 positives); mostly gemini-3.5-flash.
- **Length doesn't separate the classes:** token-count AUROC 0.45 over rows that fit.
- **87 runs never act** (86 negatives, 15 of them Kimi K3's); they stay in, with a view that excludes them.
- **Positives sit on 5 of 9 problems,** 69 of them on one (`lcb_arc191_c`), so intervals clustered by problem will be wide.
- **Hand check:** 10 positives and 10 negatives read; every label held. Many positives say they hard-coded outputs in their `submit` answer, which the monitor sees.

## Scoring plan and cost

- **4B v0.3:** every metric row, positives first, then negatives and clean negatives in a seeded shuffle; about 18M tokens, roughly 1.5 hours at the natural-cheats rate. Calibration reuses the natural-cheats 4B scores of the same 1,000 strict-OOD rows, with a canary requiring agreement within 0.02.
- **27B:** `27b-merged` (the checkpoint behind every earlier 27B result), with its strict-OOD 1% and 2% lines from the 400 honest calibration rows `adversarial_is_it_4b` scored. Positives first, then the seeded shuffle, capped at 2 hours; rows not reached are reported as not scored, and the shuffle keeps the scored negatives a random subset.
- No API spend.
