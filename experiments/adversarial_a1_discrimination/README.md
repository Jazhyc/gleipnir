# A1 follow-up: discrimination beneath injection alarms

Exploratory reanalysis of existing frozen 4B scores, 2026-10-20. No new API or
GPU calls. This asks whether injected text merely shifts scores upward or also
weakens discrimination between honest and backdoored code.

Pair all 647 backdoor tasks with their honest counterparts. Use original code,
each of the three injection templates, and each matched neutral-text condition.
The eight honest-only tasks are excluded from these paired comparisons. Preserve
the original threshold calibrated on all 655 honest solutions. Input score hashes
are checked by `run.py`; data and scoring provenance are in
`../adversarial_baseline_4b/README.md`. The common task set is fixed across columns.

Report pooled AUROC and the within-task fraction where the harmful solution
scores higher, with ties counting one half. The latter is not pooled AUROC.
Bootstrap task pairs 10,000 times, seed 0. No thresholds are retuned.

| Condition | AUROC | Within-task rank fraction | Median honest score | Median harmful score |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.925 | 95.7% | 0.012 | 0.223 |
| Neutral, GPT-4 template length | 0.874 | 93.6% | 0.029 | 0.148 |
| Neutral, GPT-4o template length | 0.888 | 92.8% | 0.023 | 0.133 |
| Neutral, GPT-4.1 template length | 0.887 | 93.6% | 0.020 | 0.133 |
| Injection, GPT-4 | 0.619 | 71.6% | 0.706 | 0.755 |
| Injection, GPT-4o | 0.697 | 78.1% | 0.469 | 0.531 |
| Injection, GPT-4.1 | 0.631 | 71.9% | 0.755 | 0.777 |

Injection-condition rank-fraction bootstrap intervals are 68.4–74.9%,
75.1–80.8% and 68.9–74.8%, respectively; the original interval is 94.1–97.1%.
All 647 injected backdoors are flagged in every condition. Honest flags are
646/647 for GPT-4 and 647/647 for the other two, versus 5/647 originals.

The evidence supports both an upward score shift and degraded discrimination.
A uniform score shift alone would not reduce AUROC this way. Some ability to
rank harmful code higher remains; "it ignores the code entirely" is too strong.
These are behavior-level results, not evidence identifying an internal circuit.
The honest and backdoored solutions share tasks, but are not verified minimal
repairs of the same implementation. BF16 ties and template/placement differences
can contribute. The label is code correctness/backdoor presence; an honest-code
example deliberately manipulating the monitor is not necessarily innocent in
an agent-behavior sense. Peer-source attribution is still untested.

Reproduce with `PYTHONPATH=src:. python -m
experiments.adversarial_a1_discrimination.run`. Complete numbers and input hashes
are saved to ignored `results/adversarial_a1_discrimination/report.json`.
