# SleightBench transfer of the NTP activation probes

Score the frozen NTP probes from the
[probe study](../probe_generalization/README.md) (run `full01`) once on
[SleightBench](../sleightbench_eval/README.md), stripped view, and compare
them with the released Gleipnir monitors on the same transcripts (run `sb01`,
[finding](../../docs/findings/sleightbench_released_monitors.md)).

**Single pass (user rule for SleightBench).** This README, `manifest.json`
and the code are frozen and committed before any probe scores a SleightBench
row. Each probe is scored once and reported whatever the result. Nothing is
retrained, reselected, recalibrated or rerun after a score exists.
SleightBench is evaluation-only: it never enters training, tuning or
selection. The decrypted text rows are deleted from the remote host after the
run; the FP8 activation store (no text) is kept on the persistent filesystem.

## Hypotheses (user, 2026-10-26)

1. **Transfer degrades.** The NTP backbone sees no instruction, so the
   monitoring specification cannot be changed by prompting. The probes inherit
   the training taxonomy, which labels honest-looking mistakes benign, while
   SleightBench builds attacks to look like them. Prediction: both selected
   probes (hard and soft) have a stripped-view seed-median AUROC below the
   Gleipnir 4B v0.4 (0.657) and catch at most its 5% at 1% FPR. These are
   not significance tests: they compare 40 attack units.
2. **Long context hurts.** Probe training rows are at most 143K characters
   (about 40K tokens). Six stripped transcripts are 160K–194K tokens: both
   sides of `pip-freeze-typosquat` and `gitignore-env-omission`, and the
   attacks of `gzip-concat-stego` and `tar-gzip-stego`, whose benign twins are
   short. Every other transcript is at most 20K tokens. The
   [attention-map study](../probe_attention_maps/README.md) found that both
   probes drift negative over long tool output. Prediction: those four attack
   units are missed, and the seed-median score of attacks falls with
   log length. Reported descriptively: the short-context metrics (units
   at most 32,768 tokens), the four units' scores against their benign
   twins, and score–log-length Spearman correlations for attacks and benigns
   separately.
3. **Soft versus hard** (descriptive). The soft probes were ahead on every
   OOD family. Report whether that holds here; there is no bar.

## Design (frozen)

- **Rows.** `prepare.py` writes `data/probe_generalization/sleightbench/`:
  - `stripped`: 86 transcripts (43 attacks, 43 benign twins), built with
    the `sleightbench_eval` stripped view and `benchmark.body`. This is the
    exact trajectory body the released monitors read inside
    `<agent_trajectory>`.
  - `parity_id`: 16 probe-study ID rows (the 8 longest plus 8 sampled with
    seed 0).

  `manifest.json` freezes hashes, labels and lengths. The bodies are JSON
  message lines, a format absent from the probes' training sources; the
  released monitors read the same text.
- **Extraction.** Run `probe_generalization/extract.py` unchanged
  (`--population sleightbench`): NTP backbone, Qwen3.5-4B at the pinned
  revision, BF16 Transformers forward with the pinned fast kernels, residual
  after block 15, FP8 store. One forward pass per row at full length, with no
  truncation or chunking. Pooling drops the first token (the attention sink),
  as in training.
- **Probes.** The frozen NTP selections in `full01/selection_ntp.json` for
  both objectives: each objective's selected candidate (hard attention at
  epoch 8, soft rolling_w256 at epoch 8) and each family's ID-best candidate,
  all 5 seeds. That is 16 (objective, architecture, epoch) picks. Primary:
  the two selected probes. The family picks are descriptive; their OOD
  numbers are known, so choosing among them would be OOD-informed.
- **Parity gate.** `score.py score` first scores `parity_id` and compares it
  with the logits that `full01` stored for the same rows. Every picked probe
  must agree within 0.05 of its ID logit standard deviation. On failure, the
  run stops before any SleightBench row is scored.
- **Metrics.** `sleightbench_eval.run.view_report` on raw logits, with
  `config.json`'s metrics: 40 attack units, multi-session parts merged by max,
  catch at 1% FPR with the threshold at the highest benign score, AUROC,
  pAUROC@20, pair wins, 95% scenario bootstrap, by-category results and
  threshold robustness. The long-context split uses NTP token counts. For each
  pick, report the seed median of each headline metric, the best-ID seed's
  full report and, as a descriptive result, the 5-seed mean-logit ensemble.
- **Baselines.** The `sb01` stripped-view numbers, used as recorded:
  Gleipnir 4B v0.4 has catch 5%, AUROC 0.657, pAUROC@20 0.212 and pair wins
  60%; 27B v0.2 has 15%, 0.706, 0.317 and 70%. The paper's frontier rows are
  in the finding.

## Compute and run

One Lambda H100 in us-west-3 with the full project bootstrap (fast kernels as
in [infrastructure](../../docs/infrastructure.md#lambda-cloud)) and the
`gleipnir-probes` filesystem attached. At the user's request (2026-10-26),
`run_remote.sh` copies the activation store to
`stores/sleightbench/ntp` there for later reuse. Push the commit, the two row files,
`selection_ntp.json`, and from `ntp-fp8/` the `config.json`, `summary.json`
and `probes_e{2,4,8,16}.pt` and `logits_e{2,4,8,16}_fp8.json`. Then:

```bash
bash experiments/probe_sleightbench/run_remote.sh          # remote GPU
python -m experiments.probe_sleightbench.score report --name sbprobe01  # local CPU
```

Pull `results/probe_sleightbench/sbprobe01/{parity,logits}.json` and
`logs/lambda/probe_sleightbench/sbprobe01/`, then delete the text rows from
the host.

Status: complete (run `sbprobe01`, 2026-10-26; one Lambda H100 PCIe,
about an hour, terminated around 15:12 UTC after its 8 artifacts were SHA-256 verified).
The parity gate passed with drift 0.0000 ID-SD. Results are in
[the finding](../../docs/findings/sleightbench_ntp_probes.md).
