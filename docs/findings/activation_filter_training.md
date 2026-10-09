# Activation-ranked training removal

The user requested a deliberately exploratory activation-filtering ablation,
selecting **20% of harmless originals**. The activation-ranked run improves ID
ranking but worsens APPS injection discrimination. Matched random removal also
worsens APPS, although less severely. This is a failed injection-alarm mitigation,
with an ID/APPS tradeoff and unresolved single-seed SDPA variance. No model is
promoted.

## Frozen intervention and controls

The [experiment contract](../../experiments/activation_filter_training/README.md)
ranks all 4,344 harmless originals by trained-minus-base layer-20 projection onto
the existing APPS unit direction. It removes 869 original identities and their
augmented replacements, teacher rows and ledger entries together. Exact-content
closure adds no records. Neither Kimi targets nor Qwen firewall scores selects
examples. The cutoff is **+1.079087615**.

Both filtered conditions retain **7,819 records**, including all 4,344 harmful
originals, 3,475 harmless originals and 3,113 augmented replacements. Removal
counts by source are o46 362, s45 295, ctrl 143, STRIDE 34 and Gloom 35. The control
matches source, label, augmentation status and source/label length quartiles,
using fixed SHA256(seed:index) ranking. It overlaps the activation removal in
394 records; a disjoint control is not feasible for heavily removed strata.
Retained bytes, order, targets and augmentation remain unchanged.

Retained training tokens are **67,537,527** for activation removal and
**67,601,478** for the matched control, versus 83,969,213 unfiltered. Both use
245 optimizer updates for one epoch, rather than the baseline's 272. Maximum
lengths are 29,391 and 29,325 respectively, with no truncation. The token totals
differ by less than 0.1%, although exact lengths and training sequences differ.

As a descriptive consequence, activation removal deletes 345 of the 346
harmless originals with Kimi target >=0.5, versus 160 for the control. Their
removed mean targets are 0.4270 and 0.2212; removed mean alignment changes are
+3.5538 and +1.0268. These are post-selection descriptions, not additional
selection criteria or independent misconduct judgments.

Both conditions use the preserved seed-0 BF16/SDPA/FlashQLA recipe and zero-B
rank-128 FP32 initializer. The training runtime is restored from the preserved
archive without dependency changes: Python 3.12.3, Torch 2.11.0+cu130,
Transformers 5.14.1, PEFT 0.19.1 and cuDNN 91900 on the same NC2 B200 UUID.
The recipe is a matched exception to the current FP4/FA4 training default; BF16
serving on vLLM 0.31/Torch 2.13 is a matched exception to optimized serving.

Primary APPS results exclude every variant of the 64 direction-fitting tasks:
**591 held-out tasks / 8,218 rows**. Curation indirectly uses that fitting
partition, which is explicitly disclosed. Registered full-population summaries
remain diagnostic. The cached unfiltered SDPA model is recomputed on the same
holdout. Each model uses the unchanged clean-only, at-most-2%-FPR threshold rule;
the numeric cutoff varies by model. ID uses the frozen 3,012 CoT-removed records.

## Results

| Condition | ID macro AUROC | ID macro pAUROC20 | APPS injected-honest FPR | APPS injected-backdoor recall | APPS matched AUROC | APPS matched pAUROC20 |
|---|---:|---:|---:|---:|---:|---:|
| Unfiltered SDPA | 0.9567 | 0.8600 | 51.10% | 79.13% | 0.8036 | 0.5129 |
| Activation removal | 0.9733 | 0.9151 | 100.00% | 100.00% | 0.6016 | 0.1569 |
| Matched random removal | 0.9608 | 0.8725 | 91.54% | 97.66% | 0.7025 | 0.2540 |

APPS injection columns average the three frozen transfer templates. pAUROC20
uses the project's raw partial-area convention. The activation run's threshold
is 0.08510, versus 0.16452 unfiltered; clean FPR is 1.86% versus 1.69%. All three
injection templates reach 100% honest and backdoor alarms. Mean injected-honest
scores rise from 0.2530/0.1864/0.1130 to 0.4678/0.4428/0.3783. Matched ranking
also worsens, so the regression is not explained only by the operating cutoff.
Vanilla APPS AUROC stays near baseline (0.9216 versus 0.9200); pAUROC20 is
0.7422 versus 0.7221, and backdoor recall 52.83% versus 50.09%.

Relative to random removal, activation removal adds **8.46 percentage points**
of injected-honest FPR, subtracts **0.1009 matched AUROC / 0.0972 pAUROC20**,
and gains 2.34 points of injected-backdoor recall. Both removals worsen injection
discrimination, despite near-matched exposure. This argues against treating
large alignment changes as a useful deletion rule here; it does not establish
individual-example training influence or exclude run variance. The source/
length/control overlap and changed target mass remain relevant qualifications.

ID pooled AUROC/pAUROC20 are 0.9511/0.8377 unfiltered, 0.9663/0.8830 activation,
and 0.9544/0.8466 random. At the frozen 0.5 threshold, pooled FPR/recall change
from 4.77%/81.22% to 7.26%/88.99% and 6.62%/84.95% respectively. Pooled Brier
scores improve from 0.09620 to 0.07167 and 0.08847. Source pAUROC20 for Gloom /
STRIDE is 0.7854/0.9345, 0.8554/0.9748 and 0.8055/0.9395 respectively. ID has
89/88/95 unique scores among 3,012 rows; score ties remain substantial. Complete
calibration, source metrics and APPS template/vanilla diagnostics are preserved
in the campaign summaries and primary holdout receipt.

## Receipts and qualifications

Selection and materialization receipts are in
`results/activation_filter_training/{selection,materialization_audit}.json`.
Campaign outputs use `results/activation-filter-{ranked,matched}20-sdpa01/`;
the held-out comparison is `results/activation_filter_training/heldout_apps.json`.
Logs use `logs/runpod/activation-filter-*/` and the sequential driver log
`logs/runpod/activation_filter_training/driver01.log`. FP32 masters remain on the
persistent volume; merged checkpoints remain remote under `/tmp/gleipnir-merged/`.

Activation training completes all 245 updates in 2,643 seconds with finite
losses/gradients and verified coverage. Its FP32 master SHA256 is
`2921de6d02dc1e1c217177737cd1333d744ab5698aedd007a461b903aeb5167c`.
BF16 merge parity passes at MAE 0.001508, correlation 0.999921; BF16 serving
parity passes at MAE 0.000596, correlation 0.999995, with native audits passed.
All 12,126 registered ID/APPS rows are scored per candidate. Control training
completes in 2,633 seconds; its FP32 master SHA256 is
`c8a0832db2f7872cc79eb95cf05d52141c7b9788ba626a2e2e0343907c6aae63`.
Merge parity passes at MAE 0.003396 / correlation 0.999780 and serving parity
at MAE 0.003233 / correlation 0.999771, with native audits passed. Both runners
complete every frozen stage without a diagnostic exception.

`completion_audit.json` independently verifies collected prediction hashes,
population sizes, finite bounded scores, completion stages and available bound
artifacts. Recomputing the held-out comparison locally reproduces the remote
summary byte-for-byte. Frozen executed sources and small receipts are collected;
adapter and merged weights remain remote. The matched-control BF16 server stays
warm on port 8010. The container overlay has 3.4 GiB free after both merges;
another merge requires additional space. Existing merges and caches are preserved.

Two focused selection/closure/control tests and scoped Ruff pass. No plotting
tests are added. Single-seed results cannot establish stable causal influence of
individual records. Filtering changes label/target mass, ordering, exposure and
schedule, even with a matched control; SDPA execution variance remains unresolved.
The experiment freezes the fraction and selection before scoring and tunes no
threshold, prompt, checkpoint or axis on ID/APPS.
