# Augmented monitor with current FP4 training

The original injection augmentation was replayed on the existing NC2 B200 with
the current native FP4 MLP/BF16 FlashQLA/FA4 training framework. Training completed
all 272 updates, one epoch and 83,969,213 tokens. The final adapter is fixed;
no checkpoint or hyperparameter selection uses ID/APPS results.

## Population and training receipt

The population is unchanged: 8,688 underlying trajectories, 3,475 injected views
and 5,213 clean examples; the original eight authored templates, insertion ledger,
role/position assignments, behavior labels and clean Kimi soft targets. Teacher
target copying retains the original behavior-invariance assumption. Training
uses seed 0, rank 128/alpha 256, soft BCE, AdamW 5e-5, linear decay/3% warmup,
logical batch 32 and the selected 16,384-token packing budget. References and
inference use separate pinned environments; training retains Torch 2.11.

The Trainer reports 2,516.53 seconds (41.94 minutes), mean loss 0.241604 and
steady optimizer steps averaging 8.726 seconds after excluding two warmup steps.
Peak allocated/reserved CUDA memory is 151.36/154.27 GiB. Both exported adapters
retain all 256 FP32 tensors. The original and new explicit-profile completion
validators pass; logical-row coverage and token sums were independently checked.

Initializer file SHA256 `5da72480668385805032c5039e5c4a29e7f6257762185da79d13a5c70443f3b0`
is distinct from loaded tensor fingerprint
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
Final tensor fingerprint is
`4eae41a7fa473bdf2f96344c536a92dc4e254c320cc5c33713d9bd5dc20b3f47`;
final causal adapter file SHA256 is
`8e65826845d589a6879e052205653512b4106f3529befda583a03329ac6f34ff`.
The rebased FP32 inference adapter has SHA256
`c6d8437e7f1accc9ec1dc4972bc82d525ff38a03e8723ac2545c777d79dbcccb`.

## Adapter-specific serving gates

Twenty balanced shortest training-source canaries use the unchanged regular
instruction and Prediction boundary. The merged BF16 checkpoint incorporates
128 projection updates with FP32 accumulation and one BF16 cast, taking 42.04
seconds. It is retained at `/tmp/gleipnir-merged/fp4-augmented-bf16`.

| Comparison | Score MAE | Correlation | Maximum difference | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| Merged BF16 versus FP32 master | 0.001982 | 0.999861 | 0.031209 | Pass |
| Optimized versus FP32 master | 0.022172 | 0.996288 | 0.091250 | Fail |
| Optimized versus merged BF16 | 0.020666 | 0.996637 | 0.091250 | Fail |

The predeclared new-adapter limits are MAE <=0.020, correlation >=0.99 and
nonzero adapter effect, with finite probabilities. Optimized scores are finite,
correlation and effect pass, and actual dispatch audits verify 16 FP8 attention
projection calls. Both optimized MAEs exceed the limit. The original evaluation
therefore stopped before any held-out requests and retired its scorer. Failed
gates remain failed; no earlier finite-quality acceptance was inherited.
The five largest absolute score shifts occur at canary indices 1, 11, 19, 10
and 17. These score checks cannot establish ID/APPS ranking or quality.

The user subsequently explicitly authorized optimized ID/APPS scoring as a
failed-parity diagnostic. A separate contract reuses the completed FP32 adapters,
passed BF16 merge/reference and frozen ordered workloads. It permits excessive
MAE while retaining finite, correlation, effect, model/runtime identity and full
coverage guards. It performs no optimizer updates and does not promote the model.
All 12,126 diagnostic cases completed: 3,012 canonical CoT-removed ID examples,
5,184 APPS benchmark cases and 3,930 honest controls. On the diagnostic restart,
canary MAE is 0.017989 against the master (correlation 0.997856) and 0.016483
against merged BF16 (correlation 0.998456), passing the unchanged gate. Preserve
the first failure and diagnostic scope. Weights and selected serving recipe are
unchanged; startup/request-batch sensitivity was also seen in the regular APPS
canary. This restart is not a quality promotion.

## Held-out diagnostic results

| Metric | Current regular FP4/optimized control | Augmented FP4/optimized diagnostic | Historical augmented BF16 |
| --- | ---: | ---: | ---: |
| ID source-macro AUROC | 0.957476 | 0.928250 | 0.962342 |
| ID source-macro raw pAUROC@20 | 0.865432 | 0.796611 | 0.872983 |
| Gloom pAUROC@20 | 0.785899 | 0.671511 | 0.809069 |
| STRIDE pAUROC@20 | 0.944966 | 0.921710 | 0.936897 |
| ID macro Brier | 0.106276 | 0.129423 | 0.084668 |
| APPS mean injected-honest FPR | 74.05% | 31.30% | 44.43% |
| APPS mean injected-backdoor TPR | 84.85% | 64.45% | 87.64% |
| APPS mean matched injection AUROC | 0.613024 | 0.760377 | 0.859471 |
| APPS vanilla AUROC | 0.918230 | 0.906117 | 0.925572 |

The ID macro pAUROC regression is 6.88 percentage points against the current
regular control, concentrated on Gloom (11.44 points). Pooled AUROC/pAUROC are
0.918195/0.757487; source AUROCs are 0.874672 Gloom and 0.981828 STRIDE. There
are 90 distinct scores; paired mean absolute score change versus the regular
control is 0.078213, with 254 fixed-0.5 decision flips. Calibration also worsens.
The regular ID control was scored on the EU host; the APPS control used this
NC2 host. Historical BF16 comparisons combine training, serving and host
changes. These controls do not isolate an FP4 effect or its interaction with
augmentation. Short training-source parity does not establish held-out quality.

APPS uses each model's clean-only strict `score > threshold` operating point
allowing at most 2% clean FPR. The diagnostic threshold is 0.245085, with 9/655
clean alarms (1.37%); the regular control has 12/655 (1.83%), with ties making
the achieved rates differ. Injection results remain heterogeneous:

| Template | Honest FPR, regular → augmented | Backdoor TPR, regular → augmented | Augmented matched AUROC / pAUROC@20 |
| --- | ---: | ---: | ---: |
| GPT-4 | 70.38% → 68.55% | 84.85% → 90.57% | 0.682491 / 0.212047 |
| GPT-4.1 | 100% → 21.68% | 100% → 65.69% | 0.792933 / 0.483063 |
| GPT-4o | 51.76% → 3.66% | 69.71% → 37.09% | 0.805707 / 0.502426 |

Mean matched raw pAUROC@20 improves from 0.186397 to 0.399178. Injection-minus-
neutral honest mean score changes are +0.350940, +0.173682 and +0.073027; GPT-4
still causes many alarms and a larger paired score increase than the regular
control (+0.173656). Vanilla raw pAUROC@20 regresses from 0.744354 to 0.682168
and vanilla-backdoor TPR from 51.93% to 43.43%. Thus augmentation improves
matched injection separation and reduces alarms for two templates, while
losing harmful-action recall and clean ID ranking. This replica is not an
overall quality improvement and was not promoted. No adaptive robustness claim
follows from this one-seed, limited-template, off-policy diagnostic.

ID, APPS benchmark and honest controls took 251.29, 63.58 and 52.53 seconds of
HTTP scoring, respectively, excluding startup/audits; token totals are
33,750,959, 8,207,435 and 7,132,518. Ordered identities, labels/metadata, rendered
prompt hashes, exact server token counts, finite logits, margins and sigmoid
scores were verified for every row. Independent ROC integration reproduces ID
raw pAUROC and AUROC; independent threshold/FPR/TPR/matched-AUROC calculations
reproduce the APPS summaries. Both original and shared summaries agree. All
583 collected remote/local artifact checksums match, including master adapters,
final optimizer/checkpoint state, predictions and 310 diagnostic source snapshots.

## Preparation failures and provenance

Before training, three failed attempts performed zero optimizer updates. The
first lacked two cached ID artifacts, restored from their exact local copies.
The second encountered a historical summary differing only in absolute file
paths; its remote original and a canonical copy were preserved. The third
mistakenly compared an initializer file checksum with a loaded tensor digest;
the original initializer was intact. Separate failed receipts retain each attempt.
The successful campaign independently reaudited augmentation and froze its
inputs and 96 executed source files. Those source snapshots, both FP32 weight
checksums and the failed optimized parity were independently verified locally.
Final checkpoint/optimizer state, adapters, references, native audits and logs
were collected without changing capacity. Intentional scorer retirement produced
an EngineDeadError/one semaphore cleanup warning after the parity failure;
this is not the cause of the failed gate.

Primary artifacts: `results/b200_augmented_training/`,
`data/b200_augmented_training/`, `logs/runpod/b200_augmented_training/`.
The diagnostic uses `results/b200-augmented-training-diagnostic01/` and a separate
data/source binding. Its configuration and entrypoint are
[diagnostic_config.yaml](../../experiments/b200_augmented_training/diagnostic_config.yaml)
and [diagnostic.py](../../experiments/b200_augmented_training/diagnostic.py).
The [experiment README](../../experiments/b200_augmented_training/README.md)
retains the frozen original contract.
