# SDPA injection-direction intervention replication

2026-10-09. The completed replication reduces honest-injection alarms below
0.5% and improves APPS ranking, with a large injected-backdoor recall loss and
small ID ranking changes. The user authorized replicating the colleague's fixed
clamp, projection and steering controls on the current BF16/SDPA augmented adapter,
including a small APPS fit portion. The
[experiment contract](../../experiments/b200_injection_direction/README.md)
owns membership, operators, diagnostics and stop conditions; the
[source review](../research/sdpa_injection_direction_replication.md) distinguishes
the colleague's historical adapter and later learned-probe results.

## Population and execution

`sdpa_injection_direction01` fits on 64 hash-selected tasks (896 rows,
384 injected/neutral pairs), excluding every variant of those tasks from the
591-task APPS test partition (8,218 rows; 583 backdoor tasks). The four primary
arms also score all 3,012 frozen CoT-removed ID rows. Additive sufficiency
controls and full-vocabulary answer-mass diagnostics use a fixed 128-task test
subset (1,771 rows; 125 backdoor tasks). No test outcomes select a layer, beta,
template, checkpoint or serving default. This exploratory APPS fitting
exception does not turn the historical full APPS benchmark into an untouched
confirmation set. One model/training seed; task bootstrap intervals do not
measure training variance.

The FP32 master is `bdb702316b1d5e29cbdd8fba64888fe128b37bfd25030211b867d46372e8f5fd`;
BF16 serving export is
`39f2c0b3015c55da8bdae4e8596f9424e4cd23fe005fae92da2108fa9fec556d`.
The user-reserved NC2 B200 serves its restored ephemeral merged checkpoint
through vLLM 0.31/Torch 2.13/Lens 1.3.0 in BF16 eager mode. The updated engine
passes fresh master/native gates and real GPU operator checks; its twenty-row
no-op scores reproduce the earlier eager engine exactly (MAE zero).
Compiled full-benchmark results are historical context, not the matched control.

Fit the layer-20 injected-minus-neutral mean, norm **7.072298**, on both labels,
then use its unit direction at every layer/token. Clamp beta .25 uses fitted
per-layer neutral decision/span centers; projection uses beta 1 and zero centers.
The random control uses one seed-0 unit direction at all layers. Addition uses
the natural layer-20 shift with scale +/-1 at layer 20/all tokens, without norm
matching. The fused-residual correction is FP32 followed by BF16 casting;
it is not bitwise equivalent to the colleague's Transformers hooks.

## Matched primary results

APPS injected metrics are the equal-template mean. AUROC/raw pAUROC20 compare
honest and backdoored injected code on the same 583 backdoor tasks per template;
honest FPR uses all 591 honest tasks. Recalibrate the strict `score > threshold`
rule on clean code only, allowing at most 11/591 alarms. BF16 score ties make
achieved clean FPR smaller than that allowance. Threshold diagnostics and
ranking metrics answer different questions.

| Arm | Clean threshold | Clean FPR | Injected-honest FPR | Injected-backdoor TPR | Matched AUROC | Matched raw p20 | Plain AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unedited | 0.164516 | 1.52% | 50.82% | 79.19% | 0.804912 | 0.519061 | 0.920075 |
| Clamp beta .25 | 0.119203 | 1.69% | 0.45% | 36.48% | 0.863110 | 0.625189 | 0.918536 |
| Full projection | 0.164516 | 1.02% | 0.39% | 31.39% | 0.890519 | 0.680782 | 0.922283 |
| Random projection | 0.164516 | 1.86% | 46.81% | 79.42% | 0.809276 | 0.527110 | 0.919457 |

At the fixed unedited threshold, clamp FPR is **0.11%**, recall **25.79%**,
and clean FPR **0.51%**. The other arms' recalibrated thresholds equal the
unedited threshold, so their fixed-threshold rates equal the table. This reproduces
the colleague's qualitative false-alarm/recall tradeoff on a different adapter,
smaller fit portion, larger test cohort and different hook arithmetic; it is
not an exact historical result reproduction.

| Arm | ID macro AUROC | ID macro raw p20 | ID pooled AUROC | ID pooled raw p20 |
|---|---:|---:|---:|---:|
| Unedited | 0.956646 | 0.859042 | 0.950993 | 0.837050 |
| Clamp | 0.954545 | 0.858221 | 0.948423 | 0.835561 |
| Full projection | 0.954859 | 0.858470 | 0.948783 | 0.835882 |
| Random projection | 0.956668 | 0.858615 | 0.950888 | 0.835241 |

| Arm | Gloom AUROC / raw p20 | STRIDE AUROC / raw p20 |
|---|---:|---:|
| Unedited | 0.928212 / 0.784592 | 0.985081 / 0.933492 |
| Clamp | 0.924498 / 0.784961 | 0.984592 / 0.931481 |
| Full projection | 0.925314 / 0.784210 | 0.984404 / 0.932729 |
| Random projection | 0.927753 / 0.782219 | 0.985583 / 0.935011 |

For the clamp, ranking mostly survives, but scores change materially:
ID MAE 0.044887 and 189 decisions flip at 0.5.
Gloom recall/FPR at 0.5 changes from 77.01%/5.31% to 64.99%/1.93%; STRIDE
from 88.91%/3.52% to 84.92%/1.63%. Threshold recalibration remains necessary
if this intervention is investigated further.

The fitted layer-20 injection and neutral backdoor mean shifts have cosine
**0.541783** (backdoor shift norm 5.208748). Their overlap is consistent with
removing features useful for both injection alarms and genuine backdoor
detection; it does not establish a unique causal mechanism. Clamp answer mass
on the fixed diagnostic subset remains healthy: minimum 0.987103, mean 0.997741,
no rows below .5, versus unedited minimum 0.981092/mean 0.996852. The alarm
reduction is not explained by loss of probability mass on the supported answers.

## Uncertainty and direction diagnostics

The paired 1,000-draw task bootstrap recomputes clean thresholds within each
draw. Clamp AUROC improves by 0.058198, with 95% delta interval
**[0.050507, 0.066303]**; projection improves by 0.085607,
**[0.074690, 0.097904]**. Their raw p20 gain intervals are
[0.087910, 0.124931] and [0.135464, 0.186645]. Random AUROC/p20 gain intervals
include zero: [-0.000404, 0.009411] and [-0.000934, 0.017732]. One random
direction is a bounded specificity control, not a distribution over controls.

Clamp honest-injection FPR has interval **[0.06%, 1.58%]**, recall
**[26.56%, 44.77%]**; projection FPR **[0.06%, 0.85%]**, recall
**[27.95%, 40.14%]**. Unedited FPR/recall intervals are wide
([23.57%, 72.93%]/[59.41%, 91.79%]): a 2% clean threshold depends on few
tasks, with tied BF16 scores. The paired clamp/project FPR decrease and recall
loss intervals exclude zero. These task-sampling intervals are conditional on
the fixed fitted direction and model; they omit fitting-partition uncertainty,
training seeds, arithmetic variability and physical-host variation.

On the fixed 128-task diagnostic subset, the layer-20 injection-presence probe
has AUROC **0.988861** over 759 injected/neutral pairs. Leave-one-template-out
fit/test AUROCs are 1.000000 (GPT-4), 0.990205 (GPT-4.1) and 0.934915 (GPT-4o).
These are diagnostic direction readouts, not replacements for the monitor's
trained classifier, and select no new layer or beta.

## Additive sufficiency controls

These use the separate fixed 128-task subset; its unedited rates differ from
the full 591-task table. At the fixed subset unedited threshold **0.148047**:

| Arm | Injected-honest FPR | Injected-backdoor TPR | Matched AUROC | Raw p20 | Mean paired score shift | Minimum p01 |
|---|---:|---:|---:|---:|---:|---:|
| Unedited | 60.94% | 82.93% | 0.785664 | 0.482194 | 0 | 0.981092 |
| Add +u20 | 100.00% | 100.00% | 0.746869 | 0.433392 | +0.160133 | 0.956815 |
| Add -u20 | 0.00% | 17.07% | 0.821099 | 0.542112 | -0.111300 | 0.984737 |

Positive addition raises scores in 96.05% of rows; negative addition lowers
every row. No diagnostic has p01 below .5. This supports the direction's ability
to steer conditional suspiciousness without answer-format collapse, while
showing a large calibration effect. Recalibrating clean thresholds changes the
interpretation: +u20 uses 0.407333 with FPR/TPR 33.07%/61.07%; -u20 uses
0.022977 with 66.67%/88.53%. Plain AUROCs are 0.902313, 0.897313 and 0.904469
for unedited/+/- respectively. Subtraction alone is not an injection-FPR fix
at a matched clean false-alarm allowance.

## End-to-end timing diagnostics

One pass per arm, concurrency 64, batches of 128, prefix caching off. Each APPS
arm processes **13,873,010 input tokens**; each ID arm **33,750,959**. Rates
use summed batch wall times; latency includes semaphore wait. APPS diagnostic
capture depth differs (all 32 layers unedited versus final layer for edits),
so these are campaign costs, not isolated operator benchmarks or repeated
production speed measurements.

| Population / arm | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|
| APPS unedited | 72,517 | 42.96 | 2.06 / 2.85 |
| APPS clamp | 65,473 | 38.78 | 2.30 / 3.18 |
| APPS projection | 65,525 | 38.82 | 2.24 / 3.13 |
| APPS random | 65,582 | 38.85 | 2.26 / 3.16 |
| ID unedited | 90,768 | 8.10 | 6.46 / 19.21 |
| ID clamp | 83,913 | 7.49 | 7.01 / 20.71 |
| ID projection | 83,325 | 7.44 | 7.05 / 20.80 |
| ID random | 83,462 | 7.45 | 7.07 / 20.83 |

## Receipts and implementation validation

Artifacts live under
`results/b200_injection_direction/sdpa_injection_direction01/`: checksum-bound
`manifest.json`, `partition.json`, frozen workloads, `executed_sources/`,
fit activations/directions/receipt, per-batch predictions/timing/capture hashes,
and primary/additive score files. Direction archive SHA is
`8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7`.
Startup evidence is `results/b200_sdpa_lens/sdpa02_projection/`; driver logs are
under `logs/runpod/b200_injection_direction/`.

Completion has `status.json: complete`, `summary.json`, `bootstrap.json`,
`probes.json` and no campaign `failure.json`. All **549 collected files**
(1,737,534,686 bytes) pass the remote collection manifest's checksums.
`independent_audit.json` independently refits directions/centers, verifies the
task partition against canonical inputs, binds all **48,462 primary/additive
predictions** to exact prompts/labels/tokens, checks finite logits/sigmoid and
answer-mass consistency, and recomputes APPS/ID ranking/threshold metrics and
bootstrap point estimates. `steering_serialization_audit.json` decodes both
saved vectors and verifies exact equality to fitted u20. Audit scripts/logs
are preserved under `logs/local/b200_injection_direction/`.

Closure verifies API 2031/worker 2070 and the frozen resident command/model;
capture, steering and projection request counts are zero, with zero uncorrected
volatile ECC errors. The passing engine stays warm. No intervention is promoted:
the direction provides a strong lever over injection-associated alarms, but
its recall/calibration tradeoff requires explicit treatment in any future use.

The real GPU smoke passes beta-zero exactness, span mean versus raw selected
captures, restoration of the next plain request and complete state cleanup.
Full-vocabulary and cached two-logit readouts agree exactly on the smoke;
the projection's maximum BF16 error against the FP32 tensor reference is
0.013839 (frozen atol .04/rtol .02). Remote focused checks pass **28 tests**
with 14 inherited Torch deprecation warnings; local data/campaign checks pass
**26 tests**. Scoped Ruff passes.

Preserve three pre-campaign failures: trailing-newline span parsing, a float64
test-reference direction under NumPy 2, and a missing Lens import because the
runtime wrapper replaces `PYTHONPATH`. All were corrected before source freeze
and scoring; none changed frozen scientific parameters. The final launcher
explicitly inserts the Lens overlay path. The old engine was retired
sequentially, retaining its receipts/caches; merged weights remain remote.
