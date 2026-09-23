# Real-activation INT4 calibration screen

Hypothesis: clipping, channel rebalancing, orthogonal rotations, smaller groups,
or selective higher precision reduce the naive W4A4 MLP error. This is a bounded
activation-reconstruction experiment, not model promotion or a full judge run.

Freeze twelve shortest distinct trajectories, three per source/label stratum,
from the existing checksum-pinned 512-row subset, excluding every iteration32
identity/original-trajectory hash and original four canaries. In each stratum,
first and third become calibration, second held-out. Reject shared known hashes
across splits; broader generator lineage is not available in this materialization.
This short-input selection is biased and must not be called population validation.
Capture 256 uniformly spaced token positions including endpoints for layers
0,16,31, at gate/up inputs and down inputs. No prompt truncation. Eight calibration
and four held-out rows give 2048/1024 activation vectors per projection.

Use Transformers causal-LM BF16 SDPA only for this bounded activation capture,
since hooks are needed; this is an explicit exception to persistent-vLLM scoring.
Load the original merged model unchanged; preserve checkpoint/input/prompt hashes,
selected IDs, position-selection rule, software and backend. Capture once per
row, without repeated dataset timing. No training or external API calls.

Numerical reference is FP32 X @ W.T with TF32 disabled. Test naive INT4, weight-
only and activation-only INT4 diagnostics, BF16, and FP8 dequantization simulation.
Calibrate clipping on independent X/W multipliers {1,.95,.9,.8,.7,.6} of row max.
Rebalancing scales each input channel by activation_max^alpha / weight_max^(1-alpha),
alpha in {.25,.5,.75}; verify the unquantized product is unchanged. Test fixed
seeded signed block-Hadamard rotations of sizes 128 and 512 on both X and W;
verify product invariance. Test INT4 groups 64 and 128 along K. Pick the best
setting in each family using calibration relative L2 only, then freeze it before
evaluating held-out activations. Also select a BF16/FP8 fallback for the three
most sensitive of the six projections based only on calibration naive error.
These are standalone linear interventions, not full QuaRot/SmoothQuant algorithms
or validated graph rewrites for Qwen3.5's nonlinear/recurrent architecture.

Report per-projection and unweighted macro relative L2, max-error/reference-RMS,
and calibration/held-out gap. No fixed quality-pass threshold or AUROC inference.
Stop on nonfinite data, identity drift, missing captures, transformation-invariance
error >1e-5, or incorrect arithmetic. Save partial results after each projection.
All new artifacts use results/int4_calibration; never overwrite prior attempts.

After numerical screening, measure applicable native recipes on layer-0 MLP
shapes using real calibration activations and one 256-call window each. Include
online transforms/quantization/scaling, exclude offline weight preparation, and
retain preallocated-buffer BF16 reference. Small-group implementations require
partial-sum scaling and must not inherit the earlier row-scaled timing. Numerical
simulation times are never reported as native INT4 serving speed. Monitoring is
active-turn only; this session has no agent scheduling tool for later wakeups.

## Results (2026-09-24)

All five families were tested. The main issue is activation quantization, not
integer arithmetic. On real inputs, average held-out linear-output error is
larger than on the earlier synthetic Gaussian screen. Values below are unweighted
means across six sampled projections, **not judge-score drift or AUROC**:

| Recipe | Calibration relative L2 | Held-out relative L2 |
| --- | ---: | ---: |
| Naive W4A4 | 51.98% | 52.06% |
| Weight-only INT4 diagnostic | 15.24% | 15.40% |
| Activation-only INT4 diagnostic | 49.67% | 49.64% |
| Calibrated clipping | 38.19% | 38.10% |
| Channel rebalancing | 36.37% | 38.49% |
| Signed block-Hadamard rotation | 21.79% | 21.92% |
| Grouped INT4 | 14.27% | 14.29% |
| FP8 numerical reference | 2.62% | 2.62% |
| BF16 reference | 0.21% | 0.21% |

The calibration-chosen group size was 64 and rotation block size 512 in all six
projections. Clipping selected activation multiplier 0.6 throughout and weight
multipliers 0.6–0.8. Several winners sit on the tested grid boundary; this is a
bounded screen, not an optimized quantizer. Rebalancing selected alpha 0.5 or 0.75
and was much less effective for down projections. Its calibration/held-out gap
is larger than the other families, illustrating limited calibration coverage.

Selective higher precision was also simulated: retaining BF16 for the three
calibration-most-sensitive projections (16_down, 0_down, 16_gate_up), with naive
INT4 elsewhere, reduced macro held-out error to **23.01%**. FP8 fallback gave
**24.42%**. These are local mixed-precision comparisons, not a full-model policy
or serving benchmark. No combined rotation-plus-grouping/clipping optimization
or quantization-aware training was attempted.

## Native performance and costs

Native timings use **layer-0 calibration activations**, not held-out activations;
all online activation work is included, offline weight transforms excluded.
One 256-call window per distinct condition, no repeats:

| Complete path | Gate/up ms | Down ms |
| --- | ---: | ---: |
| BF16 | 2.0846 | 1.0755 |
| Naive INT4 | 0.7496 | 0.3150 |
| Calibrated clipping | 0.7585 | 0.2839 |
| Channel rebalancing | 0.9829 | 0.8132 |
| Rotation, dense transform prototype | 1.0346 | 1.4649 |
| Rotation, fast Triton Hadamard | **0.8153** | **0.5611** |
| Group-64, unfused partial-output prototype | 23.3320 | 10.6683 |

The fast-Hadamard variant was added after the dense transform's overhead was
observed. It uses the same calibration-selected rotation, verified against the
dense transform before timing. Its **2.557x / 1.917x** speedup over the saved
same-screen BF16 timings includes rotation, packing, native GEMM and output
scaling. It was timed in a separate process at cooler temperatures, so do not
interpret small differences or claim an end-to-end speedup. All timing windows
are uncontrolled short-window measurements.

The group prototype is roughly 10–11x slower than BF16 because it launches a
separate GEMM per K-group, materializes all INT32 partial matrices, and then scales
and sums them. This demonstrates the cost of that implementation, **not an
inherent bound on grouped INT4**. A fused kernel could avoid much of it, but was
not implemented. FP8 selective paths have numerical evidence only; we did not
benchmark a new mixed-precision vLLM engine.

## Validation, provenance, and limitations

Capture processed eight calibration and four held-out trajectories once each,
with prompt lengths 2,219–4,173 and no truncation. The 32-row evaluation slice
and original canaries were excluded. Twelve forwards took 13.20 seconds excluding
loading. Transformers warned that its gated-delta fast path was unavailable and
used the Torch fallback; that backend choice is recorded and is **not vLLM
activation parity**. Captured token positions are uniformly spaced, not an
independent sample of trajectories. Only layers 0,16,31 and MLP linears were
covered; quantization errors were not propagated through a quantized model.

Local unquantized transformations passed the <1e-5 invariance check. Every native
group/recipe GEMM was checked against exact integer products. A first native
attempt completed only gate/up smoothing, then failed comparison with simulated
group quantization. Investigation found differing FP32 division rounding around
quantization ties: PyTorch's scale division uses reciprocal multiplication,
whereas the native packer uses explicit rounded division. Recovery separately
checked native scales/packed integers and final scaled BF16 outputs exactly, and
retained the discrepancy from the numerical simulation as a diagnostic. Native
versus simulated relative output differences were approximately 0.16–1.8%; they
must not be described as identical quantizers. The original smoothing timing was
reused without repetition; its earlier validation is retained, not retroactively
relabeled with the additional checks. Fast-Hadamard variants passed all checks.

No model checkpoint, serving defaults, or installed package was changed. No
quality promotion, 32-row AUROC evaluation, or native W4A4 vLLM integration.
**Conclusion:** FP8 remains the measured end-to-end candidate. Fast rotated INT4
offers an interesting kernel tradeoff, but its reconstruction error is still far
above FP8. Grouped INT4 improves reconstruction most and needs a fused backend
before its speed can be assessed fairly. These are short-input development
findings, not population or full-stack conclusions.

Artifacts under `results/int4_calibration/`: `capture/manifest.json`, captured
tensors and weights, `screen/*_selection.json` (choices saved before held-out
evaluation), `screen/result.json`, `native/result.json` (partial),
`native_recovered/result.json`, `native_fht/result.json`, and `comparison.json`.
The combined report includes source hashes and an audit of calibration-only
selection. Logs are `logs/local/local_inference/int4_calibration_*.log`.

Entrypoints (use distinct output directories when reproducing):

```bash
.venv/bin/python -m experiments.int4_calibration.capture --output results/int4_calibration/capture
.venv/bin/python -m experiments.int4_calibration.screen --capture results/int4_calibration/capture --output results/int4_calibration/screen
.venv/bin/python -m experiments.int4_calibration.native --capture results/int4_calibration/capture --screen results/int4_calibration/screen --library /tmp/gleipnir_int4_gemm.so --output results/int4_calibration/native
# Add --previous <partial-native-dir> only to recover untimed conditions.
# Add --fast-rotation to time the distinct Triton transform implementation.
.venv/bin/python -m experiments.int4_calibration.summarize
```
