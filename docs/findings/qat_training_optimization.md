# Merged QAT training systems optimization

The [screen](../../experiments/qat_training_optimization/README.md) optimizes the
completed [merged-weight QAD recipe](quantization_recovery.md), separately from
the ordinary frozen-base FP4 training path. It uses the same authorized NC2 B200,
pinned Torch 2.11 environment, shared compiler caches, strict-filtered 7,836-row
population and cached BF16 teacher. The original implementation and completed
research adapter remain unchanged.

## Numerical screens

`qat-training-optimization01` replaces the attention projection reference GEMMs
with Torch native E4M3 GEMMs that accumulate in FP32 and apply the original
per-token/per-channel FP32 descales before BF16 conversion. It also omits the
unused MLP down-output recomputation in backward. All nine isolated GPU GEMM
checks pass, with relative L2 between 4.68e-8 and 1.53e-7. Nevertheless, the
first logical batch fails the frozen full-model gates: score MAE 0.005474,
maximum score difference 0.031552, KL difference 0.001007 and adapter-gradient
relative L2 0.813334. Baseline KL is 0.00152248 versus 0.00252949 for the candidate.
The tiny GEMM discrepancies cannot establish full-model gradient parity in this
quantized, near-teacher regime. This does not measure subsequent model quality.
The failed receipt, original UUID-format startup failure and exact source archive
remain under `results/qat-training-optimization01/`. Native FP8 stays experimental.

`qat-training-optimization02` retains the original FP32 FP8 projection reference.
It shares the frozen GDN QKV/Z input packing and fused forward contraction,
preserving their separate STE backward calls, and removes the unused MLP backward
down-output contraction. Cache reuse requires the same input object/version and
weight-cache identity and ends after the sibling projection consumes it.

The full-model baseline repeat and candidate reproduce scores, KL and all 256
adapter gradients exactly on the first logical batch: all measured differences
are zero. The original 29,337-token longest-row backward also passes, with KL
0.79923505 and gradient norm 13.86179, matching the original preflight. Peak
allocated memory is 137,556,208,640 bytes (128.11 GiB), essentially unchanged.
CPU controls verify dense STE gradients, removal of the unused contraction,
shared-projection output/input-gradient parity, cache invalidation and scoped
restoration after exceptions.

## Complete-update timing

The matched timing queue completes baseline, optimized and optimized repeat,
twenty updates each, with updates 11–20 designated for comparison. Each starts
from identical FP32 adapters, seed-zero order and fresh AdamW state, retaining the
original 245-update learning-rate schedule. Only scratch weights are overwritten.
No teacher calls, held-out selection or model promotion occur.

| Trial | Mean seconds/update | Training tokens/s | Mean time reduction | Median paired time reduction |
| --- | ---: | ---: | ---: | ---: |
| Reference | 16.88658 | 18,337 | — | — |
| MLP/GDN optimized | 16.13599 | 19,190 | 4.44% | 1.73% |
| Optimized repeat | 15.69983 | 19,724 | 7.03% | 2.80% |

Both repeats pass the predeclared 2% mean-time gate and add zero FP4 plans in
measured updates. All twenty per-update KL losses and gradient norms reproduce
the reference exactly, as do physical partitions, learning rates and token
counts. All three final FP32 masters have the same tensor/name digest,
`6fafbd2f4b2c61aedce65d3e6bbcc943a497b1acc6bf193bda81ba0266e3e21a`.
The successful reference repeat also has zero first-batch numerical differences.
Independent local recomputation and source-archive verification are recorded in
`results/qat-training-optimization02/verified_comparison.json`.

The typical paired reduction is modest. Reference update 13 takes 19.7836 seconds
versus 14.7780/14.6487 for the optimized runs; update 14 takes 23.5380 seconds
versus 23.1973/19.9252. These variations inflate mean gains. Do not remove those
samples or substitute the descriptive medians for the frozen selection metric.
No complete compiler-specialization census or baseline timing repeat was recorded,
so compilation and execution noise are not independently isolated. This bounded
screen selects the opt-in structural implementation for systems use, with exact
tested trajectories, rather than establishing a 4–7% full-corpus wall-time gain.
It does not replace the frozen historical training entrypoint or promote a model.

Only the shared scratch adapter is overwritten. The completed research adapter
is preserved. The model worker remains resident and idle as PID 7591, with
receipt and log paths under the experiment's results and Runpod log directories.
Fourteen focused CPU tests and Ruff pass.

An isolated FP8 projection profile, collected only after the model timing worker
became idle, confirms substantial unused hardware potential. For
M=4097/N=8192/K=2560, the original decoded-FP32 contraction takes 3.029 ms versus
0.073 ms for native FP8; for M=16384 the same shape takes 10.837 versus 0.255 ms.
These warmed, uninstrumented calls include operand conversion and use FP32
accumulation. They measure isolated contractions, not complete training updates,
and do not override the failed full-model numerical gate. The CUDA trace identifies
the reference SIMT SGEMM and native FP8 tensor-core kernel. Timings, incremental
allocation peaks and the single-cycle profile are preserved under the same output.

## Future Kimi targets

The shared target loader validates both existing complete 7,836-row teacher
caches against the student workload. Source identities and ground-truth fields
must match exactly, with explicit teacher-model and cache checksums. The Kimi K3
cache has different teacher prompt hashes; its privileged prompt provenance is
kept separate from the student prompt identity. Binary KL still uses the cached
soft distribution rather than source hard labels. No new Kimi training has run.
