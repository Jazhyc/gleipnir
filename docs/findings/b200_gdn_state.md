# BF16 GDN state in vLLM

The 2026-10-07 B200 screen finds BF16 recurrent-state storage/native I/O
numerically feasible, but **no useful serving speed gain**: c128 input throughput
changes **+0.044%**, below the frozen >1% screen. Retain the selected
[repaired score reference](../decisions/b200_monitor_score_reference.md).
This changes neither training nor the inference selection.

## Scope and admission

The user authorizes replacing the leftover vLLM server on the existing Runpod
B200. The [experiment](../../experiments/b200_gdn_state/README.md) freezes native
and model gates, three c1/six c128 repeats, rejection/recovery and no promotion
before launch. Keep the selected merged 4B adapter, all FP4 GEMMs/SwiGLU,
MXFP8 full attention, causal LAST pooling, cache policy and frozen quick64/full320
prompts. Gates and MMA accumulation remain FP32; Q/K/V/convolution remain BF16.

FlashInfer 0.6.12's SM100 adapter supports BF16 initial/output state even though
its public prefill docstring specifies FP32. vLLM's ordinary wrapper upcasts
state; the candidate bypasses that upcast, supplies BF16 output buffers, and
sets `--mamba-ssm-cache-dtype bfloat16`. Loaded/dispatch/cache audits confirm
all 24 GDN layers and BF16 actual state tensors. The pinned kernel explicitly
supports only FP32 MMA accumulation. BF16 state I/O is not BF16 accumulation.

All seven native cases pass the frozen 3% output/state relative-L2 ceiling:
lengths 1,17,129, ragged [1,127,513],4096,[8192,8192],32768, nonzero state,
strided inputs, ragged isolation, sequential FP32 oracle, continued state and
changed-input CUDA-graph replay. Maximum output/state relative-L2 is
**0.01493%/0.16913%**. Host-inclusive isolated calls at [8192,8192] and 32768
take 0.37417/1.32293 ms versus 0.37925/1.34078 ms, about 1.3% shorter.
These bounded calls are not full-serving speed measurements.

Separately round `exp(g)` and update beta to BF16, then convert them back to the
required FP32 gate boundary. All synthetic cases remain finite; maximum
output/state error versus the FP32 reference is **0.52170%/0.48902%**. This
supports further numerical investigation, not model-level gate parity or a
speed claim. It does not test BF16 gate storage inside the kernel or lower
MMA accumulation; the synthetic distribution is not the model's gate population.

## Complete serving result

`bf16_state02` reuses the passed native receipt from `bf16_state01`, with every
bound source hash unchanged. Candidate API/engine 19718/19741 becomes ready
in 340.79 s, including 196.74 s compilation. The twenty-row accepted-reference
canary passes (mean error 0.00080652, correlation 0.99996689, nonzero adapter
effect); inherited strict-master failures remain separate.

| Repeat-median metric | Selected reference | BF16 state |
|---|---:|---:|
| c1 input tokens/s | 97272 | 97250 |
| c1 p50 | 31.225 ms | 31.472 ms |
| c1 p95 | 128.537 ms | 125.958 ms |
| c128 input tokens/s | 215707 | 215802 |
| c128 p50 | 2260.057 ms | 2237.618 ms |
| c128 p95 | 2615.344 ms | 2610.868 ms |

c1 scores/margins are exact, with unchanged AUROC. c128 source-macro/pooled
AUROC deltas are **+0.01619/-0.09181 percentage points**; mean/max score drift
is **0.001851/0.057704**, with no threshold flips. Per-source ranking,
calibration, ties, repeat variation and undefined single-label sources are
preserved in `c1_comparison.json` and `c128_comparison.json`. This is the frozen
training-seen development cohort, not held-out quality or production evidence.

Reference/candidate c128 repeat ranges are 212562–217083 / 201235–217751
input tokens/s. One candidate repeat is substantially slower. The sequential
archived-control comparison does not establish a 0.044% causal improvement.
Quality/latency screen limits pass; the throughput limit fails. Do not promote.

BF16 halves the D128/H32 state payload per slot from 2 MiB to 1 MiB, but the
worker still reserves the configured 90% GPU envelope. vLLM changes hybrid
attention block size from 528 to 272 tokens and reallocates the shared pool.
This does not halve total resident GPU memory or isolate state arithmetic from
cache layout effects. Initial `gdn_state.json` reports `state_bytes` summed over
logical views (433.1 GiB); shared backing makes that **not physical allocation**.
The audit now labels logical counts explicitly and deduplicates backing storage,
which itself includes other cache views. Preserve the original run receipt.

## Recovery, artifacts and verification

`bf16_state01` passes native checks, then fails before model loading because
the pinned remote checkout predates the local `serving.sources` move. Its
automatic recovery also fails by trying to stop an already-exited API. Fix the
new worker to verify its directly recorded sources; archive the failed process
only after verifying no GPU workers remain. Reconstruct the staged environment
from checksum-bound public runtime/frontend/cache receipts and retry, without
package upgrades, cold-cache reset or repeated controls. Both failed receipts
and logs remain archived.

After rejection, the driver retires the candidate and restores the original
reference in 100.37 s, passing its score canary and excluded warmups without
rerunning timing controls. Restoration is recorded in
`bf16_state02_reference_restored`; API/engine 20025/20048 is retained warm.
No capacity is created or terminated. The user subsequently directs that
reference servers must not be restarted after each rejected kernel trial; retire
the candidate and leave the GPU available for the next change. This supersedes
the initial restoration policy and is implemented in the runner and inference
guide. Live status must still be rechecked before future operations; these PIDs
describe the earlier restoration, not a lasting inventory.
Final closure verifies no resident vLLM/GPU process and 0 MiB GPU allocation;
the B200 capacity remains running. `bf16_state02/final_process_state.json`
records that observation without rewriting the earlier restoration receipt.

Artifacts under `results/b200_gdn_state/` preserve native sources/receipts,
the failed launch, executed candidate sources, parent/runtime/input/reference
bindings, every prediction repeat, full comparisons, screen and recovery.
Driver logs are under `logs/runpod/b200_gdn_state/`; archived server logs under
`logs/runpod/b200_attention_gdn_serving/`. Six focused tests and Ruff pass;
the source implementation is committed at `a672ad5`.

The expected explicit state-dtype override warning and existing pooling
PIECEWISE/development-profiler warnings remain logged. Real requests also
report first-shape Triton JIT compilation, including a later recurrent-update
variant. Preserve every timing repeat and
the slow first c128 repeat; do not silently discard it or attribute its delay
to a specific warning without launch-level evidence. No new mutation-analysis
failure or nonfinite score is observed.
