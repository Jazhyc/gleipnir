# Length-aware admission for monitoring scores

Hypothesis: prioritizing short waiting prompts improves mixed-length request
latency without sacrificing the selected B200 scorer's bulk throughput. This is
an opt-in waiting-queue intervention, not a kernel or precision change.

The completed initial screen fails its latency/fairness and pooled-AUROC bounds.
Keep upstream FCFS selected; see [the finding](../../docs/findings/b200_length_admission.md).

The custom vLLM 0.24.0 synchronous pooling scheduler sorts waiting requests into
<=1024, <=4096, <=16384 and larger-token buckets, FIFO within each bucket.
After 250 ms since vLLM's request arrival, a request precedes fresh buckets in
original FIFO order. Promotion occurs at scheduler steps; it is not a wall-clock
completion guarantee and cannot bypass memory or blocked-request eligibility.
Preempted requests retain their original age. Both waiting and skipped queues
use the same ordering, with one timestamp per step. Running requests, chunked
prefill, token/sequence limits, caches, causal LAST pooling, precision, graph
settings and the repaired classifier remain upstream. The baseline pooling
runner already disables async scheduling by default; pin that explicitly here.
Installed scheduler/queue and both helper source hashes must match at startup.
CPU policy and source hashes are passed in `GLEIPNIR_LENGTH_ADMISSION_CONFIG`
and saved as `length_admission.json`, separately from the GPU compilation key.
The override forwards vLLM's per-step prefill-throttle argument unchanged.

The first candidate is fixed in `config.json`; do not tune on final ID. Reuse all
selected archived c1/quick64 and c128/full320 controls through the shared score
runner: one warmup, three c1 and six c128 passes, adapter canary, finite outputs,
exact prompt identities, pooled/per-source/source-macro AUROC, score drift,
calibration, ties and flips. This training-seen systems cohort is diagnostic.
Keep inherited strict precision failures distinct from finite acceptance.

Closed-loop throughput alone cannot demonstrate admission fairness. The separate
`traffic` entrypoint freezes full320 order and Poisson arrivals (seed 17), at
40 and 80 requests/s, with three repeats after excluded warmups. Include client
dispatch lag in arrival-to-response latency and report it separately; no client
semaphore reshapes arrivals. Reject a timing interpretation if p95 dispatch lag
exceeds 5 ms. Record short/medium/long p50/p95/p99 and prompt tokens/s. A new FCFS
mixed-arrival control is needed because archived closed-loop results cannot
answer this question; reuse that new control for later candidates. This is not
authorization to rerun the archived throughput control.

Screening rule, frozen before GPU work: >=10% reduction in under-4K mixed-arrival
p95 at 40 requests/s; <=2% loss of archived c128 median prompt throughput;
<=10% increase in >=16K mixed-arrival p95 at either rate; <=2% c1 median/p95
regression; absolute pooled/macro AUROC shifts <=0.1 percentage points. Preserve
repeat variation and undefined single-label sources. No automatic promotion.
Stop on source/input drift, nonfinite output, canary/server failure or completion.
Failed candidates are retired without restarting the reference. Keep a completed
candidate warm only for compatible follow-ups; retire it if subsequently rejected.

Preparation is CPU-only and never queries or changes the GPU:

```bash
PYTHONPATH=src:. python -m experiments.b200_length_admission.run \
  --name length_prepared --prepare-only
```

When capacity is free, use a verified live selected scorer as the parent, or
`--retired-parent <preserved_server.json>` to reuse its staged runtime and caches
on an idle GPU. The runner restores the selected recipe in its command before
adding the policy; it does not start a reference server as an intermediate step.
It refuses an idle-parent launch if any GPU compute process is present.
For a live-parent replacement it also rejects compute processes outside that
server's recorded API/worker pair. It never stops the other agent's workload.

```bash
PYTHONPATH=src:. python -m experiments.b200_length_admission.run \
  --name length01 --retired-parent <preserved_server.json>
PYTHONPATH=src:. python -m experiments.b200_length_admission.traffic \
  --name length_arrivals01 --reference results/b200_length_admission/fcfs_arrivals01
```

The traffic client only sends requests to an already-ready scorer. Run its
control against the same selected FCFS recipe before comparing the candidate.
Use `run --control-only --name fcfs_start01 --retired-parent <receipt>` to start
the matched FCFS control with adapter canary and excluded warmups only. Then
run `traffic --name fcfs_arrivals01` before replacing that server with the
candidate. This delegates admission entirely to upstream vLLM and never reruns
archived throughput timing controls.
Neither entrypoint creates or terminates capacity. Results remain ignored under
`results/b200_length_admission/`; shared server metadata and runtime logs retain
the established scoring harness locations. Monitor any future launch in the
active session; no after-turn monitoring is installed by this implementation.
