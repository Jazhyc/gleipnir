# B200 inference kernel screen

Hypothesis: native vLLM per-channel-weight/per-token-activation FP8 MLP GEMMs
improve merged-model throughput and latency on B200. Use the pinned 0.24.0 online
quantizer and native kernel selector, rather than training autograd wrappers.
Weights are quantized once during loading; include activation conversion in all
end-to-end timings. Only the 32 decoder MLPs change precision. Full attention,
GDN projections/recurrence, embeddings and vocabulary head remain BF16.

Freeze the existing 64 prompts, 269,411 input tokens, shuffled order, one-token
HTTP scoring contract, seed 0, concurrency 1/4/16 and two repeats, prefix caching
off, 32,768 context/token budget, 16 engine sequences and 0.25 memory fraction.
Compare with the completed merged BF16 suite, not the unmerged baseline.
The development rows are training-seen; labels never select a kernel. This is
neither a held-out quality evaluation nor a production arrival-rate/SLO result.

Use the existing NC2 B200, without allocating capacity. Reuse the completed
merged BF16 results and stop its idle server; the user explicitly questions the
need to reserve another 49 GB of GPU memory for an already measured control.
The FP8 candidate uses localhost 8010. Keep the merged checkpoint on ephemeral
disk; online quantization adds no persistent weight artifact. Reuse shared disk
compiler/kernel caches and record new cache keys/runtime/worker PIDs.

At model loading, audit all 64 fused MLP projection dtypes/methods/resolved GEMM
classes and require BF16 in every other loaded linear. Reject weight-only FP8
fallback, incorrect precision scope or incomplete layer coverage. Bind all
executed code/config/input/merged-artifact checksums. Run a fresh twenty-row
canary against the archived master, unmerged and merged BF16 references with
unchanged mean score difference <=0.02, correlation >=0.99 and nonzero adapter
effect. Stop before timings if it fails; preserve the failure and ask only if a
diagnostic continuation needs separate authorization. Stop on OOM, compiler or
backend failure, nonfinite/missing score, input drift or truncation.

Report prompt tokens/s first, latency and requests/s, paired repeat-median
score/margin differences and threshold flips against the merged control, and
within-candidate variation. A >10% warmed gain merits follow-up; passing a small
canary is not enough to promote production precision. Do not repeat the BF16
control, rerun full ID, tune concurrency or launch a broad kernel sweep here.
Retain a successful candidate for compatible work. Do not keep a separate
resident control unless a new matched control measurement needs it.

Historical SM120 experiments motivate this first choice: MLP-only native FP8
passed a different adapter/workload's checks and improved full-split throughput
1.26x, whereas tested FP4 inference layouts failed score gates. Those results
do not establish speed or fidelity on this B200/final FP4-trained adapter.
See `docs/findings/blackwell_inference_search.md`.

```bash
python -m experiments.b200_inference_kernels.run --output fp8_mlp02
```

Results: `results/b200_inference_kernels/`; logs:
`logs/runpod/b200_inference_kernels/`. Inspect startup every 30–60 seconds.
No in-chat scheduling tool is available; active-turn checks cannot promise an
agent wakeup after the turn ends.

The first attempt, `fp8_mlp01`, stops before model loading because port 8001
already has a listener. Preserve its receipt; the retry uses verified-free port
8010 and leaves the existing listener untouched. No GPU timing or numerical
measurement comes from the failed launch.
