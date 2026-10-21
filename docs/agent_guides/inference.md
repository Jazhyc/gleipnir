# Inference and evaluation instructions

These instructions are required for the tasks routed here by the root
[AGENTS.md](../../AGENTS.md). Standing preferences were moved from that
file on 2026-10-07; their scope and historical acceptance remain intact.

Also read [compute.md](compute.md) before launching or operating a run.

## Serving and adapter parity

Importing `gleipnir` puts the running interpreter's `bin/` first on `PATH`
(`src/gleipnir/_environment.py`), so JIT builders find the venv's `ninja`
without activating it. Launch runners through a project entrypoint that
imports `gleipnir`; scripts that don't import it must set `PATH` themselves.

Standing user preference, updated 2026-10-13: use BF16 vLLM for ordinary adapter
iteration and training-recipe comparisons. Use optimized precision serving only
for QAD-adapted deployment candidates. Monitor scoring
uses the cached causal two-logit path in the
[serving decision](../decisions/b200_monitor_score_reference.md), including its
validated context handling and pooling boundary correction. Select this path
explicitly in new experiment launch configurations; the decision defines the
precision selection, recipe and supported hardware/model envelope. Reuse a
persistent engine and
continuous batching rather than an eager training microbatch.

New adapters and backend/layout combinations require master-to-serving score
agreement and a nonzero adapter effect before scaling evaluation. Reuse unchanged
native/kernel receipts and persistent caches. Record unsupported model/task/GPU
combinations and any required backend exception explicitly; use Transformers
eager for bounded parity diagnostics or unsupported adapters. Preserve frozen
past evaluation contracts and their named controls. Training precision and
optimization are governed by the separate training guide.

Standing user preference, 2026-10-08: create an adapter-specific merged BF16
checkpoint for every trained adapter used in inference, serving or evaluation,
serving without dynamic adapter projections.
Keep FP32 master adapters and pinned base weights persistent. On Runpod, store
reconstructable merged checkpoints on ephemeral container storage, with merge
source/file checksums and evaluation receipts on the network volume. Accumulate
the merge in FP32 before exporting BF16 and run adapter-specific serving parity
before scaling evaluation. Record unsupported layouts or failed parity explicitly;
do not silently fall back or change frozen historical evaluation contracts.
Changing adapters requires loading the corresponding merged model; preserve disk
compiler caches across those restarts. See the merged serving protocol in
`experiments/b200_inference_benchmark/README.md`.

If a training repeat reproduces the exact master/export weight hashes and
equivalent adapter/base identities, reuse the existing merged checkpoint,
parity and evaluation receipts. Record identity and score reuse explicitly;
repeat scoring only when measuring serving variability or changing its recipe.

## Released optimized serving image

`ghcr.io/jazhyc/gleipnir-serve:0.2` (digest `ba602599…`) serves the released
Gleipnir 4B v0.4 and 27B v0.2 through their validated optimized stacks on one
B200, at up to 262,144 tokens, with opt-in `--prefix-caching`. `0.1`
(`15673c65…`) is the earlier 32K image. It is a frozen snapshot of the recorded runs' command, environment,
receipts and runtimes. Use it to score released models on a B200 without
restoring the volume stack:

- **Launching on Runpod.**
  - Create the Pod from the image with no volume mounted at `/workspace` or
    `/tmp`; the image keeps its repository at `/workspace/gleipnir` and its
    runtimes under `/tmp`.
  - For SSH, set the start command to `/start.sh` (entrypoint
    `/opt/nvidia/nvidia_entrypoint.sh`).
  - Then run `/opt/nvidia/nvidia_entrypoint.sh /opt/gleipnir-release/entrypoint.sh {4b|27b} --port 8000`.
    It fetches `Jazhyc/Gleipnir-*-BF16@<version>`, verifies the hashes in
    `experiments/release_serving/targets.py`, and replays the recorded
    server command (`experiments/release_serving/serve.py`).
- **Scoring.** Send rendered prompts to `POST /v1/monitor/score` with
  `{"model": "monitor", "prompt": ...}`.
  `experiments/release_serving/validate.py` scores the frozen ID/OOD
  workloads in the recorded campaign's batching and compares them with the
  recorded rows.
- **Scope.** It covers the released checkpoints only. New adapters, recipes
  or stack changes still go through the volume stack, merged parity and the
  serving decision above.
- **Rebuilding.** Follow the procedure in
  `experiments/release_serving/README.md`:
  - restore on an EU-RO-1 Pod with the volume;
  - remerge with byte identity;
  - trace, build with `crane`;
  - validate on a fresh Pod against repeat-relative gates.

  Never ship partially traced compile caches.

Stock `vllm serve` of the released `-BF16` repositories is documented in the
root README and validated with `cluster/slurm/release_stock_vllm.sh`. It
needs the locked vLLM 0.31 environment `.venv-vllm031`
(`UV_PROJECT_ENVIRONMENT=.venv-vllm031 uv sync --locked`); the cluster's
default `.venv` keeps vLLM 0.24 for frozen controls, and that version
returned NaN logits there.

## Persistent serving process

Standing user preference: when an inference change needs a different
serving process, stop the old server rather than keeping both resident. Preserve
its measured baseline results, merged checkpoint, logs and persistent caches.
Reuse completed matched baseline results; do not retain or rerun a control merely
because a new kernel is being tested. Keep the active candidate warm for compatible
trials. After a rejected or failed kernel trial, retire its worker and leave the
GPU available for the next change; do not automatically restart the reference
server. Restore a reference only when the user requests it or the next task needs
that process. This is process replacement, not authorization to terminate capacity.

## Optimization reporting

Standing user preference, 2026-10-06: inference optimization reports must include
prompt tokens per second alongside requests per second and request latency.
Use prompt-token throughput as the main compute-throughput comparison for the
one-token monitor, keeping the prompt-length distribution and cache policy fixed.
Label input/prompt throughput explicitly rather than confusing it with generated
output-token throughput.

Standing user preference, 2026-10-06: every inference kernel update must report
AUROC deviation against the same frozen baseline, alongside speed and score
parity. Include pooled/per-source AUROC and macro across sources with both
labels, repeat variation, explicit undefined single-label sources, and bind
labels to exact prompt identities. Reuse archived baseline predictions rather
than rerunning the control. Small training-seen optimization sets are diagnostic
and do not establish held-out quality parity or select production precision.

## B200 comparison controls

Read the [serving decision](../decisions/b200_monitor_score_reference.md) and
`experiments/b200_inference_benchmark/baseline.json` before preparing comparisons.
Its checksum-bound controls supersede the October 6 GDN-only reference below;
the future-experiment default does not rewrite these frozen measurements.
Compare timing and quality on matched concurrency/cohorts using all available
repeats. The [concurrency](../findings/b200_score_scaling.md),
[fixed 2K](../findings/b200_context_scaling.md),
[long-context](../findings/b200_long_context.md) and
[1M-context](../findings/b200_context_million.md) records define their additional
measurement contracts. Preserve failed strict checks separately from accepted
finite quality. Acceptance remains adapter/recipe specific and does not waive
finite-output checks, new-adapter parity or held-out evaluation requirements.

Keep the current launch selection and frozen comparison bindings separate, as
defined in the linked serving decision;
avoid duplicating changing result paths, measurements or worker PIDs here.

## Historical GDN baseline acceptance, 2026-10-06

The following preserves the earlier standing preference and its numerical
acceptance. Its selected control and BF16 full-attention recipe are historical;
new comparisons use the current reference above.

Standing user preference, 2026-10-06 (updated): use FROST native FP4 MLPs
and all 48 large GDN QKV/Z and output projections as the B200 inference
optimization baseline. Keep small gate projections, convolution and recurrence
operands BF16, gates/state FP32, and full attention BF16 FlashInfer. The user
explicitly accepts the c128 source-macro AUROC drop of 0.47 percentage points;
retain the strict failed score-canary receipt separately as `user_accepted_finite`.
The selected control is `results/b200_attention_gdn_serving/fp4_gdn_projection02`,
with five-pass warmed confirmation at `fp4_gdn_confirmation01`, bound by
`experiments/b200_inference_benchmark/baseline.json`. Prioritize c128 throughput,
retaining peak throughput, lower-concurrency latency and AUROC diagnostics.
New conditions compare against this selected baseline; preserve explicit
historical references. Reuse archived controls and persistent caches. Stop the
old server before changing active kernels so the candidate gets the whole GPU.
Keep BF16 GDN/MLP and FP8 comparisons as historical context. This acceptance
applies to the current adapter and pinned serving recipe; it changes neither
training nor frozen past evaluations and does not waive missing/nonfinite
outputs or adapter-specific checks for future layouts/adapters.
