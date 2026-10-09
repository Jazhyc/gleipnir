# vLLM Lens on the pooling monitor

2026-10-08. Lens 1.3.0 capture and steering work on the selected merged 4B
monitor under vLLM 0.31, in opt-in eager research mode on the existing NC2 B200.
The [experiment README](../../experiments/b200_vllm_lens/README.md) owns the
scope, commands, API and reproduction contract. The compiled serving selection
is unchanged. No learned direction, held-out evaluation or precision promotion.

## Integration and correctness

Upstream Lens targets generation and pins vLLM 0.30. A separate hash-locked
overlay installs Lens/compression wheels without dependencies, explicitly
retaining 0.31. Disable the generation plugin and attach a pooling worker
extension instead. All native classifier/precision/source audits remain active.
The client accepts upstream `SteeringVector` and decodes its compressed native
BF16 tensor format. TP/PP are restricted to one; generic hooks and Q/K capture
are outside this implementation.

Each request carries its own configuration in `PoolingParams.extra_kwargs`.
Capture the post-layer residual, including both halves of Qwen's fused return,
before final model normalization. Clone only the modified branch for steering.
Use Lens's vector addition and full-residual norm matching. Read the V1 runner's
packed CPU offsets and computed-token counters for absolute positions; our
MXFP8 metadata omits the fields expected by generation hooks. No token offsets
are inferred from KV cache pages.

`lens03` passes real-model checks on GDN layer 0 and attention layers 3/31:

- No-op capture and zero steering preserve logits and activations exactly.
- Nonzero final-token steering changes decision logits by -0.5/-0.25; the next
  plain request restores logits/activations exactly. This random direction is a
  functional test, without a semantic or quality-improvement claim.
- Mixed steered/plain requests preserve the plain score exactly in the smoke
  control. Steering the first token at the last layer preserves the last token
  and pooled readout, while changing the selected first-token activation.
- Eight concurrent 28,733-token prompts exercise partial prefill chunks, including
  boundaries at 4,035/8,070/12,105/16,140/20,175 tokens; captures at positions
  zero and 28,732 have the requested shape and ordering.
- The public client returns `(2,270,2560)` full-token captures; their final-token
  values equal last-only extraction. GPU norm matching with scale 0.1 yields a
  relative addition norm of 0.100099.
- A real HTTP timeout/disconnect leaves zero capture/steering entries;
  subsequent ordinary scores restore exactly. Normal completion also drains
  state. Shape/position errors are rejected before scheduling.

Eight CPU contract tests pass in the actual 0.31/Lens environment. The fourteen
inherited `torch.jit.script_method` deprecation warnings remain recorded.

## Numerical baseline

Eager execution is not numerically equivalent to the selected compiled recipe.
The twenty-row canary has score MAE **0.013987**, correlation **0.998307**
and nonzero adapter effect **0.820710**. The existing 0.005 reproduction gate
remains **failed**. `lens01` stops there; `lens02` exposes the metadata mismatch.
Both failed receipts and stopped-process evidence are kept. Corrected `lens03`
uses the explicitly user-authorized research eager mode, requires finite scores/
adapter effect, and tests Lens correctness against plain eager scoring. It does
not weaken or promote the compiled default.

Eager-versus-compiled development score MAE is 0.039510 on quick64/c1 and
0.044293 on full320/c128. Do not interpret studies as interventions on an
identical compiled numerical baseline. Capture-versus-plain-eager MAE is zero
at c1 and 0.002178 at c128. The latter comparison changes request-return/RPC
timing and batch composition in the already batch-dependent quantized model;
no-op hooks preserve isolated computations.

## Matched systems measurements

One excluded warmup and three timed repeats per new condition/cohort. Reuse all
three completed NC2 compiled full320/c128 controls; collect quick64/c1 once from
the warm compiled parent before replacement. Each column below is internally
matched; c1/c128 use different cohorts (269,411/1,310,581 input tokens).
Latency includes localhost HTTP, encoding/inference and capture serialization/
transport, excluding semaphore wait. Throughput includes complete pass wall time
and client-side capture validation.

| Condition | c1 input tokens/s (range) | c1 p50/p95 | c128 input tokens/s (range) | c128 p50/p95 |
| --- | ---: | ---: | ---: | ---: |
| Compiled control | 26,735 (26,629–26,864) | 151.36/251.38 ms | 175,093 (166,769–176,741) | 2.567/3.077 s |
| Plain eager | 23,990 (23,963–24,090) | 161.44/262.33 ms | 146,139 (145,347–147,030) | 3.144/3.660 s |
| Eager + layer-31 last-token capture | 22,733 (22,644–22,771) | 167.53/270.23 ms | 156,008 (152,458–156,242) | 3.023/3.736 s |

Capture costs 5.24% input throughput relative to plain eager at c1. At c128,
capture throughput is 6.75% higher while p95 rises 2.09%; this is a closed-loop
comparison with different completion timing/batch shapes, not isolated hook
cost or evidence that capture improves kernel throughput. Compared with compiled
c128, plain eager is 16.54% slower and capture 10.90% slower.

Ranking deltas versus compiled controls, in percentage points:

| Condition/cohort | Pooled AUROC | Source-macro AUROC | Pooled pAUROC@20 | Source-macro pAUROC@20 |
| --- | ---: | ---: | ---: | ---: |
| Plain eager, quick64/c1 | +1.2808 | -6.9444 | +1.5862 | -0.3030 |
| Capture, quick64/c1 | +1.2808 | -6.9444 | +1.5862 | -0.3030 |
| Plain eager, full320/c128 | +0.2520 | +1.8389 | +0.6179 | +1.2798 |
| Capture, full320/c128 | +0.2325 | +1.7431 | +0.8230 | +0.7051 |

Quick64 has many undefined single-label groups. These training-seen diagnostics
do not select directions or serving precision. All repeat scores, calibration,
ties, per-source results, undefined sources and length-bin latency are retained.

## Artifacts and closure

`results/b200_vllm_lens/lens01/02/03/` retain controls, failures, successful smoke,
native audits, executed sources, timed/warmup predictions and client checks.
`summarize` exports source-bound JSON/CSV and PNG/SVG overhead figures. Important
artifacts/logs are collected and checksum-verified locally and retained on the
volume. At this experiment's completion the passing eager scorer was API 11667,
engine 11706; current lifecycle state is in [infrastructure](../infrastructure.md).
Capture buffers are empty after completion/disconnect. No capacity creation/
termination or persistent cache reset occurs.

## Next intervention study

The user selected the completed current BF16/SDPA augmented replay as the next
mechanistic-intervention target, because its injection sensitivity more closely
resembles historical behavior. Use its preserved FP32 master and merged BF16
checkpoint, with the frozen ID/APPS scores as controls. Build on the Lens
activation-capture/steering integration to study prompt-injection responses.
Freeze the intervention/selection contract and establish the matched plain
research-serving baseline before attributing effects to steering. The current
SDPA/FA4 comparisons do not resolve training nondeterminism; the user deferred
that investigation. The restoration below establishes the runnable baseline;
no learned-direction study or replacement capacity is launched.

## Restored BF16 Lens baseline for the SDPA adapter

On user-reserved NC2 Pod `64h2ad4w3nfq2q`, restore the checksum-verified vLLM
0.31/Torch 2.13 serving runtime, Lens 1.3.0/native overlays and SDPA merged
checkpoint from the retained volume. No package installation or new merge is
needed. The training runtime remains archived. Direct SSH is enabled through
the account proxy without restarting the container. The physical GPU, driver
and cgroup limits match the prior NC2 Pod.

`b200_sdpa_lens/sdpa01` launches the same adapter-specific unquantized BF16
command, adding eager execution and request-scoped Lens hooks. Production
selection remains unchanged. Startup takes 54.26 seconds, excluding archive
restoration and prelaunch identity checks. Its 20-row master canary passes at
MAE **0.000706**/correlation **0.999995**, maximum drift 0.004761 and adapter
effect 0.897826. Against the same adapter's compiled BF16 canary, MAE is
0.002009/correlation 0.999883, maximum drift 0.031209. This bounded gate is
not a full ID/APPS eager quality evaluation. Reuse the existing merge/master
receipt because every restored merge file is identical.

Native audits verify all BF16 projections, eight actual causal BF16 attention
calls and cached BF16 head rows 15/16. No-op capture/zero steering are exact;
nonzero final-token steering changes logits by -0.5/-0.25 and the next plain
request is restored exactly. Mixed-request plain score drift is zero. Six of
eight 28,733-token concurrent requests exercise chunked capture. Absolute
position isolation and invalid-position rejection pass. The public client
verifies all-token shape `[2,270,2560]`, full/last equality, norm-matched
relative addition 0.099990 for requested scale 0.1 and disconnect cleanup.
Capture and steering request counts are both zero at closure.

Thirty-one local campaign/startup tests and fifteen tests in the restored
Lens runtime pass; fourteen inherited Torch deprecation warnings remain in
the remote log. An initial test-helper JSON-index error is corrected and its
failed log retained; no server failure occurs. Independent verification
recomputes canary identity/logits/sigmoid/parity, validates native/smoke/client
receipts and checks 29 collected files, eight executed/current sources and
eight pinned inputs. The passing scorer remains warm: API 1335, worker 1389,
localhost port 8010, zero ECC errors. Merged weights stay on the B200.

The [restoration contract](../../experiments/b200_sdpa_lens/README.md),
`results/b200_sdpa_lens/sdpa01/` and `completion_audit.json` retain provenance.
The shared functional smoke accepts this adapter's prevalidated canary rather
than comparing its scores with the regular adapter's canary. No full quality
benchmark, intervention selection or determinism study occurs.
