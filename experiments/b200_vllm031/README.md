# vLLM 0.31 runtime and scheduler migration

Hypothesis: vLLM 0.31 can run the selected Qwen3.5-4B monitor with its
FP4 projections/MLPs, MXFP8 full-attention prefill and FlashInfer GDN,
while retaining causal LAST pooling, exact tokenization and adapter effects.
The user authorizes testing on the existing B200; no capacity lifecycle changes.

Preserve the 0.24 environment, frozen inputs, archived predictions, compiler
caches and failed receipts before installing an isolated candidate runtime.
Port the source-bound integrations explicitly; never disable version guards
without examining the corresponding API and recording new runtime/source
identities. Do not select the opt-in vLLM CuTeDSL GDN backend. FlashInfer's own
Blackwell implementation uses CuTeDSL internally; retain the audited native
dependency overlay. Keep prefix caching disabled, 32,768
scheduled tokens, 128 sequences, the same merged checkpoint and native frontend.
Use V1 for the first matched migration; assess newer scheduler options separately.
The bounded scheduler comparison uses the upstream 4,096-token long-prefill
cap with its adaptive fair-share floor. The single-request path retains the
full chunk budget. A separate stock-FCFS condition caps active admission at
16 while retaining runner capacity 128. This cap is chosen from the archived
0.24 concurrency sweep's throughput plateau before observing these timings;
it is not combined with adaptive chunking. Compare each with migrated stock FCFS
as well as the frozen 0.24 control; changed physical row sizes can also change
the selected FP4 producer path, so this is an end-to-end scheduling comparison.

FlashInfer 0.7 introduces automatic context-parallel GDN dispatch. The paired
native diagnostic compares it and `use_cp=False` with the preserved 0.6 kernel
on identical short, long and mixed-length inputs. The migrated recipe requires
bitwise agreement of non-parallel output/state before selecting that published
route through a source-bound adapter. Keep the new CUDA GDN path and its Triton
ablation explicit; a setting named "decode" also changes prefill normalization.

Before timing, verify the classifier rows, finite scores, exact input token
counts and a nonzero adapter effect. Use the existing twenty-row accepted
serving canary (mean score error <=0.005, correlation >=0.995), recording
strict master diagnostics separately. Check chunked versus unchunked scores,
the exact context cap and mixed-length request isolation. Preserve evidence
if a guard fails; no silent backend fallback or automatic baseline promotion.

Benchmark frozen quick64 at c1 with three repeats and full320 at c128 with six
repeats, after excluded warmups and with fresh HTTP clients per pass. Compare
all archived selected repeats: input tokens/s, requests/s, latency bins,
score/margin differences, pooled/per-source/source-macro AUROC, repeat variation,
ties, calibration and threshold flips. This mixed systems set includes rows seen
by the adapter during training and does not establish held-out quality; the
[finding](../../docs/findings/b200_vllm031.md#development-cohort-quality) records
its composition and exact training overlap. Report host/runtime differences
explicitly.

Stop on provenance drift, truncation, missing/nonfinite outputs, failed canary,
startup failure, OOM or suite completion. On 2026-10-08 the user explicitly
authorized a finite failed-parity diagnostic benchmark after reviewing the
0.012–0.020 score MAE. Only that mode may continue past failed score agreement:
`--finite-canary-diagnostic` preserves `passed: false`, checks finite margins,
exact token counts and adapter effect, and keeps promotion disabled. Native
provenance/arithmetic guards remain required. The timed comparison uses CUDA
GDN, automatic FlashInfer context parallelism and stock/adaptive/active-16 scheduling;
normalization/non-CP/eager screens remain separate failed canary diagnostics.
Retire failed candidate processes;
retain a passing candidate warm for authorized follow-up work. Keep run history
and numerical evidence under `results/b200_vllm031/` and the linked finding.
See [the measured migration finding](../../docs/findings/b200_vllm031.md).

The root lock retains Transformers 5.14.1 (vLLM 0.31 requires <5.18) and TRL
1.9.0 without its unused vLLM extra, whose <=0.25.1 cap blocks the migration.
The preserved training environment is separate; this inference screen does
not validate training under Torch 2.13. Published vLLM-lens 1.3.0 pins 0.30.0;
any 0.31 lens compatibility test must record an explicit dependency override.
Scheduler preflight reads the merged model configuration when available, otherwise
the pinned base snapshot's cached configuration; it never loads model weights.

Startup reconciles frozen diagnostic-generator bindings from the checksum-checked
pre-migration source archive. It may restore only archived canary/comparison
scripts, never runtime helpers; executable source drift requires new validation.
The restoration receipt records every previous and restored hash. This preserves
the original numerical evidence across the repository's source-layout changes.

On the authorized B200, prepare `.venv-vllm031` without modifying the preserved
0.24 environment and use the retained native dependency overlays. On a fresh
container, keep the exact native Gigatoken package archive at
`.cache/runtime-resume/gigatoken-0.10.0-exact.tar.gz`. The candidate staging helper
restores its `/tmp/gleipnir-gigatoken-0.10.0` directory when missing, verifies the
selected frontend's package hashes, and registers its path in the persistent
environment through a `.pth` file copied into the staged environment. Registration
is once per Python environment; restoration is needed after losing container
storage, not for ordinary scorer restarts. The directory must include its
`gigatoken-0.10.0.dist-info` metadata as well as the native extension.

```bash
uv venv .venv-vllm031 --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-vllm031 uv sync --locked
PYTHONPATH=src:. .venv-vllm031/bin/python -m experiments.b200_vllm031.stage
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_vllm031.validate --name validation01
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_attention_gdn_serving.fp4_prepare_canary \
  --output results/b200_vllm031/vendor01.json
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_vllm031.gdn_compare \
  --output results/b200_vllm031/gdn_compare01.json
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_vllm031.run --name default01 \
  --native-receipt results/b200_vllm031/validation01/native.json \
  --gdn-receipt results/b200_vllm031/gdn_compare01.json \
  --vendor-receipt results/b200_vllm031/vendor01.json
```

The launcher requires an idle GPU, performs the score canary before timing,
archives executed sources and retains a passing scorer. Subsequent incompatible
conditions require retirement of that identity-verified process first.
Use `--scheduler-mode adaptive` for the fair-share screen, `--gdn-cp auto` for
the unmodified FlashInfer route, and `--gdn-decode-kernel triton` for the
normalization ablation. `--enforce-eager` isolates compilation and provides the
research intervention mode. `runtime_compare.py` uses serialized identical
inputs in both preserved interpreters to compare FP4 packing, QK, BF16 GEMM
and eager/compiled residual normalization; no reference timing suite is rerun.
Completed timings and quality are summarized with
`python -m experiments.b200_vllm031.summarize <run-name>`.
Use `--max-num-active-seqs 16` for the independent admission-cap comparison.

## Fixed-adapter ID evaluation

The user requests the same completed 272-update FP4/FA4-trained 4B adapter on
the frozen 3,012-row CoT-removed ID set (946 STRIDE, 2,066 Gloom). Hypothesis:
the migrated stock recipe changes low-FPR ranking relative to the same adapter's
archived 0.24 optimized and BF16 dynamic-LoRA results. `id_config.json` freezes
those controls, the canonical inputs, merged artifact and completed 0.31 stock
diagnostic recipe. No checkpoint, backend, threshold or prompt selection.

Run one complete pass with the original ordered 128-row partitions at c128,
excluded quick64 warmup, synchronous stock FCFS, CUDA GDN/automatic context
parallelism and the same 32K limits. Reuse native receipts and persistent caches.
The existing failed 0.24 score-parity guard remains recorded; diagnostic
continuation requires finite scores, exact tokens, nonzero adapter effect and
reproduction of the archived 0.31 canary within MAE 0.005/correlation 0.995.
Stop on provenance, reproduction, missing/nonfinite, truncation, transport or
startup failure, or suite completion. Preserve partial batches as diagnostics.
Report pooled/per-source/source-macro pAUROC@20 and AUROC, calibration, fixed-0.5
thresholds, score/margin drift, ties, prompt throughput and latency. One pass
does not measure repeat variation or promote a recipe. Retire only the owned
evaluation server after collecting receipts; leave capacity and caches intact.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_vllm031.id --name id01
```
