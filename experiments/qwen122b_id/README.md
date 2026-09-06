# Qwen3.5-122B-A10B FP8 ID teacher screen

## Current findings and follow-up authorization

Qwen3.5-27B FP8 completed all 3,012 rows with macro pAUROC@20
0.9133745803, macro AUROC 0.9785879965, and macro Brier 0.0858109206.
Per-source pAUROC@20 is 0.8796745339 (Gloom) and 0.9470746267 (STRIDE).
Its macro pAUROC advantage over the 122B MoE is 0.0398246579, far outside
the user's 0.001 speed-tiebreak band. It leads both sources and calibration.
The recorded scoring invocation took 1,443.58 seconds versus 1,076.56 for
122B, so the MoE was about 1.34x faster under this fixed no-cache screen;
that does not override this quality gap or predict prefix-cached throughput.
Qwen3.5-27B initialization took 95.87 seconds (122B: 642.46); these startup
times are not matched cold-compilation measurements. Teacher selection still
awaits the bounded Qwen3.8 eager diagnostic.

The user subsequently authorized selecting a prefix teacher from these ID
screens, preferring measured faster scoring within 0.001 source-macro pAUROC@20
of the best. Prefix annotation is a separate prompt-reviewed campaign under
`experiments/monitoring_prefix_supervision/`; the original screen's no-annotation
stop condition below describes its initial scope, not this later authorization.

The 122B FP8 run completed all 3,012 predictions. Source-macro pAUROC@20 is
0.8735499224 (Gloom 0.8506562270; STRIDE 0.8964436178), macro AUROC
0.9674993400, and macro Brier 0.1418375199. Recorded scoring invocation time is
1,076.56 seconds, separate from 642.46 seconds of engine initialization.
Complete prediction and result artifacts have been collected locally. The
supervisor's final prediction-count field is stale; the actual JSONL and result
both contain 3,012 rows.

Qwen3.8-27B failed the frozen five-row batch/singleton numerical canary:
MAE 0.0158917046, maximum 0.0567193810 (limit 0.05). The largest difference
was on the longest Gloom row (0.6791787122 versus 0.6224593312). No full ID
score exists; do not treat this as a quality comparison or relax the gate after
observing the failure. Artifacts were collected for diagnosis.

The dependent Qwen3.5-27B queue consequently failed before model loading. After
verifying no evaluator remained and both GPUs were idle, its unchanged frozen
evaluation was started independently (supervisor PID 139344). The earlier queue
status is retained remotely as `dependency_failure_status.json`. This is not a
retry of Qwen3.8 and does not change either model's numerical acceptance gate.

Frozen question (2026-09-06): can the official FP8 MoE provide useful local
teacher scores at practical throughput? Evaluate all 3,012 existing ID rows
(946 STRIDE test, 2,066 Gloom exfiltration). No OOD reads, annotation campaign,
training, or prompt/threshold tuning is authorized by this screen.

Use the full existing teacher rubric, non-thinking Qwen chat wrapper and
selected literal 0/1 logprobs, generating one token. The official FP8 revision
is a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 (127,163,011,832 weight-file bytes).
Both reserved Lambda GPUs are H100 80GB HBM3, driver 580.126.20. TP2 is required;
skip the vision tower, retain BF16 activations and checkpoint-declared dynamic
block FP8 quantization. No BF16-versus-FP8 quality equivalence is claimed.

Initial systems contract: 32K context, eight concurrent sequences, 8,192
batched prefill tokens, GPU-memory utilization 0.90, Triton GDN prefill and
prefix caching disabled. This new model has no historical backend contract;
Triton avoids the recurring FlashInfer GDN evaluation stalls. This is not a
prefix-cache throughput test or a matched speed comparison against 27B.

Before full scoring, require the four source-label canary rows plus the longest
ID request to return finite decision scores and fit without truncation. Repeat
those requests individually; require batch/singleton mean absolute probability
error <=0.02 and maximum <=0.05. This checks numerical consistency, not accuracy
or FP8 quantization parity. Keep the original batch scores, record repeats
separately, and evaluate the remainder in the same persistent engine.

Report source-macro and per-source pAUROC@20, AUROC, Brier, threshold diagnostics
and ties. Existing ID teacher/student scores are contextual baselines; models
with different prompts or precision are not matched capacity ablations. The
old 27B OOD score is not an ID comparator. Stop after the fixed full endpoint;
no automatic teacher selection or billion-token annotation. Stop on provenance
drift, tokenizer audit drift, GPU contention, OOM, missing logits, failed
canary, nonfinite scores, or 20 minutes without prediction progress after the
canary. Initialization has a separate 30-minute no-progress allowance.

Hydra authoring resolves once to a hashed JSON execution contract. The thin
entrypoint reuses the existing Qwen evaluator, preserves resume identities,
records GPU-memory samples, and owns a ten-minute process watchdog
with a bounded stall failure (not merely a log timestamp).
This is not an agent heartbeat: neither the watchdog nor the queue timer wakes
Codex to inspect the experiment. The user clarified that their requested
heartbeats mean agent-level review after startup, not these process timers.
No agent-wakeup scheduler is exposed in the current session; do not claim that
background agent monitoring has been enabled. Keep the watchdog as a separate
fail-closed process safeguard.

```bash
python -m experiments.qwen122b_id.run prepare
python -m experiments.qwen122b_id.run run
```

Artifacts: `results/qwen122b_id/`; logs: `logs/lambda/qwen122b_id/`.

The 122B run started successfully on 2026-09-06. The exact ID audit contains
37,077,317 prompt tokens (maximum 30,325). FP8/TP2 was verified at runtime;
weights used 57.83 GiB per GPU and the memory profiler retained 10.72 GiB per
GPU for KV cache. Cold initialization took 642.46 seconds, including compiler
and DeepGEMM warmup. The five-row balanced-plus-longest canary passed:
batch/singleton mean absolute probability difference 0.007415, maximum
0.026606. Full evaluation then started in the same engine. vLLM reported a
default, rather than shape-tuned, MoE kernel configuration, limiting claims
about best achievable throughput. These are startup findings, not final scores.

## Authorized sequential follow-up: Qwen3.8-27B FP8

### Bounded numerical recovery diagnostic

After the compiled Qwen3.8 canary failed, prepare one separate eager-serving
diagnostic (`qwen38_27b_eager.yaml`). This disables vLLM compilation/CUDA graphs,
not the optimized attention or GDN kernels. Keep model revision, precision,
prompt, batching, seed, and the exact five-row numerical gate unchanged. Run
only after the active Qwen3.5-27B evaluation releases the GPUs. If the canary
passes, finish the fixed ID endpoint; if it fails, stop this backend diagnostic
without repeated retries or loosened tolerances. Keep its artifacts separate
from the failed compiled run. A passing result would be backend-qualified, not
evidence that the original compiled configuration is reliable.

On 2026-09-06 the user added Qwen/Qwen3.8-27B-FP8 after the current 122B
evaluation. Revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a has
30,866,866,928 weight-file bytes. The installed vLLM 0.24.0 supports its
Qwen3_5ForConditionalGeneration architecture and explicit swish GDN output gate.
Its tokenizer/chat boundary and full ID token lengths are audited independently.

The additional Hydra config inherits the same FP8/TP2, full-rubric,
non-thinking, Triton, no-prefix-cache, batch/concurrency, metrics and canary
contract. Neither this model's quality nor its speed is assumed to exceed the
122B model. Its smaller weights should free more memory for KV cache, but this
fixed-concurrency screen is not a maximum-throughput or prefix-cache benchmark.
Compare both completed ID endpoints; no new annotation, training or OOD work.

Separate artifacts/logs are under `results/qwen38_27b_id/` and
`logs/lambda/qwen38_27b_id/`. The dependency gate checks predecessor success and
complete coverage, then waits for GPU release. It fails on predecessor failure
and never preempts the current run. Queued checks run every ten minutes.

```bash
python -m experiments.qwen122b_id.run prepare --config-name qwen38_27b
python -m experiments.qwen122b_id.run run \
  --result-dir results/qwen38_27b_id --after-result-dir results/qwen122b_id
```

## Matched 27B generation comparison: Qwen3.5-27B FP8

The user additionally authorized Qwen/Qwen3.5-27B-FP8 on the same ID set to
test whether Qwen3.8 regresses at the immediate one-token decision interface.
Revision 97f5941bf617e31c5e237364a8602ce3f03a551a has 30,866,882,312
weight-file bytes. The `qwen35_27b` Hydra config inherits the exact same
instruction, FP8/TP2, Triton, no-prefix-cache, batching, canary and metric
settings. Its independent tokenizer audit must pass. Run after the already
queued Qwen3.8 benchmark; no current process is interrupted or reordered.

Report paired ID source-macro/per-source ranking and calibration results for
the two 27B checkpoints. General benchmark strength does not establish this
interface's quality. No outcome-driven prompt tuning or automatic annotation
promotion is authorized. Separate artifacts/logs: `results/qwen35_27b_id/`
and `logs/lambda/qwen35_27b_id/`.

```bash
python -m experiments.qwen122b_id.run prepare --config-name qwen35_27b
python -m experiments.qwen122b_id.run run \
  --result-dir results/qwen35_27b_id --after-result-dir results/qwen38_27b_id
```
