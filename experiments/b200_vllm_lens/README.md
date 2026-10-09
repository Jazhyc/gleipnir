# vLLM Lens for monitor scoring

Hypothesis: Lens 1.3.0's residual-stream capture and steering operators can be
used with the selected vLLM 0.31 pooling monitor through a request-scoped bridge.
The published package pins vLLM 0.30; the isolated, hashed requirements install
it without dependencies and explicitly retain our locked 0.31 runtime. Its
generation plugin is disabled. No NNsight or NDIF packages are installed.

Use the same merged adapter, FP8 attention projections, FP4 MLP/GDN, MXFP8
attention, native tokenizer/direct FROST, causal LAST two-row classifier and
stock corrected pooling scheduler. Research mode explicitly uses eager execution
and TP=PP=1. The optimized serving selection is unchanged. Keep prefix caching
off, 32K context/chunk and 128 engine sequences. Replace the identity-verified
existing scorer sequentially; preserve its artifacts and caches. Use only the
already authorized B200, without creating or terminating capacity.

The user explicitly permits eager execution for these research interventions.
Compare its adapter canary with the compiled selection, preserving a failed
0.005-MAE reproduction flag when eager arithmetic differs. This does not waive
or change the production selection's limit. Research operation requires finite
scores and a nonzero adapter effect; Lens no-op/zero-vector correctness is
measured against plain eager scoring. Reuse completed compiled controls on
retries with `--compiled-baseline <previous-run-directory>`.

Capture means the **post-decoder-layer residual stream**, including the residual
half of Qwen's fused `(hidden, residual)` return, before final model normalization.
Layers are zero-based. Return selected absolute input-token positions, including
`last` or `all`, in ascending order. Steering uses upstream `SteeringVector` and
its raw-addition/norm-match semantics; 3D vectors support absolute position
selection across chunk boundaries. Captures observe the state after steering.
Each request carries its own configuration in `PoolingParams.extra_kwargs`.
Never change global steering or reuse another request's state. Cleanup runs on
completion, errors and cancellation. Generic Python hooks, generation, Q/K
capture, multiple GPUs and persistent hooks are outside this first integration.

Freeze the smoke checks before launch: source/runtime/merged provenance; zero
and no-op capture parity against the same eager server; correct layers, shapes,
positions and finite values; nonzero steering and restoration of the next plain
request; mixed steered/plain request isolation; actual chunked capture; and no
retained request state. Malformed positions/vectors must fail before scheduling.
Stop on provenance/native/transport failure, nonfinite output, missing hook
coverage, isolation/cleanup failure, OOM or completion. No held-out quality
selection or claims about learned directions.

Measure one excluded warmup and three frozen quick64/c1 and full320/c128 passes
for plain eager scoring and final-layer/last-position capture. Reuse the current
NC2 compiled c128 controls; collect quick64/c1 from the existing warm compiled
server before replacement. Report input tokens/s, latency and ranking/score
changes for these training-seen systems cohorts. Arbitrary nonzero steering is a
functional test, not a candidate to promote on ID/OOD.

```bash
PYTHONPATH=src:. .venv-vllm031/bin/python -m experiments.b200_vllm_lens.stage
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_vllm_lens.run --name <run>
```

The stage helper installs only the hash-locked Lens/compression wheels into an
optional overlay and copies it to ephemeral storage. The run command configures
that overlay and `VLLM_LENS_DISABLE=1`, preserving the 0.31 environment.
For subsequent research launches without a benchmark, retire the verified
existing scorer with `experiments.b200_vllm031.stop --archive-name <receipt>`
and use the same runtime wrapper with `-m experiments.b200_vllm_lens.start
--name <startup>`. It verifies provenance and hook coverage and keeps the server
warm. The completed [integration finding](../../docs/findings/b200_vllm_lens.md)
records correctness, performance, failed receipts and eager numerical drift.

Use the client inside that overlay environment:

```python
from gleipnir.serving.lens import MonitorLensClient
from vllm_lens import SteeringVector

client = MonitorLensClient("http://127.0.0.1:8010")
result = client.score(rendered_prompt, capture_layers=[0, 3, 31])
h = result["activations"]["residual_stream"]  # (3, 1, 2560), native BF16
reference = client.score(contrast_prompt, capture_layers=[31])
direction = h[-1, 0] - reference["activations"]["residual_stream"][0, 0]
vector = SteeringVector(activations=direction[None, :], layer_indices=[31], scale=0.1)
steered = client.score(rendered_prompt, steering_vectors=[vector])
```

`POST /v1/monitor/lens` accepts the ordinary model/prompt plus `capture_layers`,
`capture_positions` (default `last`) and JSON-serialized `steering_vectors`.
The response includes ordinary scores, activation layer/position metadata and
Lens's compressed tensor wire format. The client decodes tensors automatically.
`GET /v1/monitor/lens/info` reports the supported shape and request-state counts.
Default capture storage is bounded to 512 MiB per request; select fewer layers
or positions for long prompts.

Run the eight CPU contract tests in the optional overlay environment with
`VLLM_LENS_DISABLE=1 PYTHONPATH=/tmp/gleipnir-vllm-lens-1.3.0:src:.
/tmp/gleipnir-vllm031-runtime/bin/python -m pytest -q
experiments/b200_vllm_lens/test_lens.py`. The main benchmark also checks the
public client, all-token capture, GPU norm matching and real disconnect cleanup.
Export tables and standalone figures with `python -m
experiments.b200_vllm_lens.summarize results/b200_vllm_lens/<run>`.

For the current SDPA-trained augmented adapter, use the separate
[BF16 Lens restoration contract](../b200_sdpa_lens/README.md) and its startup
entrypoint. It binds that adapter's master/merge/canary controls and preserves
unquantized BF16 eager serving; the selected-backbone command above targets
the regular optimized adapter.
