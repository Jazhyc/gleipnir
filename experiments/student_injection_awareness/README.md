# Student prompt-injection awareness

Hypothesis: explicitly warning a monitor about collusion and misleading claims
improves transfer when its teacher signal stays fixed. Train matched regular and
injection-aware students at 4B, then 9B. This intervention changes the student
instruction in both training and evaluation; it does not isolate training from
test-time elicitation. Keep every trajectory, label and cached Kimi K3 soft
target identical within each pair. No new teacher requests or deception data.

Use all 8,688 frozen tool trajectories for one epoch, seed 0, rank 128/alpha 256,
AdamW 5e-5, linear decay, 3% warmup, zero weight decay and dropout, logical batch
32 and soft BCE only. The published mixed-data Gleipnir models are historical
references, not matched controls. The regular instruction is the matched control.
Use the selected packed BF16 B200 profile, FP32 masters and selected-token
projection. 4B starts without checkpointing; 9B uses checkpointing in all layers
because its larger base/activations need a separate longest-batch preflight.
Use nonreentrant checkpointing and the supported checkpointed shell compilation
policy for both 9B conditions; the uncheckpointed compile policy rejects
checkpointed full-attention layers.
The packed-runtime guard also requires both gradient checkpointing and
nonreentrant checkpointing when accepting this compilation policy. Both
conditions still run the strict eager/compiled packing canaries and longest-row
memory preflight before any optimizer update.
Preserve strict eager/compiled packing isolation gates, FlashQLA finite-policy
receipts, compiler cache limits and finite-gradient checks. Do not shrink the
token budget or truncate examples after a failed gate.

Freeze the final one-epoch checkpoint for every condition. There is no checkpoint
or prompt selection on ID or OOD. Evaluate all 3,012 canonical CoT-removed ID rows
and all 6,395 strict six-source OOD rows under each model's training instruction.
Keep original source-grouped holdouts and provenance. The ID input transformation
does not alter reasoning-bearing training trajectories or teacher targets.
Report source-macro pAUROC@20 as primary, AUROC, calibration, threshold diagnostics
and ties, with per-source results. Require causal-master/vLLM score correlation
>=0.99, mean absolute difference <=0.02, and nonzero adapter effects before full
evaluation. The bounded parity cohort is drawn from training data, never OOD.

Stop on input/target/hash drift, missing targets, failed isolation/parity,
nonfinite values, OOM, truncation or incomplete evaluation. Preserve failed
receipts. Upload each successfully evaluated FP32 master and checked serving
artifact with its exact instruction, prompt contract and result provenance;
record upload failure and continue to the next model size as instructed.

Artifacts: `data/student_injection_awareness/`,
`results/student_injection_awareness/`; logs:
`logs/runpod/student_injection_awareness/`.

Prepare locally, sync code/data to the existing authorized B200, then run:

```bash
.venv/bin/python -m experiments.student_injection_awareness.prepare
.venv/bin/python -m experiments.student_injection_awareness.train --size 4b --variant regular
.venv/bin/python -m experiments.student_injection_awareness.train --size 4b --variant injection_aware
.venv/bin/python -m experiments.student_injection_awareness.evaluate --size 4b --backend reference
VLLM_CACHE_ROOT="$PWD/.cache/vllm/student_injection_awareness_v1" \
TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor/student_injection_awareness_v1" \
.venv/bin/python -m experiments.student_injection_awareness.evaluate --size 4b --backend vllm
.venv/bin/python -m experiments.student_injection_awareness.publish --size 4b
```

The campaign prompt variants are independent of the historical teacher/student
manifest. Their prompt-set IDs and hashes are recorded in the preparation
manifest. Canonical source files and teacher identities stay frozen.

Run reference canaries in the pinned FLA/convolution environment; run vLLM in
the serving environment, without the isolated training Triton on its import
path. The evaluator holds one persistent engine for both adapters and both
splits, retaining exact prediction and artifact identities for safe resumption.
Use the campaign compiler cache directories above: copied vLLM compiler
artifacts can retain absolute paths from their original cache root and fail
to load after relocation. Keep the failed logs when recovering with fresh
cache directories.
Publication runs locally using `HF_TOKEN` from ignored `.env`, after artifact
collection. Repeat the same commands with `--size 9b` after the 4B stage.
After an observed FlashInfer serving stall, the 9B evaluation recovery uses
`--gdn-prefill-backend triton`. Archive the previous evaluation outputs before
changing this backend and rerun both conditions and both complete suites. The
backend is recorded in parity and runtime receipts and changes prediction-cache
identity; the frozen inputs, BF16 weights, batching, digit scoring and parity
limits remain the same. A backend override does not change training or targets.
Public uploads use an explicit allowlist: the two adapter layouts and loading
configs, license and model card (including the student instruction). Full
training metadata, provenance, evaluation reports, parity receipts and release
manifests remain local. Remote revision, exact file coverage and both adapter
LFS checksums are verified before recording successful publication.
Add `--variant regular` (or `--variant injection_aware`) to the publisher
command to publish a condition as soon as its own full audits finish while
the other evaluates.
Omitting `--variant` publishes both conditions.

Systems receipts and eventual quality results are recorded in
[the campaign finding](../../docs/findings/student_injection_awareness.md).


## Completed campaign

All four models completed one epoch over 8,688 trajectories with the same
cached targets and learning rate 5e-5. Every condition completed all 3,012 ID
and 6,395 OOD examples without truncation. Both model pairs passed serving
score parity; all four public adapter releases were independently verified.
The values below are equal-source macro pAUROC@20, the frozen primary metric.

| Public adapter | ID | OOD | Training-loop runtime |
| --- | ---: | ---: | ---: |
| [4B regular](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-Regular) | 0.846273 | 0.776238 | 1h 5m 16s |
| [4B aware](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-InjectionAware) | 0.847301 | 0.775323 | 57m 51s |
| [9B regular](https://huggingface.co/Jazhyc/Gleipnir-9B-ToolTrajectories-Regular) | 0.879559 | 0.837368 | 1h 43m 50s |
| [9B aware](https://huggingface.co/Jazhyc/Gleipnir-9B-ToolTrajectories-InjectionAware) | 0.896654 | 0.822471 | 1h 45m 18s |

Awareness left 4B primary scores nearly unchanged and worsened calibration.
At 9B it improved ID primary pAUROC by 0.017096 and reduced OOD primary
pAUROC by 0.014897. Source effects are mixed. This is one seed and changes the
instruction in both training and evaluation; it does not establish isolated
training-time resistance or robustness against deliberate collusion attacks.
The separate failed strict training-kernel gradient comparisons remain recorded
under the user-selected finite-gradient acceptance recipe. All mandatory
packing and serving gates passed. 4B used FlashInfer GDN serving; both 9B
conditions used Triton after the documented runtime stall, limiting pure
backbone-size attribution across sizes. No promotion or threshold tuning was
performed. Full metrics, source calibration, ties, hardware and failure receipts
are recorded in the campaign finding and ignored local result artifacts.
