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
Publication runs locally using `HF_TOKEN` from ignored `.env`, after artifact
collection. Repeat the same commands with `--size 9b` after the 4B stage.

Systems receipts and eventual quality results are recorded in
[the campaign finding](../../docs/findings/student_injection_awareness.md).
