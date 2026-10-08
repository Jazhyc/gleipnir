# Disable direct FP4 SwiGLU output

Hypothesis: restoring BF16 SwiGLU activation materialization and whole-row
dynamic FP4 scaling recovers low-FPR ranking lost by direct packed output.
The user requests this single predeclared ablation on the existing B200.

Completed results and qualifications are in the
[output-ablation finding](../../docs/findings/b200_swiglu_output.md).

Keep the current FP8 full-attention / FP4 GDN and MLP-weight default, final
272-update FP4/FA4-trained adapter, merged BF16 checkpoint, vLLM 0.31,
MXFP8 attention core, native frontend/direct FROST, CUDA GDN/automatic context
parallelism, stock synchronous FCFS, prefix caching off and 32K/128 limits.
Small-row and medium-row arithmetic remain the same; above 4,096 physical
rows replace direct local-block FP4 output with the existing fused symbolic
GEMM/SwiGLU BF16 output and separate whole-row dynamic FP4 activation packer.
The down projection remains FP4. This disables direct epilogue FP4 output,
not FP4 activation quantization or MLP weights, and retains SwiGLU fusion.

Freeze the archived `b200_attention_precision/precision01/fp8` development
repeats and ID predictions as the current-default control. No unchanged
control rerun. Refresh native arithmetic/isolation/changed-input graph replay
for all nine whole-row producer cases through 32K on the current runtime,
with the unchanged 1% quantized-reference ceiling and a nonzero changed-row
effect. Require all 32 actual fused MLP calls and zero direct-output dispatch;
preserve unchanged attention/GDN/MLP/head/core/native identities and receipts.

After native/provenance admission, run the twenty-row canary, three quick64/c1
and six full320/c128 repeats with excluded warmups and fresh HTTP pools, then
one full ordered 128-row-group pass on the 3,012 CoT-removed ID examples.
The requested diagnostic may continue past finite score disagreement with
exact tokens and a nonzero adapter effect; preserve strict failed flags.
Report score drift, pooled/per-source/source-macro AUROC and raw normalized
pAUROC@20, calibration, ties, fixed-0.5 diagnostics, all repeats, input tokens/s,
requests/s and latency. No OOD, threshold fitting or ID-driven search.
Development diagnostic preference requires macro pAUROC gain >=0.005 with
no pooled decline; neither this rule nor the descriptive single ID pass
automatically changes the serving default.

Stop on provenance/native drift, unsupported dispatch, direct-output calls,
truncation, missing/nonfinite scores, zero adapter effect, startup/transport
failure, OOM or suite completion. Preserve failed receipts, partial passes,
capacity and persistent caches; retire only the owned candidate process.
No scheduling tool is available; monitoring continues during the active turn.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_swiglu_output.native \
  --output results/b200_swiglu_output/native01.json
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_swiglu_output.run --name output01 \
  --native-receipt results/b200_swiglu_output/native01.json
```
