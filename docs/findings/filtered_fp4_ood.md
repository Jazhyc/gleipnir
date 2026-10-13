# Strict-filtered FP4/FA4 OOD evaluation

On 2026-10-13 the user authorized OOD scoring of the original strict-filtered
FP4 MLP/BF16 FlashQLA/FA4 adapter, selecting BF16 serving only. The frozen
[experiment contract](../../experiments/filtered_fp4_ood/README.md) evaluates
two fixed passes over all 6,395 unchanged examples against cached filtered
SDPA, released Gleipnir 4B and unfiltered references. Repeated use of this OOD
suite and missing assistant prose remain disclosed; there is no selection or
promotion on OOD results.

The FP32 master SHA256 is
`4eb97e5385f53b135c0b77cb92d0cbbfa78a661100f9ba8ac4205dbabb34f057`.
Its original FP32 inference export reconstructs the original BF16 merged model
with exact weight, config and tokenizer hashes. Three fresh serving canaries
and the BF16 native dispatch audit pass. Driver PID 12028 on the existing NC2
B200 is scoring examples; this is a launch receipt, not a completed quality
result. Artifacts live under `results/filtered-fp4-ood01/` and logs under
`logs/runpod/filtered-fp4-ood01/`. The pipeline retires its scorer on completion
and leaves capacity running. No in-chat scheduling tool is available.
