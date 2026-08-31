# Qwen3.5-0.8B mixed-distillation systems preflight

## Outcome

The planned Qwen3.5-0.8B mixed-data soft-distillation and OOD campaign was
intentionally stopped on 2026-08-31. It did not produce a full training
checkpoint, and neither the trained condition nor the unadapted base was run on
the strict OOD suite. This record contains systems evidence only and must not be
used as a model-quality result.

The planned statistical intervention was the inherited one-epoch, seed-0,
rank-128 (`alpha=256`) Kimi-soft-only BCE recipe on all 21,837 mixed rows. Hard,
completion, rationale, direct-label, and pairwise losses were all zero. The
model was pinned to `Qwen/Qwen3.5-0.8B` revision
`2fc06364715b967f1860aea9cf38778875588b17`; effective batch remained 32 and
maximum length remained 29,696.

## Matched systems results

All three measurements used the same longest-32 selection, rank, effective
batch, maximum length, model revision, objective, and H100:

| Weight/layout path | Microbatch | Accumulation | Optimizer-step time | Observed memory | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| NF4 QLoRA | 8 | 4 | 189.54 s | about 80 GB | reject |
| NF4 QLoRA | 1 | 32 | 51.44 s warm | about 10 GB | fastest |
| BF16 LoRA | 1 | 32 | 75.78 s | about 8 GB | reject |

The two QLoRA layouts produced byte-identical FP32 master and serving adapter
hashes on the preflight, confirming that the microbatch change preserved the
effective update. Microbatch 8 was slower even on equally long examples, and
mixed-length padding made it still less attractive. BF16 LoRA was slower than
QLoRA, showing that reduced weight traffic outweighed the cost of 4-bit
dequantization on this training path.

The final QLoRA restart reached step 12 of 683 before the stop. Its step time
settled around 15--18 seconds, implying approximately 2.8--3.3 hours of GPU
training plus preprocessing. The completed 4B mixed campaign took 5.49 hours,
so the projected 0.8B advantage was only about 1.7--2.0x rather than anything
close to the parameter-count ratio.

## Decision

The campaign was stopped because Gleipnir 4B already dominates the observed
low-cost frontier, while the incremental 0.8B point was not worth several more
hours of training and evaluation under the measured stack. This is a portfolio
decision, not evidence that a 0.8B monitor would perform poorly.

Edge inference also should not be inferred from this training throughput.
Inference avoids backward passes, optimizer state, and activation
checkpointing, and can keep sub-gigabyte INT4 weights resident in a specialized
runtime. Long monitoring prompts remain a likely edge bottleneck, but whether
an edge-deployed action monitor is novel or practical requires a separate
literature and deployment study.

The ignored artifact tree contains `stop_record.json`, the raw launcher status,
and all systems logs. The queued OOD runner was stopped before it launched, and
there are no 0.8B OOD metrics.
