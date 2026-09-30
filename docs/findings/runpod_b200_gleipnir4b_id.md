# Runpod B200: released Gleipnir 4B ID verification

Date: 2026-09-30. Execution contract:
[`runpod_gleipnir4b_id`](../../experiments/runpod_gleipnir4b_id/README.md).
This verifies infrastructure and a frozen checkpoint; it does not select a
new model or establish a B200 training-throughput recipe.

## Hardware and environment

One NVIDIA B200 (183,359 MiB, compute capability 10.0), driver 580.126.09,
CUDA 13.0, Python 3.12.3, PyTorch 2.11.0+cu130, Transformers 5.14.1, and
vLLM 0.24.0. The checked-in lock SHA-256 is
`6aaa85d882a2fddaf276a76463be6edaa6b08f5f3dac62e65a4e4f64db68a56d`.
Pod `alzfug70g5237b` in US-NC-2 costs $6.79/hour. The independent 100 GB
standard network volume `ixbh81vf9c` costs $7/month, and the 50 GB running
container disk costs $5/month. The user requested leaving the Pod running.

All runtime inputs, adapters, environment, and caches are under
`/workspace/gleipnir`. The Runpod API key stays local. Network storage rejects
ownership changes, so file transfers disable rsync owner/group preservation.

## Startup findings

The pinned causal-conv1d 1.6.2.post1 source build took 12m43s and is cached.
It supports the bounded Transformers master-parity path; full ID inference
uses vLLM. BF16 forward/backward passed with FLA 0.5.2, causal-conv1d, and
Triton 3.7.1, with both Transformers fast paths bound. This is a kernel
canary, not a training benchmark.

The first vLLM launch detected mixed files from the overlapping CUTLASS base
and CUDA-13 wheels and selected Triton/FLA GDN prefill. That attempt was stopped
before full evaluation. Reinstalling the locked CUDA-13 wheel 4.5.2 last
restored complete RECORD checksum agreement. Bootstrap now performs this
repair and checks integrity. The restarted engine confirmed active FlashInfer
GDN prefill and FlashInfer attention with TRT-LLM prefill. The runner rejects
a serving canary without explicit FlashInfer GDN activation.

The upstream packaging conflict is documented in
[CUTLASS issue 3170](https://github.com/NVIDIA/cutlass/issues/3170).
Preserved diagnostics are in
`logs/runpod/runpod_gleipnir4b_id/pre_repair/`; successful-run logs use its
parent directory. vLLM compilation caches were moved to persistent storage;
the restart reused them (22.99s versus 110.30s for model compilation).

## Master and serving parity

The existing balanced four-row canary passed the frozen gate: correlation
at least 0.99, mean absolute score difference at most 0.02, and nonzero
adapter effect in both backends.

| Comparison | Mean absolute difference | Maximum difference | Correlation |
| --- | ---: | ---: | ---: |
| Base: eager vs vLLM | 0.01391193 | 0.02782386 | 0.99999947 |
| Adapter: FP32 master vs serving | 0.00231023 | 0.00924085 | 0.99993766 |

Maximum adapter effect was 0.06919265 in eager and 0.05995180 in vLLM.
The gate constrains mean error, not maximum error. This bounded check supports
serving the released artifact; whole-set agreement is assessed separately.

Artifacts: `results/runpod_gleipnir4b_id/hardware.json` and
`results/runpod_gleipnir4b_id/parity/report.json`.

## Frozen evaluation population

All 3,012 canonical CoT-removed ID examples: 946 TestStride and 2,066 Gloom
exfiltration. Visible assistant prose is preserved. No OOD population was
accessed, and no checkpoint, threshold, or prompt was selected on these scores.

The compact student prompt, literal decision tokens `0`/`1`, one-token
generation, raw decision logprobs, and threshold 0.5 are unchanged. The serving
engine uses BF16, TP=1, length 32,768, at most 16 sequences, 32,768 batched
tokens, prefix caching, rank-128 LoRA, and batch dispatch of 128 rows. GPU
memory utilization is 0.5 rather than the historical 0.9; other engine
settings are preserved.

Input SHA-256:
`7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4`.
Manifest SHA-256:
`abcc8267a77f6eff0a252ea846a6a59283e8a027cc860210e3984064eec611f1`.
Configuration SHA-256:
`aa00d2089c215d3f1a115046ad92ad04e196899ac7b48c628f416d83967c1165`.
The pinned base revision and master/serving adapter checksums are in the
configuration and preparation record. The historical predictions checksum is
`6cc4e931de18bbb40ec5846d8608077f536a0d3ccf09e5976ef4618f389f71e0`.

## Completed ID results

The evaluation completed on its first full-set attempt, without watchdog
restarts. Coverage is exactly 3,012 unique identities. Every row has finite
scores and both requested raw decision logprobs. Source/label membership,
source and rendered prompt hashes, token counts, and decision-token IDs match
the saved historical baseline. The maximum prompt length is 29,513 tokens,
below the 32,768 limit; there was no truncation.

| Population | AUROC | pAUROC@20 | Brier | Balanced accuracy | FPR | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TestStride, n=946 | 0.983388 | 0.929272 | 0.056497 | 0.933480 | 0.032520 | 0.899480 |
| Gloom, n=2,066 | 0.922655 | 0.773619 | 0.116340 | 0.843955 | 0.058937 | 0.746848 |
| Unweighted source macro | 0.953021 | 0.851445 | 0.086418 | 0.888718 | 0.045729 | 0.823164 |
| Pooled | 0.947311 | 0.827382 | 0.097545 | 0.874811 | 0.051994 | 0.801617 |
| Historical source macro | 0.952674 | 0.850083 | 0.086675 | 0.889774 | 0.045246 | 0.824794 |

The whole-set correlation with historical scores is 0.99976462, mean absolute
difference 0.00332625, and maximum difference 0.05868826. Ten of 3,012 fixed
threshold decisions differ. Small ranking changes are backend numerical
variation, not evidence of model improvement; balanced accuracy and recall
are slightly lower. The results support serving transfer, not bitwise parity.

There are 1,580 unique scores and 1,432 rows beyond the first occurrence of
each score (the shared evaluator's tied-row diagnostic). Ten-bin ECE is
0.056658 for TestStride, 0.085552 for Gloom, and 0.076477 pooled. Probabilities
remain downward biased relative to prevalence. The summary preserves complete
5-, 10-, and 20-bin calibration diagnostics; no calibration was fitted.

The scoring loop took 662.09 seconds (11m02s), including prompt rendering and
batch dispatch, excluding model loading and engine initialization. That is
4.55 examples/s or 50,976 nominal prompt tokens/s over 33,750,959 tokens.
Prefix caching is enabled, so nominal tokens/s is not uncached prefill
throughput. The restarted campaign including kernel and master/serving checks
took 33m07s. Cold setup, the initial stopped attempt, and its repair are
additional costs. Peak observed serving allocation was about 96,746 MiB;
the final GPU check reported zero volatile uncorrected ECC errors.

## Artifacts and operational state

Results and diagnostic logs were collected locally and retained on the
network volume. Local summary includes whole-set historical score agreement;
`audit.json` records exact identity and prompt-contract checks. Final artifacts:

- `results/runpod_gleipnir4b_id/summary.json`
- `results/runpod_gleipnir4b_id/audit.json`
- `results/runpod_gleipnir4b_id/status.json` (`complete`)
- `results/runpod_gleipnir4b_id/evaluation/4b/adapters/soft-n21837-mixed-qwen35-4b-seed0/`
- `logs/runpod/runpod_gleipnir4b_id/`

Predictions SHA-256:
`b27694e3914e521b630835dc8b460f14bec3fa64fd13c388a3836268b4a157d4`.
Local summary SHA-256:
`62f3b7aabb64dbd5abbd2cf0237602b6e31c84923e58a4a526c317c93f09283c`.

The Pod remains RUNNING as requested. Compute is $6.79/hour; the combined
published running storage charges are $12/month (approximately $6.81/hour
overall when amortized over a month). No API server was requested or started.
