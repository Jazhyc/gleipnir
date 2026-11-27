# Experiments

Each hypothesis has one folder, grouped by research area under
`experiments/<category>/<name>/`. Each folder contains a README with the
hypothesis, baselines, data provenance, frozen selection rule, commands, and
current status; configuration and entrypoints; and focused tests where practical.
Reusable code should graduate to `src/gleipnir/` instead of being imported from
another experiment. Experiment names are unique across categories, and
`results/<name>/`, `data/<name>/` and `logs/<platform>/<name>/` keep the bare
name.

| Category | Scope |
| --- | --- |
| [`training/`](training/) | Monitor training recipes, objectives, data and capacity scaling, and release candidates. |
| [`teachers/`](teachers/) | Teacher selection, agreement, filtering and teacher-student distillation sources. |
| [`evaluation/`](evaluation/) | Held-out, OOD and external-benchmark evaluation, audits and base-model screens. |
| [`adversarial/`](adversarial/) | Stress tests of the released monitors: attacks, cheating agents and external reward-hacking suites. |
| [`injection/`](injection/) | Prompt-injection robustness: detection, augmentation, injection directions, steering and training censuses. |
| [`probes/`](probes/) | Activation probes on monitor representations. |
| [`quantization/`](quantization/) | Quantization-aware training and distillation (QAT/QAD) and calibration. |
| [`systems/`](systems/) | Training throughput, kernels, precision and compilation on B200. |
| [`serving/`](serving/) | Inference and serving: kernels, precision, context length, scheduling and throughput. |

## Kept at their recorded paths

These experiments stay directly under `experiments/` because other files check
their exact bytes and paths, and some of them locate the checkout from their
own depth:

- Serving stack. Its runtime (servers, workers, scheduler) now lives in
  [`gleipnir.serving.stack`](../src/gleipnir/serving/stack/__init__.py). These
  directories keep the canaries, comparison scripts, selections and frozen
  controls that kernel receipts and the accepted serving default bind by path
  and SHA-256, and several of those scripts locate the checkout from their own
  depth, so they move only with fresh receipts: [`b200_attention_gdn_serving`](b200_attention_gdn_serving/), [`b200_attention_precision`](b200_attention_precision/), [`b200_context_million`](b200_context_million/), [`b200_frost_inference`](b200_frost_inference/), [`b200_gdn_fast`](b200_gdn_fast/), [`b200_gdn_route`](b200_gdn_route/), [`b200_inference_benchmark`](b200_inference_benchmark/), [`b200_last_layer_trim`](b200_last_layer_trim/), [`b200_monitor_score`](b200_monitor_score/), [`b200_mutation_analysis`](b200_mutation_analysis/), [`b200_prefix_replay`](b200_prefix_replay/), [`b200_vllm031`](b200_vllm031/), [`release_serving`](release_serving/).
- [`fp4_stability`](fp4_stability/): the FlashQLA training backend checks
  `patch_tilelang_fp16.py` against the installed kernel manifest, and the
  script finds that install from its own depth.
- [`quantization_recovery`](quantization_recovery/): its `train.py`,
  `reference.py` and `evaluate.py` are text templates that the QAT/QAD
  generators in `quantization/` rewrite by exact substitution.

## Historical names

Directories were grouped on 2026-11-27. The old flat names keep working:
`experiments.<name>` imports and `python -m experiments.<name>.<module>`
resolve through `EXPERIMENT_MOVES` in
[`src/gleipnir/_compat.py`](../src/gleipnir/_compat.py), and recorded
`experiments/<name>/...` source paths translate through
`canonical_source_reference`. Archived receipts keep their original paths and
hashes; code-bound historical runs reproduce from their recorded revision.
