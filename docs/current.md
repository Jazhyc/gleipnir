# Current state

Status: current; update this page whenever a selection, release or open item
changes. Date: 2026-11-27.

What is in force now. Each line links to the document that owns the detail;
read that before acting on it. Historical notes are not a live compute
inventory: check the provider before assuming a machine or volume exists.

## Released models

- **Gleipnir 4B v1.0 and 27B v1.0** on Hugging Face (adapters and merged BF16
  checkpoints; [root README](../README.md#released-models)). Same weights as
  4B v0.4 (the [three-seed average](../experiments/training/seed_average/README.md))
  and 27B v0.2. Mean-OOD pAUROC@20: 4B 0.804 BF16 / 0.770 optimized; 27B
  0.935 BF16 / 0.951 optimized, scored under the full teacher prompt
  ([decision](decisions/evaluation/evaluate_gleipnir_27b_under_the_full_teacher_prompt.md)).
- Gleipnir 9B is superseded by Gleipnir 4B.
- The technical report's PDF is on the
  [project page](https://jazhyc.github.io/gleipnir-monitor/); it has no arXiv
  identifier yet.

## Evaluation contract

- [Research program](research_program.md#evaluation-contract) owns the
  contract: Mean-OOD pAUROC@20 on the frozen strict-OOD suite
  ([frozen inputs](decisions/evaluation/frozen_ood_comparability.md)) and the
  [CoT-removed ID set](decisions/evaluation/cot_removed_id_validation.md).
- Precision and kernel choices may select on the OOD suite under frozen rules
  ([decision](decisions/evaluation/precision_selection_on_ood.md)).
- Serving benchmarks use the 96 agent trajectories at 8 concurrent requests.

## Training

- Data: tool-trajectory rows only
  ([decision](decisions/training/train_gleipnir_4b_on_tool_trajectory_data_only.md)),
  Kimi K3 soft targets at temperature 1
  ([finding](findings/training/distillation_temperature.md)), with the
  [strict teacher-agreement filter](decisions/teachers/strict_teacher_filtering.md).
- Qwen3.5-4B recipe: `qwen35_4b_b200_default`, BF16/SDPA with the final-layer
  trim ([decision](decisions/systems/b200_bf16_sdpa_training_default.md)).
- Qwen3.8-27B recipe: `qwen38_27b_b200_default`
  ([decision](decisions/systems/qwen38_27b_b200_training_default.md)).
- Instructions: [training guide](agent_guides/training.md).

## Serving

- Two paths ([decision](decisions/serving/b200_monitor_score_reference.md#current-recipe)):
  stock vLLM BF16 for regular evaluation and adapter iteration; the optimized
  B200 recipe (FP8 attention projections, FP4 MLP/GDN, MXFP8 attention,
  [last-query final layer](decisions/serving/adopt_last_query_final_layer_attention_in_the_fp8_default.md),
  [GDN route gate](decisions/serving/gate_the_gdn_context_parallel_prefill_route_by_length.md)
  and [lean GDN host path](decisions/serving/adopt_the_lean_gdn_host_path_with_a_strided_gated_norm.md))
  only for QAD-adapted deployment candidates.
- The optimized recipe is bound by path and SHA-256 in
  [`serving_default.json`](../experiments/b200_inference_benchmark/serving_default.json)
  and resolved by `gleipnir.serving.reference.selected_serving_default`. Run it
  and `selected_score_reference` locally before editing any serving source; both
  only hash files.
- Prefix-cached serving is opt-in
  ([decision](decisions/serving/prefix_cached_serving.md)).
- The released stack ships as the `gleipnir-serve` image built by
  [`release_serving`](../experiments/release_serving/README.md).
- Instructions: [inference guide](agent_guides/inference.md).

## Standing policies

- Train text-only LoRAs causally, keep the FP32 master and export a
  checksum-tracked serving copy
  ([decision](decisions/training/train_text_only_loras_causally_and_export_serving_copies.md)).
- Frozen text evaluation runs on vLLM with literal decision-token logprobs
  ([decision](decisions/evaluation/default_frozen_text_evaluation_to_vllm.md)).
- Paid teacher caching is provider-sticky and auditable
  ([decision](decisions/teachers/make_paid_teacher_caching_provider_sticky_and_auditable.md)).
- Reuse warm compiler and kernel caches across compatible training runs
  ([decision](decisions/systems/persistent_training_compiler_cache.md)); short
  systems checks keep only their latest weights
  ([decision](decisions/systems/systems_training_adapter_retention.md)).
- New training-throughput screens are config-driven
  ([decision](decisions/systems/config_driven_systems_screens.md)).
- Future merged-weight QAT uses the selected reference arithmetic
  ([decision](decisions/quantization/qat_reference_training.md)); the 27B needs
  no QAD ([decision](decisions/evaluation/plot_gleipnir_at_optimized_serving_performance_no_qad_for.md)).
- Release models are priced at self-hosted B200 break-even from the released
  image's one-request throughput
  ([decision](decisions/serving/price_release_models_on_the_released_stack.md)).

## Code bound in place

The optimized serving runtime (servers, worker classes, scheduler) lives in
[`gleipnir.serving.stack`](../src/gleipnir/serving/stack/__init__.py), promoted
from its experiments on 2026-11-27. Accepted recipes still name the experiment
modules and source hashes they were validated with;
[`promotion.json`](../src/gleipnir/serving/stack/promotion.json) maps each to its
promoted path and hash, and `gleipnir.serving.sources` checks and applies it at
launch. The promotion is accepted
([decision](decisions/serving/promote_serving_stack.md)): a B200 check reproduced
`main`'s scores bit for bit for both models. Fifteen experiment directories still stay at
flat paths because kernel receipts or text-substitution generators bind their
remaining files; the
[experiments index](../experiments/README.md#kept-at-their-recorded-paths) lists
them.

## Next work

- **Base model:** Inkling 2 Mini is the selected successor to Qwen3.8-27B
  ([decision](decisions/evaluation/inkling2_mini_next_base.md)); its vLLM
  batching and training support are still open.

## Open items

- Build a new `gleipnir-serve` image from the promoted stack.
- Released adapters never adapted the Gated DeltaNet projections. Include them,
  or ablate them, in future training.
- SWE-chat deployment thresholds (1% and 0.1% FPR on 16,625 sessions) are
  deferred for budget.
- Add the arXiv identifier and citation once the report is announced.
- Long findings may later be condensed.
- Runpod volumes: EU `cb7bxu71ug` holds nothing the Storage Box lacks that the
  user chose to keep ([inventory](findings/operations/eu_volume_inventory.md));
  deleting it is the user's call. NC2 `ixbh81vf9c` (300 GB) is not yet
  inventoried.
