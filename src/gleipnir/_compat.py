"""Lazy aliases for moved modules still referenced by frozen runners."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import CodeType, ModuleType

MODULE_ALIASES = {
    "gleipnir.qwen35_loftq": "gleipnir.training.qwen35_loftq",
    "gleipnir.openrouter": "gleipnir.teachers.openrouter",
    "gleipnir.openrouter_cli": "gleipnir.teachers.openrouter_cli",
    "gleipnir.prefix_cache": "gleipnir.teachers.prefix_cache",
    "gleipnir.prefix_audit": "gleipnir.teachers.prefix_audit",
    "gleipnir.judge_injection": "gleipnir.data.judge_injection",
    "gleipnir.prefix_boundaries": "gleipnir.data.prefix_boundaries",
    "gleipnir.prefix_sampling": "gleipnir.data.prefix_sampling",
    "gleipnir.campaign_status": "gleipnir.campaigns.status",
    "gleipnir.binary_evaluation": "gleipnir.evaluation.binary",
    "gleipnir.metrics": "gleipnir.evaluation.metrics",
    "gleipnir.calibration": "gleipnir.evaluation.calibration",
    "gleipnir.decision_surface": "gleipnir.evaluation.decision_surface",
    "gleipnir.evaluation_lanes": "gleipnir.evaluation.lanes",
    "gleipnir.evaluation_shards": "gleipnir.evaluation.shards",
    "gleipnir.evaluation_watchdog": "gleipnir.evaluation.watchdog",
    "gleipnir.judge_injection_metrics": "gleipnir.evaluation.preferences",
    "gleipnir.monitoring_scoring": "gleipnir.evaluation.scoring",
    "gleipnir.adaptive_microbatching": "gleipnir.training.adaptive_microbatching",
    "gleipnir.binary_task_training": "gleipnir.training.binary_tasks",
    "gleipnir.branch_model": "gleipnir.training.branch_model",
    "gleipnir.branch_trainer": "gleipnir.training.branch_trainer",
    "gleipnir.branch_training": "gleipnir.training.branches",
    "gleipnir.distributed_training": "gleipnir.training.distributed",
    "gleipnir.mil": "gleipnir.training.mil",
    "gleipnir.packed_training": "gleipnir.training.packed",
    "gleipnir.prefix_loss": "gleipnir.training.prefix_loss",
    "gleipnir.systems_artifacts": "gleipnir.training.artifacts",
    "gleipnir.training_execution_audit": "gleipnir.training.execution_audit",
    "gleipnir.branch_data": "gleipnir.data.branches",
    "gleipnir.monitoring_campaign_data": "gleipnir.data.monitoring",
    "gleipnir.monitoring_exclusions": "gleipnir.data.exclusions",
    "gleipnir.nested_subsets": "gleipnir.data.nested_subsets",
    "gleipnir.transcript_injection": "gleipnir.data.transcript_injection",
    "gleipnir.monitoring_campaign_training": "gleipnir.campaigns.training",
    "gleipnir.monitoring_systems_screen": "gleipnir.campaigns.systems_screen",
    "gleipnir.monitoring_training_command": "gleipnir.campaigns.training_command",
    "gleipnir.staged_lanes": "gleipnir.campaigns.lanes",
    "gleipnir.openai_monitor": "gleipnir.teachers.openai",
    "gleipnir.prompts": "gleipnir.teachers.prompts",
    "gleipnir.plotting": "gleipnir.analysis.plotting",
    "gleipnir.scaling": "gleipnir.analysis.scaling",
    "gleipnir.qwen35_adapter_rebase": "gleipnir.adapters.rebase",
    "gleipnir.cudnn_fp4_epilogue": "gleipnir.kernels.fp4.cudnn_fp4_epilogue",
    "gleipnir.cudnn_fp4_gdn": "gleipnir.kernels.fp4.cudnn_fp4_gdn",
    "gleipnir.cudnn_fp4_gemm": "gleipnir.kernels.fp4.cudnn_fp4_gemm",
    "gleipnir.cudnn_fp4_mlp": "gleipnir.kernels.fp4.cudnn_fp4_mlp",
    "gleipnir.cudnn_lora_mlp": "gleipnir.kernels.cudnn_lora_mlp",
    "gleipnir.fp32_projection": "gleipnir.kernels.fp32_projection",
    "gleipnir.fp4_compiler_diagnostic": "gleipnir.training.fp4.compiler_diagnostic",
    "gleipnir.fp4_compiler_ops": "gleipnir.training.fp4.compiler_ops",
    "gleipnir.fp4_fast_selector": "gleipnir.kernels.fp4.fp4_fast_selector",
    "gleipnir.fp4_memory": "gleipnir.training.fp4.memory",
    "gleipnir.fp4_performance": "gleipnir.training.fp4.performance",
    "gleipnir.fp4_quantization_kernels": (
        "gleipnir.kernels.fp4."
        "fp4_quantization_kernels"
    ),
    "gleipnir.fp4_row_kernels": "gleipnir.kernels.fp4.fp4_row_kernels",
    "gleipnir.grouped_gdn": "gleipnir.kernels.grouped_gdn",
    "gleipnir.inference_benchmark": "gleipnir.serving.benchmark",
    "gleipnir.merged_lora": "gleipnir.adapters.merge",
    "gleipnir.serving_bundle": "gleipnir.serving.bundle",
    "gleipnir.mlp_gemm": "gleipnir.kernels.mlp_gemm",
    "gleipnir.nvfp4_artifact": "gleipnir.kernels.fp4.nvfp4_artifact",
    "gleipnir.nvfp4_gptq": "gleipnir.kernels.fp4.nvfp4_gptq",
    "gleipnir.nvfp4_pack": "gleipnir.kernels.fp4.nvfp4_pack",
    "gleipnir.nvfp4_reference": "gleipnir.kernels.fp4.nvfp4_reference",
    "gleipnir.nvidia_causal_conv1d": "gleipnir.kernels.nvidia_causal_conv1d",
    "gleipnir.nvidia_mxfp8_attention": "gleipnir.kernels.mxfp8.nvidia_mxfp8_attention",
    "gleipnir.nvidia_mxfp8_fused_attention": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_fused_attention"
    ),
    "gleipnir.nvidia_mxfp8_fused_quantize": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_fused_quantize"
    ),
    "gleipnir.nvidia_mxfp8_meta_training": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_meta_training"
    ),
    "gleipnir.nvidia_mxfp8_meta_variants": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_meta_variants"
    ),
    "gleipnir.nvidia_mxfp8_norm_rope": "gleipnir.kernels.mxfp8.nvidia_mxfp8_norm_rope",
    "gleipnir.nvidia_mxfp8_norm_rope_kernel": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_norm_rope_kernel"
    ),
    "gleipnir.nvidia_mxfp8_projection_pilot": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_projection_pilot"
    ),
    "gleipnir.nvidia_mxfp8_varlen_attention": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_varlen_attention"
    ),
    "gleipnir.nvidia_mxfp8_varlen_host": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_varlen_host"
    ),
    "gleipnir.nvidia_mxfp8_varlen_quantize": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_varlen_quantize"
    ),
    "gleipnir.nvidia_mxfp8_varlen_repack": (
        "gleipnir.kernels.mxfp8."
        "nvidia_mxfp8_varlen_repack"
    ),
    "gleipnir.serving_attention_fp4": "gleipnir.serving.fp4.attention",
    "gleipnir.serving_cache_mirror": "gleipnir.serving.cache_mirror",
    "gleipnir.serving_compile_cache": "gleipnir.serving.compile_cache",
    "gleipnir.serving_cpu_placement": "gleipnir.serving.cpu_placement",
    "gleipnir.serving_fa4": "gleipnir.serving.fa4",
    "gleipnir.serving_fp4_backends": "gleipnir.serving.fp4.backends",
    "gleipnir.serving_fp4_cute_compat": "gleipnir.serving.fp4.cute_compat",
    "gleipnir.serving_fp4_fusion": "gleipnir.serving.fp4.fusion",
    "gleipnir.serving_fp4_integration": "gleipnir.serving.fp4.integration",
    "gleipnir.serving_fp4_prepare": "gleipnir.serving.fp4.prepare",
    "gleipnir.serving_fp4_swiglu": "gleipnir.serving.fp4.swiglu",
    "gleipnir.serving_fp4_swiglu_block_reference": (
        "gleipnir.serving.fp4."
        "swiglu_block_reference"
    ),
    "gleipnir.serving_fp4_swiglu_integration": (
        "gleipnir.serving.fp4."
        "swiglu_integration"
    ),
    "gleipnir.serving_fp4_swiglu_native_output": (
        "gleipnir.serving.fp4."
        "swiglu_native_output"
    ),
    "gleipnir.serving_fp4_swiglu_native_output_integration": (
        "gleipnir.serving.fp4."
        "swiglu_native_output_integration"
    ),
    "gleipnir.serving_fp4_swiglu_native_output_validation": (
        "gleipnir.serving.fp4."
        "swiglu_native_output_validation"
    ),
    "gleipnir.serving_fp4_swiglu_overhead": "gleipnir.serving.fp4.swiglu_overhead",
    "gleipnir.serving_fp4_swiglu_overhead_integration": (
        "gleipnir.serving.fp4."
        "swiglu_overhead_integration"
    ),
    "gleipnir.serving_fp4_swiglu_overhead_validation": (
        "gleipnir.serving.fp4."
        "swiglu_overhead_validation"
    ),
    "gleipnir.serving_fp4_swiglu_pack": "gleipnir.serving.fp4.swiglu_pack",
    "gleipnir.serving_fp4_swiglu_padding": "gleipnir.serving.fp4.swiglu_padding",
    "gleipnir.serving_fp4_swiglu_validation": "gleipnir.serving.fp4.swiglu_validation",
    "gleipnir.serving_fp4_tuning": "gleipnir.serving.fp4.tuning",
    "gleipnir.serving_fp4_tuning_validation": "gleipnir.serving.fp4.tuning_validation",
    "gleipnir.serving_frost_wrappers": "gleipnir.serving.frost_wrappers",
    "gleipnir.serving_gdn_direct_caller": "gleipnir.serving.gdn.direct_caller",
    "gleipnir.serving_gdn_direct_output": "gleipnir.serving.gdn.direct_output",
    "gleipnir.serving_gdn_kernels": "gleipnir.serving.gdn.kernels",
    "gleipnir.serving_gigatoken": "gleipnir.serving.gigatoken",
    "gleipnir.serving_mxfp8": "gleipnir.serving.mxfp8",
    "gleipnir.serving_mxfp8_source": "gleipnir.serving.mxfp8_source",
    "gleipnir.serving_operator_trace": "gleipnir.serving.operator_trace",
    "gleipnir.serving_precision": "gleipnir.serving.precision",
    "gleipnir.serving_prefill_graphs": "gleipnir.serving.prefill_graphs",
    "gleipnir.serving_prompt_only": "gleipnir.serving.prompt_only",
    "gleipnir.serving_prompt_only_contract": "gleipnir.serving.prompt_only_contract",
    "gleipnir.serving_reference": "gleipnir.serving.reference",
    "gleipnir.serving_runtime": "gleipnir.serving.runtime",
    "gleipnir.silu_fp8": "gleipnir.kernels.silu_fp8",
    "gleipnir.vllm_fp32_logits": "gleipnir.serving.vllm.fp32_logits",
    "gleipnir.vllm_frost_attention_fp4": "gleipnir.serving.vllm.frost_attention_fp4",
    "gleipnir.vllm_frost_fp4": "gleipnir.serving.vllm.frost_fp4",
    "gleipnir.vllm_frost_gdn": "gleipnir.serving.vllm.frost_gdn",
    "gleipnir.vllm_frost_gdn_fp4": "gleipnir.serving.vllm.frost_gdn_fp4",
    "gleipnir.vllm_mixed_fp8": "gleipnir.serving.vllm.mixed_fp8",
    "gleipnir.vllm_nvfp4": "gleipnir.serving.vllm.nvfp4",
    "gleipnir.vllm_online_nvfp4": "gleipnir.serving.vllm.online_nvfp4",
    "gleipnir.attention_backends": "gleipnir.training.backends.attention",
    "gleipnir.bf16_lora": "gleipnir.training.bf16_lora",
    "gleipnir.flashqla_training": "gleipnir.training.backends.flashqla",
    "gleipnir.fouroversix_training": "gleipnir.training.backends.fouroversix",
    "gleipnir.native_fp4_training": "gleipnir.training.backends.native_fp4",
    "gleipnir.qwen35_fast_training": "gleipnir.training.backends.qwen35",
    "gleipnir.packed_sequences": "gleipnir.training.packing",
    "gleipnir.training_hotpath": "gleipnir.training.hotpath",
    "gleipnir.validated_startup": "gleipnir.training.startup",
    "gleipnir.packed_benchmark": "gleipnir.training.screens.benchmark",
    "gleipnir.packed_training_screen": "gleipnir.training.screens.packed",
    "gleipnir.precision_training_screen": "gleipnir.training.screens.precision",
    "gleipnir.monitoring_campaign_runtime": "gleipnir.campaigns.runtime",
    "gleipnir.monitoring_campaign_evaluation": "gleipnir.evaluation.campaign",
}

_CLI_MODULES = {
    "gleipnir.openrouter_cli",
    "gleipnir.monitoring_systems_screen",
    "gleipnir.qwen35_adapter_rebase",
    "gleipnir.serving_bundle",
}


def canonical_source_reference(reference: str) -> str:
    """Translate a live relative source path from an archived source-list key.

    Use this when constructing new source snapshots, not when reading archived
    files or checking an old artifact's checksum. Unmoved references are exact.
    """
    for legacy, canonical in MODULE_ALIASES.items():
        if reference == "src/" + legacy.replace(".", "/") + ".py":
            return "src/" + canonical.replace(".", "/") + ".py"
    return reference


def _source_path(target: str) -> str:
    relative = target.removeprefix("gleipnir.").replace(".", "/") + ".py"
    return str(Path(__file__).parent / relative)


class _AliasLoader(Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def exec_module(self, module: ModuleType) -> None:
        """Reuse the canonical object without changing its import metadata."""
        sys.modules[module.__name__] = import_module(self.target)

    def get_code(self, fullname: str) -> CodeType:
        """Support historical command modules through ``python -m``."""
        source = (
            "from importlib import import_module\n"
            f"module = import_module({self.target!r})\n"
        )
        if fullname in _CLI_MODULES:
            source += "raise SystemExit(module.main())\n"
        return compile(source, _source_path(self.target), "exec")


class _AliasFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        destination = MODULE_ALIASES.get(fullname)
        if destination is None:
            return None
        return ModuleSpec(
            fullname,
            _AliasLoader(destination),
            origin=_source_path(destination),
        )


def install_aliases() -> None:
    """Register only these module names, without importing their dependencies."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
