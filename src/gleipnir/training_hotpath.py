"""Scoped metadata and normalization-copy optimizations for packed training."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch

from gleipnir.packed_sequences import PackedSequenceLayout


def prepared_kernel_kwargs(
    layout: PackedSequenceLayout,
    device: torch.device | str,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Construct validated boundaries on CPU and transfer without device reads."""
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")
    offsets_cpu = torch.tensor(layout.offsets, dtype=torch.int32)
    cumulative = offsets_cpu.to(device, non_blocking=True)
    positions = torch.cat([torch.arange(n) for n in layout.lengths]).unsqueeze(0)
    sequence_ids = torch.repeat_interleave(
        torch.arange(len(layout.lengths), dtype=torch.int32),
        torch.tensor(layout.lengths),
        output_size=layout.total_tokens,
    ).unsqueeze(0)
    counts = [(n + chunk_size - 1) // chunk_size for n in layout.lengths]
    chunk_offsets_cpu = torch.tensor(
        PackedSequenceLayout(tuple(counts)).offsets, dtype=torch.int32
    )
    chunk_indices_cpu = torch.tensor(
        [(i, j) for i, count in enumerate(counts) for j in range(count)],
        dtype=torch.int32,
    )
    cumulative._flash_qla_prepared_varlen = (
        chunk_size,
        chunk_offsets_cpu.to(device, non_blocking=True),
        sum(counts),
        chunk_indices_cpu.to(device, non_blocking=True),
    )
    # The tensor version changes on in-place mutation, including through aliases.
    cumulative._gleipnir_validated_layout = (cumulative._version, layout)
    return {
        "position_ids": positions.to(device, non_blocking=True),
        "seq_idx": sequence_ids.to(device, non_blocking=True),
        "cu_seq_lens_q": cumulative,
        "cu_seq_lens_k": cumulative,
        "max_length_q": max(layout.lengths),
        "max_length_k": max(layout.lengths),
        "use_cache": False,
    }


def prepared_fa4_interface(original: Callable, kernel: Callable) -> Callable:
    """Skip repeated GPU boundary reads only for the validated shared tensor."""

    def attention(module, query, key, value, attention_mask, **kwargs):
        cumulative = kwargs.get("cu_seq_lens_q")
        metadata = getattr(cumulative, "_gleipnir_validated_layout", None)
        if (
            metadata is None
            or cumulative is not kwargs.get("cu_seq_lens_k")
            or metadata[0] != cumulative._version
        ):
            return original(module, query, key, value, attention_mask, **kwargs)
        layout = metadata[1]
        if (
            query.shape[0] != 1
            or key.shape[0] != 1
            or value.shape[0] != 1
            or attention_mask is not None
            or query.shape[2] != layout.total_tokens
            or key.shape[2] != layout.total_tokens
            or value.shape[2] != layout.total_tokens
            or cumulative.dtype != torch.int32
            or cumulative.device != query.device
            or kwargs.get("max_length_q") != max(layout.lengths)
            or kwargs.get("max_length_k") != max(layout.lengths)
        ):
            raise ValueError("prepared FA4 layout disagrees with attention operands")
        if kwargs.get("dropout", 0) != 0:
            raise ValueError("packed FA4 supports zero attention dropout only")
        result = kernel(
            query[0].transpose(0, 1),
            key[0].transpose(0, 1),
            value[0].transpose(0, 1),
            cu_seqlens_q=cumulative,
            cu_seqlens_k=cumulative,
            max_seqlen_q=kwargs["max_length_q"],
            max_seqlen_k=kwargs["max_length_k"],
            softmax_scale=kwargs.get("scaling"),
            causal=True,
        )
        output = result[0] if isinstance(result, tuple) else result
        return output.unsqueeze(0), None

    return attention


def normalize_qk_without_input_copy(x: torch.Tensor) -> torch.Tensor:
    """Let the existing FLA kernel load BF16 into FP32 arithmetic and output."""
    from fla.modules.l2norm import l2norm

    return l2norm(x, output_dtype=torch.float32)


@contextmanager
def nonblocking_trainer_inputs(trainer: Any) -> Iterator[None]:
    """Enqueue single-device input copies on the existing stream without waits."""
    if trainer.is_deepspeed_enabled:
        raise ValueError("nonblocking input screen does not support DeepSpeed")
    original = trainer._prepare_input
    instance_original = trainer.__dict__.get("_prepare_input")
    had_instance = "_prepare_input" in trainer.__dict__

    def prepare(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.args.device, non_blocking=True)
        # Trainer's recursion calls the scoped method for nested tensor values.
        return original(data)

    trainer._prepare_input = MethodType(prepare, trainer)
    try:
        yield
    finally:
        if had_instance:
            trainer._prepare_input = instance_original
        else:
            del trainer._prepare_input


@contextmanager
def matched_normalization_configs(autotuner: Any) -> Iterator[list[dict[str, Any]]]:
    """Keep the FP32 baseline's reduction tiling when loading BF16 inputs."""
    fp32 = str(torch.float32)
    bf16 = str(torch.bfloat16)
    previous = {}
    selected = []
    for key, config in list(autotuner.cache.items()):
        if len(key) != 5 or key[2:] != (fp32, fp32, fp32):
            continue
        target = (*key[:2], bf16, fp32, fp32)
        previous[target] = (target in autotuner.cache, autotuner.cache.get(target))
        autotuner.cache[target] = config
        selected.append(
            {"dimension": key[0], "nb": key[1], "config": str(config)}
        )
    if not selected:
        raise ValueError("resident FP32 normalization launch configs missing")
    try:
        yield selected
    finally:
        for key, (existed, config) in previous.items():
            if existed:
                autotuner.cache[key] = config
            else:
                autotuner.cache.pop(key, None)


@contextmanager
def training_hotpath_context(
    *,
    normalize: bool = False,
    match_norm_configs: bool = False,
    prepare_metadata: bool = True,
) -> Iterator[dict[str, Any]]:
    """Restore packing/router/normalization functions after a resident trial."""
    from flash_attn.cute import flash_attn_varlen_func
    from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import gleipnir.flashqla_training as boundary

    previous_kwargs = PackedSequenceLayout.kernel_kwargs
    previous_router = ALL_ATTENTION_FUNCTIONS["sdpa"]
    previous_norm = boundary._normalize_qk_fp32
    from contextlib import ExitStack

    if not getattr(previous_router, "_gleipnir_packed_boundaries", False):
        raise ValueError("hotpath screen requires an installed packed FA4 router")

    def kwargs(layout, device="cpu"):
        return prepared_kernel_kwargs(layout, device, chunk_size=CHUNK_SIZE)

    router = torch.compiler.disable(
        prepared_fa4_interface(previous_router, flash_attn_varlen_func)
    )
    router._gleipnir_packed_boundaries = True
    try:
        if prepare_metadata:
            PackedSequenceLayout.kernel_kwargs = kwargs
            ALL_ATTENTION_FUNCTIONS.register("sdpa", router)
        if normalize:
            boundary._normalize_qk_fp32 = normalize_qk_without_input_copy
        with ExitStack() as stack:
            configs = []
            if normalize and match_norm_configs:
                from fla.modules.l2norm import l2norm_fwd_kernel

                configs = stack.enter_context(
                    matched_normalization_configs(l2norm_fwd_kernel)
                )
            yield {
                "cpu_prepared_packing": prepare_metadata,
                "flashqla_chunk_size": CHUNK_SIZE,
                "fa4_repeated_boundary_reads_removed": prepare_metadata,
                "normalization_input_copy_removed": normalize,
                "normalization_arithmetic_and_output": "float32",
                "normalization_configs": configs,
                "master_parameter_dtype": "float32",
            }
    finally:
        PackedSequenceLayout.kernel_kwargs = previous_kwargs
        ALL_ATTENTION_FUNCTIONS.register("sdpa", previous_router)
        boundary._normalize_qk_fp32 = previous_norm
