"""Cache-free causal monitoring on the pinned vLLM/FROST/MXFP8 stack.

EncoderOnlyAttentionSpec is used solely as vLLM's zero-storage runner group.
The numerical kernel remains causal. All prompts must fit in a single step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend

from gleipnir.serving_mxfp8 import forward_plan, produce
from gleipnir.serving_prompt_only_contract import validate_complete_prompts


def packed_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cumulative: torch.Tensor,
    maximum: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Use fresh THD K/V with the unchanged block-scaled causal cuDNN plan."""
    if (
        query.shape[1:] != (16, 256)
        or key.shape != value.shape
        or key.shape != (query.shape[0], 4, 256)
        or out.shape != query.shape
        or any(x.dtype != torch.bfloat16 for x in (query, key, value, out))
    ):
        raise ValueError("prompt-only MXFP8 requires BF16 THD D256 GQA 16/4")
    qr, sfq = produce(query, cumulative, maximum)
    kr, sfk = produce(key, cumulative, maximum)
    vc, sfv = produce(value, cumulative, maximum, column=True)
    plan = forward_plan(query.device)
    workspace = torch.empty(
        max(1, plan.scratch_workspace_bytes()), device=query.device, dtype=torch.uint8
    )
    plan.execute(
        qr,
        kr,
        vc,
        out,
        seq_q_lens=cumulative,
        seq_kv_lens=cumulative,
        sf_q=sfq,
        sf_k=sfk,
        sf_v=sfv,
        workspace=workspace,
    )
    return out


@dataclass
class PromptOnlyMetadata:
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    max_query_len: int
    nums_dict: dict
    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    has_initial_state: torch.Tensor
    state_indices: torch.Tensor


class PromptOnlyBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: Any,
        fast_build: bool = False,
    ) -> PromptOnlyMetadata:
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        m = common_attn_metadata
        lengths = m.query_start_loc_cpu.diff().tolist()
        computed = m.num_computed_tokens_cpu.tolist()
        validate_complete_prompts(lengths, computed)
        nums, batch, offsets = compute_causal_conv1d_metadata(
            m.query_start_loc_cpu, device=m.query_start_loc.device
        )
        return PromptOnlyMetadata(
            m.num_actual_tokens,
            m.query_start_loc,
            max(lengths),
            nums,
            batch,
            offsets,
            torch.zeros(
                len(lengths), dtype=torch.bool, device=m.query_start_loc.device
            ),
            torch.arange(
                len(lengths), dtype=torch.int32, device=m.query_start_loc.device
            ),
        )


class PromptOnlyBackend(FlashInferBackend):
    # Cache updates are unnecessary; this also removes the compiler's cache op.
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "GLEIPNIR_CAUSAL_PROMPT_ONLY"

    @staticmethod
    def get_builder_cls() -> type[PromptOnlyBuilder]:
        return PromptOnlyBuilder


def gdn_core(
    self: Any,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    """Keep within-prompt conv/GDN arithmetic; discard all continuation state."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    raw = get_forward_context().attn_metadata
    if raw is None:
        self._warmup_prefill_kernels(mixed_qkv, 0)
        return
    m = raw[self.prefix]
    n = m.num_actual_tokens
    weights = self.conv1d.weight.view(self.conv1d.weight.size(0), -1)
    # The pinned conv API requires a state tensor. It is scratch for this call,
    # never read as initial state or retained for a later request/step.
    scratch = torch.empty(
        (m.query_start_loc.numel() - 1, weights.shape[0], weights.shape[1] - 1),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )
    conv = causal_conv1d_fn(
        mixed_qkv[:n].transpose(0, 1),
        weights,
        self.conv1d.bias,
        conv_states=scratch,
        query_start_loc=m.query_start_loc,
        has_initial_state=m.has_initial_state,
        cache_indices=m.state_indices,
        null_block_id=None,
        activation=self.activation,
        metadata=m,
    ).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv,
        a=a[:n],
        b=b[:n],
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_k_heads=self.num_k_heads // self.tp_size,
        head_k_dim=self.head_k_dim,
        head_v_dim=self.head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    result = chunk_gated_delta_rule(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=torch.exp(g.float()),
        beta=beta.float(),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=m.query_start_loc,
        use_qk_l2norm_in_kernel=False,
    )
    # Retain the reference output-copy path; direct output was a separate trial.
    core_attn_out[:n].copy_(result)
    self._prompt_only_observed(self.prefix, n)


def install(runner: Any, observed: Callable[[str, int], None]) -> dict:
    """Attach one zero-storage metadata group; leave scheduler cache groups empty."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheGroupSpec

    layers = runner.compilation_config.static_forward_context
    full = {n: x for n, x in layers.items() if isinstance(x, Attention)}
    gdn = {n: x for n, x in layers.items() if isinstance(x, QwenGatedDeltaNetAttention)}
    if len(full) != 8 or len(gdn) != 24:
        raise ValueError("prompt-only mode requires the pinned 32-layer Qwen3.5-4B")
    names = list(full) + list(gdn)
    for layer in full.values():
        layer.attn_backend = PromptOnlyBackend

        def forward(
            impl,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            if attn_metadata is None:
                return output.fill_(0)
            if output_scale is not None or output_block_scale is not None:
                raise ValueError("prompt-only output quantization is unsupported")
            m = attn_metadata
            n = m.num_actual_tokens
            packed_forward(
                query[:n],
                key[:n],
                value[:n],
                m.query_start_loc,
                m.max_query_len,
                output[:n],
            )
            observed(layer.layer_name, n)
            return output

        layer.impl.forward = MethodType(forward, layer.impl)
        layer.get_kv_cache_spec = MethodType(lambda self, config: None, layer)
    for layer in gdn.values():
        if layer.gdn_prefill_backend != "flashinfer":
            raise ValueError("prompt-only GDN requires the selected FlashInfer backend")
        layer._forward_core = MethodType(gdn_core, layer)
        layer.get_attn_backend = MethodType(lambda self: PromptOnlyBackend, layer)
        layer.get_kv_cache_spec = MethodType(lambda self, config: None, layer)
        layer._prompt_only_observed = observed

    def add_group(self):
        if (
            self.kv_cache_config.kv_cache_groups
            or self.kv_cache_config.kv_cache_tensors
        ):
            raise ValueError("prompt-only mode must not allocate a persistent cache")
        self.runner_only_attn_layers.update(names)
        spec = EncoderOnlyAttentionSpec(
            block_size=self.vllm_config.cache_config.block_size,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
        )
        self.kv_cache_config.kv_cache_groups.append(
            KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)
        )

    runner.may_add_encoder_only_layers_to_kv_cache_config = MethodType(
        add_group, runner
    )
    return {
        "full_attention_layers": list(full),
        "gdn_layers": list(gdn),
        "persistent_cache_bytes": 0,
        "causal": True,
        "conv_state": "per-call scratch only",
        "gdn_final_state": False,
    }


def install_api_guard() -> None:
    """Keep the HTTP API but reject continuation before admitting GPU work."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    from gleipnir.serving_prompt_only_contract import validate_prompt_only_request

    original = OpenAIServingCompletion.create_completion

    async def complete(self, request, raw_request=None):
        try:
            validate_prompt_only_request(request.max_tokens, request.n)
        except ValueError as error:
            return self.create_error_response(str(error))
        return await original(self, request, raw_request)

    OpenAIServingCompletion.create_completion = complete
    original_chat = OpenAIServingChat.create_chat_completion

    async def chat(self, request, raw_request=None):
        try:
            tokens = request.max_completion_tokens
            validate_prompt_only_request(
                request.max_tokens if tokens is None else tokens, request.n
            )
        except ValueError as error:
            return self.create_error_response(str(error))
        return await original_chat(self, request, raw_request)

    OpenAIServingChat.create_chat_completion = chat
