"""Keep full-vocabulary output precision with unchanged BF16 head weights."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)


class Fp32LogitsMethod(UnquantizedEmbeddingMethod):
    """Use BF16/FP16 tensor-core inputs with FP32 matmul output before sampling."""

    def __init__(self) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP32 logits projection expects BF16/FP16 inputs")
        shape = x.shape[:-1]
        output = torch.mm(
            x.reshape(-1, x.shape[-1]), layer.weight.t(), out_dtype=torch.float32
        )
        if bias is not None:
            output = output + bias.float()
        if output.dtype != torch.float32:
            raise RuntimeError("Output projection did not retain FP32 logits")
        logger.info_once(
            "FP32 full-vocabulary projection executed: shape=%s dtype=%s",
            tuple(output.shape),
            output.dtype,
            scope="global",
        )
        return output.reshape(*shape, layer.weight.shape[0])
