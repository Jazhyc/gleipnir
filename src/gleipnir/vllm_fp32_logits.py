"""Keep full-vocabulary output precision with unchanged BF16 head weights."""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = init_logger("vllm.gleipnir.fp32_logits")


def install_fp32_logits(model: torch.nn.Module) -> list[str]:
    """Install on loaded output heads, including tied embeddings without a config."""
    installed = []
    seen = set()
    for name, module in model.named_modules():
        head = getattr(module, "lm_head", None)
        if not isinstance(head, VocabParallelEmbedding) or id(head) in seen:
            continue
        if type(head.quant_method) not in {
            UnquantizedEmbeddingMethod,
            Fp32LogitsMethod,
        }:
            raise ValueError("FP32 logits require an unquantized vocabulary head")
        if head.weight.ndim != 2 or head.weight.dtype not in {
            torch.bfloat16,
            torch.float16,
        }:
            raise ValueError("FP32 logits require BF16/FP16 vocabulary weights")
        head.quant_method = Fp32LogitsMethod()
        installed.append(f"{name}.lm_head".lstrip("."))
        seen.add(id(head))
    if len(installed) != 1:
        raise ValueError(f"Expected one vocabulary head, found {installed}")
    logger.info("Installed FP32 output projection on loaded head %s", installed[0])
    return installed


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
