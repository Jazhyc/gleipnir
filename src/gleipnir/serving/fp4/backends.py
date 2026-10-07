"""Matched FlashInfer FP4 GEMMs with unchanged BF16 rounding and row scaling."""

from __future__ import annotations

from typing import Any

BACKENDS = ("frost", "cudnn", "cutlass", "trtllm", "cute-dsl")
ROWS = (1, 17, 129, 1536, 2304, 4096, 29184, 32768)


def bounded_cute_tuning() -> list[dict]:
    """Limit isolated native-worker CuTe autotuning to eight diverse tactics."""
    from flashinfer.gemm import gemm_base

    original = gemm_base._get_sm100_block_scaled_tactics
    records = []

    def limited(*args, **kwargs):
        tactics = original(*args, **kwargs)
        chosen = []
        for swap in (False, True):
            for tile in ((128, 256), (256, 128), (128, 128), (256, 256)):
                available = [t for t in tactics if t[0] == tile and t[2] == swap]
                available.sort(key=lambda t: (t[1] != (2, 1), t[1] != (1, 1), not t[3]))
                if available:
                    chosen.append(available[0])
        for tactic in tactics:
            if len(chosen) >= 8:
                break
            if tactic not in chosen:
                chosen.append(tactic)
        records.append({"valid_count": len(tactics), "selected": repr(chosen)})
        return chosen

    gemm_base._get_sm100_block_scaled_tactics = limited
    return records


def linear_scale_offsets(rows: int, k: int) -> list[list[int]]:
    """Reference mapping from logical NVFP4 scale rows to 128x4 storage."""
    if rows <= 0 or k <= 0 or k % 64:
        raise ValueError("positive rows and K divisible by 64 are required")
    return [
        [
            (((r // 128 * (k // 64) + c // 4) * 32 + r % 32) * 4 + r % 128 // 32) * 4
            + c % 4
            for c in range(k // 16)
        ]
        for r in range(rows)
    ]


def prepare_weight(weight: Any, backend: str) -> tuple[Any, Any]:
    """Permute packed TRT-LLM weights once, preserving every FP4/SF byte."""
    import torch

    if backend not in BACKENDS[1:]:
        raise ValueError(f"unsupported FlashInfer FP4 backend: {backend}")
    codes = weight.codes.view(torch.uint8)
    scales = weight.scales.view(torch.uint8)
    if backend != "trtllm":
        return codes, scales
    from flashinfer.fp4_quantization import shuffle_matrix_a, shuffle_matrix_sf_a

    n, packed_k = codes.shape
    k = packed_k * 2
    r = torch.arange(n, device=codes.device)[:, None]
    c = torch.arange(k // 16, device=codes.device)[None, :]
    offsets = (
        ((r // 128 * (k // 64) + c // 4) * 32 + r % 32) * 4 + r % 128 // 32
    ) * 4 + c % 4
    linear_scales = scales.reshape(-1)[offsets].contiguous()
    return shuffle_matrix_a(codes, 128), shuffle_matrix_sf_a(linear_scales, 128)


class FlashInferScaledGemm:
    """Execute raw BF16 GEMM then identical row descaling; cache weight layouts."""

    def __init__(self, backend: str, k: int, n: int) -> None:
        import torch

        if backend not in BACKENDS[1:]:
            raise ValueError(f"unsupported FlashInfer FP4 backend: {backend}")
        self.backend, self.k, self.n = backend, k, n
        self.alpha = torch.ones(1, device="cuda", dtype=torch.float32)
        self.weights: dict[int, tuple[Any, Any, Any]] = {}

    def prepare(self, b: Any) -> tuple[Any, Any]:
        """Keep the source tensor alive, preventing pointer reuse in this cache."""
        key = b.codes.data_ptr()
        if key not in self.weights:
            codes, scales = prepare_weight(b, self.backend)
            self.weights[key] = (b, codes, scales)
        return self.weights[key][1:]

    def __call__(self, a: Any, b: Any) -> Any:
        import torch
        import triton
        from flashinfer.gemm import mm_fp4

        from gleipnir.cudnn_fp4_gemm import _descale

        m = a.codes.shape[0]
        if (
            not 1 <= m <= 32768
            or a.codes.shape[1] * 2 != self.k
            or b.codes.shape != (self.n, self.k // 2)
            or a.inverse.numel() != m
            or b.inverse.numel() != 1
        ):
            raise ValueError("FlashInfer FP4 operand outside validated envelope")
        codes, scales = self.prepare(b)
        raw = mm_fp4(
            a.codes.view(torch.uint8),
            codes.T,
            a.scales.view(torch.uint8),
            scales.T,
            alpha=self.alpha,
            out_dtype=torch.bfloat16,
            backend=self.backend,
            block_size=16,
            use_nvfp4=True,
            enable_pdl=False,
        )
        result = torch.empty_like(raw)
        _descale[(triton.cdiv(raw.numel(), 1024),)](
            raw, result, a.inverse, b.inverse, raw.numel(), 1024, self.n, True
        )
        return result
