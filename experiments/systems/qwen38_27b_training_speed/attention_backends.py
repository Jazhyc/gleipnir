"""Time and check SDPA backends at Qwen3.8-27B full-attention shapes.

Calls the same Transformers ``sdpa_attention_forward`` entry point as the
segmented router (one causal sequence, 24 query / 4 KV heads, head dim 256)
under each forced backend. Reports forward+backward time per sequence length
and output/gradient relative L2 against an FP32 math reference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import sdpa_attention_forward

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
}
HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
LENGTHS = (1024, 4096, 8192, 16384, 28733)


def inputs(length: int, dtype: torch.dtype, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def make(heads):
        return torch.randn(
            1, heads, length, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
        )

    q, k, v = make(HEADS), make(KV_HEADS), make(KV_HEADS)
    do = torch.randn(1, length, HEADS, HEAD_DIM, device="cuda", dtype=dtype,
                     generator=generator)
    return [x.requires_grad_() for x in (q, k, v)], do


def run(backend, q, k, v, do):
    module = SimpleNamespace(
        num_key_value_groups=HEADS // KV_HEADS, is_causal=True, training=True
    )
    with sdpa_kernel([backend]):
        out, _ = sdpa_attention_forward(
            module, q, k, v, None, dropout=0.0, scaling=HEAD_DIM**-0.5, is_causal=True
        )
        grads = torch.autograd.grad(out, (q, k, v), do)
    return out, grads


def relative(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cudnn": torch.backends.cudnn.version(), "timing": {}, "parity": {}}
    # Parity at 4,096 tokens against an FP32 math reference.
    (q, k, v), do = inputs(4096, torch.bfloat16, seed=1)
    ref = [x.detach().float().requires_grad_() for x in (q, k, v)]
    reference = run(SDPBackend.MATH, *ref, do.float())
    for name, backend in BACKENDS.items():
        try:
            out, grads = run(backend, q, k, v, do)
            report["parity"][name] = {
                "output": relative(out, reference[0]),
                **{g: relative(a, b) for g, a, b in zip("qkv", grads, reference[1],
                                                         strict=True)},
            }
        except RuntimeError as error:
            report["parity"][name] = {"error": str(error).splitlines()[0][:300]}
    del ref, reference
    for length in LENGTHS:
        (q, k, v), do = inputs(length, torch.bfloat16)
        for name, backend in BACKENDS.items():
            if "error" in report["parity"][name]:
                continue
            try:
                for _ in range(3):
                    run(backend, q, k, v, do)
                torch.cuda.synchronize()
                start = time.perf_counter()
                repeats = 10
                for _ in range(repeats):
                    run(backend, q, k, v, do)
                torch.cuda.synchronize()
                seconds = (time.perf_counter() - start) / repeats
                report["timing"].setdefault(str(length), {})[name] = seconds
            except RuntimeError as error:
                report["timing"].setdefault(str(length), {})[name] = str(error)[:200]
        print(length, report["timing"].get(str(length)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["parity"], indent=1))


if __name__ == "__main__":
    main()
