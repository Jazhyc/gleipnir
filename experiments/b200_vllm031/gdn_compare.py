"""Compare preserved FlashInfer 0.6 GDN with 0.7 on identical native inputs."""

import argparse
import hashlib
import importlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch

from experiments.b200_inference_benchmark.run import write
from gleipnir.flashqla_training import tensor_comparison


def legacy_kernel(directory: Path) -> Callable[..., None]:
    """Import only the three preserved Blackwell source files, in isolation."""
    package = ModuleType("gleipnir_legacy_blackwell")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    return importlib.import_module(
        package.__name__ + ".gdn_prefill"
    ).chunk_gated_delta_rule_sm100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope_gate,
    )

    old = Path("/tmp/gleipnir-serving-runtime/venv/lib/python3.12/site-packages")
    directory = old / "flashinfer/gdn_kernels/blackwell"
    legacy = legacy_kernel(directory)
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*.py")
    }
    torch.manual_seed(310)
    checks = []
    for lengths in ([17], [129], [4096], [17, 511, 129]):
        total = sum(lengths)
        q, k = [
            torch.nn.functional.normalize(
                torch.randn(total, 16, 128, device="cuda"), dim=-1
            ).to(torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(total, 32, 128, device="cuda", dtype=torch.bfloat16)
        gate = torch.exp(-torch.rand(total, 32, device="cuda") * 0.1)
        beta = torch.rand_like(gate)
        state = torch.randn(len(lengths), 32, 128, 128, device="cuda") * 0.01
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        expected = torch.empty_like(v)
        expected_state = torch.empty_like(state)
        legacy(
            q, k, v, gate, beta, expected, cu, state, expected_state, 1 / math.sqrt(128)
        )
        output, final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu,
            backend="flashinfer",
        )
        no_cp_output, no_cp_final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu,
            backend="flashinfer",
            use_cp=False,
        )
        torch.cuda.synchronize()
        checks.append(
            {
                "lengths": lengths,
                "output": tensor_comparison(output, expected),
                "state": tensor_comparison(final, expected_state),
                "output_exact": torch.equal(output, expected),
                "state_exact": torch.equal(final, expected_state),
                "no_cp_output": tensor_comparison(no_cp_output, expected),
                "no_cp_state": tensor_comparison(no_cp_final, expected_state),
                "auto_vs_no_cp_exact": torch.equal(output, no_cp_output),
            }
        )
        write(args.output, {"sources": sources, "checks": checks})
        print("gdn_pair", json.dumps(checks[-1]), flush=True)
    # The new norm_beta path should reproduce the old caller's FP32 weight+1.
    q = torch.randn(129, 8192, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(129, 1024, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cache = torch.randn(4096, 64, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(129, device="cuda")
    inputs = (cache, positions, 1e-6, 16, 4, 256, 64)
    old_qk = fused_qk_rmsnorm_rope_gate(
        q, k, *(w.float() + 1 for w in weights), *inputs
    )
    new_qk = fused_qk_rmsnorm_rope_gate(q, k, *weights, *inputs, norm_beta=1.0)
    qk = {
        "effective_weight_exact": all(
            torch.equal(a, b) for a, b in zip(old_qk, new_qk, strict=True)
        )
    }
    write(args.output, {"sources": sources, "checks": checks, "qk": qk})
    print("qk_pair", qk, flush=True)


if __name__ == "__main__":
    main()
