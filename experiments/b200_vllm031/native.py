"""Revalidate MXFP8 and compare actual 0.31 K/V strides with the old layout."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch

from experiments.b200_inference_benchmark.run import sha, write
from gleipnir.serving.mxfp8 import paged_forward


def qk_checks() -> list[dict]:
    """Check the changed Qwen kernel on Torch 2.13 without the old repair."""
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

    def forward(q, k, qw, kw, cache, positions):
        return fused_qk_rmsnorm_rope_gate(
            q,
            k,
            qw,
            kw,
            cache,
            positions,
            eps=1e-6,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=256,
            rotary_dim=64,
        )

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    checks = []
    for rows, strided in ((17, False), (129, True)):
        step = 2 if strided else 1
        inputs = (
            torch.randn(rows * step, 8192, device="cuda", dtype=torch.bfloat16)[::step],
            torch.randn(rows * step, 1024, device="cuda", dtype=torch.bfloat16)[::step],
            torch.ones(256, device="cuda", dtype=torch.bfloat16),
            torch.ones(256, device="cuda", dtype=torch.bfloat16),
            torch.randn(32768, 64, device="cuda", dtype=torch.bfloat16),
            torch.arange(rows, device="cuda", dtype=torch.int64),
        )
        before = [value.clone() for value in inputs]
        expected = forward(*inputs)
        observed = compiled(*inputs)
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(expected, observed, strict=True)):
            raise ValueError("0.31 Qwen eager/compiled output mismatch")
        if not all(torch.equal(a, b) for a, b in zip(inputs, before, strict=True)):
            raise ValueError("0.31 Qwen kernel mutated an input")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(*inputs)
        inputs[0].mul_(0.9)
        graph.replay()
        torch.cuda.synchronize()
        expected = forward(*inputs)
        if not all(torch.equal(a, b) for a, b in zip(expected, captured, strict=True)):
            raise ValueError("0.31 Qwen changed-input graph replay mismatch")
        checks.append(
            {
                "rows": rows,
                "strided": strided,
                "compiled_exact": True,
                "inputs_unchanged": True,
                "changed_input_replay_exact": True,
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.b200_attention_gdn_serving.mxfp8_canary",
            "--output",
            str(args.output),
        ],
        check=True,
    )
    receipt = json.loads(args.output.read_text())
    if not receipt["arithmetic_passed"]:
        raise ValueError("0.31 MXFP8 quantized arithmetic validation failed")
    torch.manual_seed(310)
    checks = []
    for query_lengths, history_lengths in (([1, 17], [17, 129]), ([129], [511])):
        pages = [(n + 15) // 16 for n in history_lengths]
        combined = torch.randn(
            sum(pages), 2, 4, 16, 256, dtype=torch.bfloat16, device="cuda"
        )
        content = torch.cat(tuple(combined.unbind(1)), dim=-1)
        cache_pair = content.split(256, dim=-1)
        table = torch.zeros(len(pages), max(pages), dtype=torch.int32, device="cuda")
        offset = 0
        for row, count in enumerate(pages):
            table[row, :count] = torch.arange(
                offset, offset + count, device="cuda"
            ).flip(0)
            offset += count

        def cumulative(values):
            return torch.tensor(
                [0, *torch.tensor(values).cumsum(0).tolist()],
                dtype=torch.int32,
                device="cuda",
            )

        query = torch.randn(
            sum(query_lengths), 16, 256, dtype=torch.bfloat16, device="cuda"
        )
        old_out, new_out = torch.empty_like(query), torch.empty_like(query)
        kwargs = {
            "query": query,
            "block_tables": table,
            "cum_seq_lens_q": cumulative(query_lengths),
            "cum_seq_lens_kv": cumulative(pages),
            "seq_lens": torch.tensor(history_lengths, dtype=torch.int32, device="cuda"),
            "max_q_len": max(query_lengths),
            "max_kv_len": max(history_lengths),
            "batch_size": len(pages),
        }
        paged_forward(**kwargs, kv_cache=combined, out=old_out)
        paged_forward(**kwargs, kv_cache=cache_pair, out=new_out)
        torch.cuda.synchronize()
        exact = torch.equal(old_out, new_out)
        if not exact or not bool(torch.isfinite(new_out).all()):
            raise ValueError("strided cache views changed native MXFP8 output")
        checks.append(
            {
                "queries": query_lengths,
                "histories": history_lengths,
                "bitwise_equal": exact,
                "strides": list(cache_pair[0].stride()),
            }
        )
    receipt["vllm031_cache_layout_checks"] = checks
    receipt["vllm031_qk_checks"] = qk_checks()
    receipt["migration_passed"] = True
    receipt["migration_native_source_sha256"] = sha(Path(__file__))
    receipt["torch"] = torch.__version__
    import cudnn
    from cudnn.gated_attention_block.kernels import proj_gemm
    from vllm.model_executor.layers import fused_qk_norm_rope

    receipt["active_native_runtime"] = {
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch_cuda": torch.version.cuda,
        "sources": {
            str(p): sha(p)
            for p in (
                Path(proj_gemm.__file__),
                Path(inspect.getfile(fused_qk_norm_rope)),
            )
        },
    }
    write(args.output, receipt)
    print("vllm031_native_passed", checks, flush=True)


if __name__ == "__main__":
    main()
