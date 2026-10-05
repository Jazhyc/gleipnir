"""Bounded native GEMM epilogue test before any full-MLP integration."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from experiments.b200_mlp_gemm.integrated_probe import timings
from gleipnir.bf16_lora import configure_bf16_reductions
from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
from gleipnir.cudnn_fp4_gemm import pack_operand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "synthetic_weights": True, "cases": []}

    def save():
        (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        torch.manual_seed(53)
        weights = {
            "gate_up": torch.randn(18432, 2560, device="cuda", dtype=torch.bfloat16)
            / 2560**0.5,
            "down": torch.randn(2560, 9216, device="cuda", dtype=torch.bfloat16)
            / 9216**0.5,
        }
        for rows in (193, 4096, 16384):
            for name, weight in weights.items():
                for backward in (False, True):
                    w = weight.t().contiguous() if backward else weight
                    n, k = w.shape
                    packed_w = pack_operand(w, weight=True)
                    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                    op = Nvfp4ScaledGemm(rows, k, n)

                    def run(operator, x=x, packed_w=packed_w):
                        return operator(
                            pack_operand(
                                x,
                                row_amax=True,
                                chunked_rows=True,
                                hardware_packing=True,
                            ),
                            packed_w,
                        )

                    reference = run(op.reference)
                    actual = run(op)
                    case = {
                        "rows": rows,
                        "projection": name,
                        "backward": backward,
                        "bitwise_output": torch.equal(actual, reference),
                        "finite_nonzero": bool(torch.isfinite(actual).all())
                        and bool(torch.count_nonzero(actual)),
                        "tile_config": op.plan.tile_config_name,
                    }
                    report["cases"].append(case)
                    save()
                    if not case["bitwise_output"] or not case["finite_nonzero"]:
                        raise ValueError(f"row-descaling arithmetic failed: {case}")
                    source = x.clone()
                    source[17:].mul_(31.7)
                    x.copy_(source)
                    changed = run(op)
                    case["row_isolation_bitwise"] = torch.equal(
                        actual[:17], changed[:17]
                    )
                    if not case["row_isolation_bitwise"]:
                        raise ValueError("row-descaling isolation failed")
                    graphs, outputs = {}, {}
                    warm = torch.cuda.Stream()
                    warm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warm):
                        for operator in (op.reference, op):
                            for _ in range(3):
                                run(operator)
                    torch.cuda.current_stream().wait_stream(warm)
                    torch.cuda.synchronize()
                    for leg, operator in {
                        "reference": op.reference,
                        "fused": op,
                    }.items():
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs[leg] = run(operator)
                        graphs[leg] = graph

                    def replay(graph, x=x, source=source):
                        x.copy_(source)
                        graph.replay()

                    for _ in range(6):
                        for graph in graphs.values():
                            replay(graph)
                    case["graph_timing"] = timings(
                        {leg: lambda g=g: replay(g) for leg, g in graphs.items()}
                    )
                    case["graph_copy_included"] = True
                    source.mul_(0.7)
                    replay(graphs["fused"])
                    torch.cuda.synchronize()
                    case["changed_input_replay_bitwise"] = torch.equal(
                        outputs["fused"], run(op)
                    )
                    if not case["changed_input_replay_bitwise"]:
                        raise ValueError("row-descaling replay is stale")
                    save()
                    print(json.dumps(case), flush=True)
                    del graphs, outputs, graph, reference, actual, changed, op
        report["status"] = "complete"
    except Exception as exc:
        report.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
