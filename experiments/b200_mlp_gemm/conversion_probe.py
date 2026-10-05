"""Bitwise parity and matched complete-conversion timing for native FP4 packing."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.b200_mlp_gemm.integrated_probe import timings
from gleipnir.cudnn_fp4_gemm import PACKING_KERNEL_METADATA, pack_operand


def check_hardware_packing(output: Path) -> dict:
    report = {"status": "starting", "cases": []}

    def save():
        (output / "packing.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    # Keep the existing complete-MLP fixture's RNG stream unchanged.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(52)
        for rows in (193, 4096, 16384):
            for width in (2560, 9216, 18432):
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                edges = torch.tensor(
                    [
                        0.0,
                        -0.0,
                        0.25,
                        -0.25,
                        0.75,
                        -0.75,
                        1.25,
                        -1.25,
                        1.75,
                        -1.75,
                        2.5,
                        -2.5,
                        3.5,
                        -3.5,
                        5.0,
                        6.0,
                    ],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                x[1].copy_(edges.repeat(width // 16))
                functions = {
                    "reference": lambda x=x: pack_operand(
                        x, row_amax=True, chunked_rows=True
                    ),
                    "hardware": lambda x=x: pack_operand(
                        x, row_amax=True, chunked_rows=True, hardware_packing=True
                    ),
                }
                ref = functions["reference"]()
                reference_metadata = PACKING_KERNEL_METADATA[(width, True)]
                fast = functions["hardware"]()
                checks = {
                    "codes": torch.equal(
                        ref.codes.view(torch.uint8), fast.codes.view(torch.uint8)
                    ),
                    "scales_including_padding": torch.equal(
                        ref.scales.view(torch.uint8), fast.scales.view(torch.uint8)
                    ),
                    "row_inverse": torch.equal(ref.inverse, fast.inverse),
                }
                case = {
                    "rows": rows,
                    "width": width,
                    "bitwise": checks,
                    "reference_metadata": reference_metadata,
                    "hardware_metadata": PACKING_KERNEL_METADATA[(width, True)],
                }
                report["cases"].append(case)
                save()
                if not all(checks.values()):
                    report["status"] = "failed"
                    save()
                    raise ValueError(f"hardware FP4 packing parity failed: {case}")
                warm = torch.cuda.Stream()
                warm.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm):
                    for fn in functions.values():
                        for _ in range(3):
                            fn()
                torch.cuda.current_stream().wait_stream(warm)
                torch.cuda.synchronize()
                graphs, outputs = {}, {}
                for name, fn in functions.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs[name] = fn()
                    graphs[name] = graph
                source = x.clone()

                def replay(graph, x=x, source=source):
                    x.copy_(source)
                    graph.replay()

                for _ in range(6):
                    for graph in graphs.values():
                        replay(graph)
                case["graph_timing"] = timings(
                    {k: lambda g=g: replay(g) for k, g in graphs.items()}
                )
                case["graph_copy_included"] = True
                source[17:].mul_(31.7)
                for graph in graphs.values():
                    replay(graph)
                torch.cuda.synchronize()
                live = functions["reference"]()
                captured = outputs["hardware"]
                case["changed_input_bitwise"] = all(
                    torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(
                        (live.codes, live.scales, live.inverse),
                        (captured.codes, captured.scales, captured.inverse),
                        strict=True,
                    )
                )
                if not case["changed_input_bitwise"]:
                    raise ValueError("hardware packing replay is stale")
                save()
                print(json.dumps(case), flush=True)
                del graphs, outputs, graph, captured, live, fast, ref, x, source
    report["status"] = "complete"
    save()
    return report
