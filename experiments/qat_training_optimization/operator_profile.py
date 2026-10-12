"""Isolated FP8 projection timing/profile after the model worker is idle."""

import time
from pathlib import Path

import torch

from gleipnir.data.monitoring import write_json
from gleipnir.training.bf16_lora import configure_bf16_reductions
from gleipnir.training.qat_optimization import native_fp8_linear


def main():
    root = Path.cwd()
    output = root / "results/qat-training-optimization02"
    import json

    if json.loads((output / "status.json").read_text())["phase"] != "idle":
        raise RuntimeError("resident timing worker must be idle")
    configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
    torch.manual_seed(9)
    results = []
    for m in (137, 4097, 16384):
        for n, k in ((8192, 2560), (1024, 2560), (2560, 8192)):
            a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)

            def baseline(a=a, b=b):
                return a.float() @ b.float().t()

            def optimized(a=a, b=b):
                return native_fp8_linear(a, b)

            timings = []
            for name, function in (("baseline", baseline), ("native_fp8", optimized)):
                for _ in range(5):
                    function()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                begin = time.monotonic()
                for _ in range(20):
                    function()
                torch.cuda.synchronize()
                timings.append(
                    {
                        "variant": name,
                        "milliseconds": (time.monotonic() - begin) * 1000 / 20,
                        "peak_extra_bytes": torch.cuda.max_memory_allocated()
                        - allocated,
                    }
                )
            results.append({"m": m, "n": n, "k": k, "timings": timings})
            if m == 4097 and n == 8192:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as profile:
                    with torch.profiler.record_function("decoded_fp8_fp32_reference"):
                        baseline()
                    with torch.profiler.record_function("native_fp8_fp32_accumulate"):
                        optimized()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(output / "operator_trace.json"))
                (output / "operator_profile.txt").write_text(
                    profile.key_averages().table(
                        sort_by="self_device_time_total", row_limit=40
                    )
                )
            del a, b
    write_json(
        output / "operator_timings.json",
        {
            "scope": "isolated_contractions_including_code_conversion_not_full_updates",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "warmup_calls": 5,
            "measured_calls": 20,
            "cases": results,
        },
    )


if __name__ == "__main__":
    main()
