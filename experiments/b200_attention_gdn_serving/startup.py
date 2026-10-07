"""Measure restart latency and bounded parity without a repeated serving sweep."""

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from experiments.b200_attention_gdn_serving.run import OUTPUT, resolve_condition
from experiments.b200_inference_benchmark.run import (
    EXPERIMENT,
    ROOT,
    benchmark,
    prepared_manifest,
    resolve_kernel_baseline,
    sha,
    verify_merged_model,
    write,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("startup run name must be a directory stem")
    base = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    manifest = prepared_manifest(base)
    selected = resolve_kernel_baseline({"baseline": "selected"})
    previous = json.loads((ROOT / selected["baseline"] / "condition.json").read_text())
    argv = previous["extra_server_args"]
    old_sources = json.loads(argv[argv.index("--additional-config") + 1])[
        "gleipnir_frost_fp4"
    ]
    sources = [
        *old_sources,
        "src/gleipnir/serving_runtime.py",
        "src/gleipnir/serving_compile_cache.py",
    ]
    raw = json.loads(
        (Path(__file__).parent / "fp4_swiglu_native_output.json").read_text()
    )
    raw["name"] = args.name
    condition = resolve_condition(raw, {p: sha(ROOT / p) for p in sources})
    config = {**base, **raw["serving_config_overrides"], "port": raw["port"]}
    out = OUTPUT / args.name
    out.mkdir(exist_ok=False)
    write(out / "condition.json", condition)
    write(out / "manifest.json", manifest)
    for source in [
        *sources,
        "src/gleipnir/serving_cache_mirror.py",
        str(Path(__file__).relative_to(ROOT)),
        "experiments/b200_inference_benchmark/run.py",
    ]:
        target = out / "executed_sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    try:
        merged = Path(raw["merged_model"])
        write(out / "merged_artifact.json", verify_merged_model(base, merged))
        asyncio.run(
            benchmark(
                config, manifest, out, 64, False, merged, condition, startup_only=True
            )
        )
    except BaseException as error:
        write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        for name in [
            "compile_identity.json",
            "loaded_precision.json",
            "native_attention.json",
            "native_preparation.json",
            "native_gemm_tuning.json",
            "native_attention_projections.json",
            "native_swiglu_output.json",
        ]:
            path = OUTPUT / name
            if path.exists():
                write(out / name, json.loads(path.read_text()))


if __name__ == "__main__":
    main()
