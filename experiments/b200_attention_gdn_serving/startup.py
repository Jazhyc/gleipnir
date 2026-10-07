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
from gleipnir._compat import canonical_source_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--frontend-validation")
    parser.add_argument("--frontend-ab", action="store_true")
    parser.add_argument("--host-wrapper-validation")
    parser.add_argument("--legacy-host", action="store_true")
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
        *(canonical_source_reference(p) for p in old_sources),
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
        "src/gleipnir/serving/sources.py",
        "src/gleipnir/serving/runtime.py",
        "src/gleipnir/serving/compile_cache.py",
    ]
    raw = json.loads(
        (Path(__file__).parent / "fp4_swiglu_native_output.json").read_text()
    )
    raw["name"] = args.name
    condition = resolve_condition(raw, {p: sha(ROOT / p) for p in sources})
    config = {**base, **raw["serving_config_overrides"], "port": raw["port"]}
    out = OUTPUT / args.name
    out.mkdir(exist_ok=False)
    from gleipnir.serving_reference import selected_host_components

    frontend, host_wrapper = (
        (None, None) if args.legacy_host else selected_host_components(ROOT, out)
    )
    if args.frontend_validation:
        frontend = {
            "backend": "gigatoken_native",
            "package_path": "/tmp/gleipnir-gigatoken-0.10.0",
            "validation": args.frontend_validation,
            "receipt_path": str(out / "frontend.json"),
            "ab_control": args.frontend_ab,
        }
    elif args.frontend_ab:
        if frontend is None:
            raise ValueError("frontend A/B requires native validation")
        frontend["ab_control"] = True
    if frontend is not None:
        write(out / "frontend_config.json", frontend)
    write(out / "condition.json", condition)
    write(out / "manifest.json", manifest)
    if args.host_wrapper_validation:
        host_wrapper = {"validation": args.host_wrapper_validation}
    if host_wrapper and frontend is None:
        raise ValueError("host wrapper control requires the native registry frontend")
    if host_wrapper is not None:
        write(out / "host_wrapper_config.json", host_wrapper)
    for source in [
        "src/gleipnir/__init__.py",
        "src/gleipnir/_compat.py",
        *sources,
        "src/gleipnir/serving/cache_mirror.py",
        str(Path(__file__).relative_to(ROOT)),
        "experiments/b200_inference_benchmark/run.py",
        *(["src/gleipnir/serving/frost_wrappers.py"] if host_wrapper else []),
        *(
            [
                "src/gleipnir/serving/gigatoken.py",
                "experiments/b200_inference_benchmark/frontend_server.py",
            ]
            if frontend is not None
            else []
        ),
    ]:
        target = out / "executed_sources" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    try:
        merged = Path(raw["merged_model"])
        write(out / "merged_artifact.json", verify_merged_model(base, merged))
        asyncio.run(
            benchmark(
                config,
                manifest,
                out,
                64,
                False,
                merged,
                condition,
                startup_only=True,
                frontend=frontend,
                host_wrapper=host_wrapper,
                use_selected_host=not args.legacy_host,
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
