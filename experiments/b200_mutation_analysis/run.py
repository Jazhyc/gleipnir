"""Native metadata gate followed by the shared bounded score-endpoint screen."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write
from experiments.b200_monitor_score.run import measure


def native_prepare(environment: dict, output: Path) -> None:
    command = [
        "/tmp/gleipnir-serving-runtime/venv/bin/python",
        "-m",
        "experiments.b200_mutation_analysis.native",
        "--output",
        str((output / "native.json").relative_to(ROOT)),
    ]
    with (output / "native.log").open("x") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not json.loads((output / "native.json").read_text())["passed"]:
        raise ValueError("native mutation gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    resume = None
    environment = None
    if args.resume_from:
        if Path(args.resume_from).name != args.resume_from:
            raise ValueError("resume name must be a directory stem")
        resume = (
            ROOT
            / "results/b200_mutation_analysis"
            / args.resume_from
            / "parent_server.json"
        )
        parent = json.loads(resume.read_text())
        retired = json.loads(
            (
                ROOT
                / "results/b200_attention_gdn_serving"
                / f"{args.resume_from}_server_retired.json"
            ).read_text()
        )
        if retired["status"] != "retired" or retired["pid"] != parent["pid"]:
            raise ValueError("parent retirement receipt changed")
        from experiments.b200_inference_benchmark.run import (
            environment as base_environment,
        )
        from gleipnir.native_fp4_training import native_fp4_environment
        from gleipnir.qwen35_fast_training import (
            DEFAULT_TRITON_TARGET,
            triton_environment,
        )
        from gleipnir.serving_runtime import local_serving_runtime

        environment = native_fp4_environment(
            triton_environment(DEFAULT_TRITON_TARGET, base_environment()), ROOT
        )
        environment["PYTHONPATH"] += f":{ROOT / '.cache/kernels/fa4'}"
        runtime = local_serving_runtime(ROOT, environment)
        if runtime is None or runtime["python"] != parent["command"][0]:
            raise ValueError("retired runtime could not be reconstructed")
        environment.update(
            {
                k: v
                for k, v in parent["cache_paths"].items()
                if k != "FLASHINFER_CACHE_DIR"
            }
        )
        environment["PYTHONPATH"] = (
            parent["frontend"]["package_path"] + ":" + environment["PYTHONPATH"]
        )
        environment["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"][
            "validation"
        ]
        environment["VLLM_SERVER_DEV_MODE"] = "1"
        if parent["frontend"].get("ab_control"):
            environment["GLEIPNIR_GIGATOKEN_AB"] = "1"
    try:
        asyncio.run(
            measure(
                args.name,
                experiment=Path(__file__).parent,
                result_group="b200_mutation_analysis",
                native_prepare=native_prepare,
                retired_parent=resume,
                resumed_environment=environment,
            )
        )
    except BaseException as error:
        write(
            ROOT / "results/b200_mutation_analysis" / args.name / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
