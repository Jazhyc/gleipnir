"""Start/submit bounded trials on an existing GPU; never provision capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.b200_mlp_gemm.resident_protocol import validate_request, write_json
from gleipnir._compat import canonical_source_reference

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/b200_mlp_gemm/warmedprofile01/summary.json"
SOURCE_SHA = "64507ef439da31bb4294b54cc3f6a310b3e4e662d2a4fe224f3f493cba07c772"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("--baseline-only", action="store_true")
    submit = sub.add_parser("submit")
    submit.add_argument("--id", required=True)
    submit.add_argument(
        "--variant",
        choices=("baseline", "gemmprofile", "candidate"),
        default="baseline",
    )
    sub.add_parser("status")
    args = parser.parse_args()
    if not args.session.isalnum():
        raise ValueError("session must be alphanumeric")
    root = ROOT / "results/b200_mlp_gemm" / args.session
    if args.action == "status":
        print((root / "worker.json").read_text())
        return
    if args.action == "submit":
        request = {"id": args.id, "variant": args.variant}
        if args.variant == "candidate":
            request["source_sha256"] = sha(
                Path(
                    os.environ.get(
                        "GLEIPNIR_FP4_RESIDENT_CANDIDATE",
                        str(Path(__file__).with_name("resident_candidate.py")),
                    )
                )
            )
        validate_request(request)
        if (root / args.id).exists() or (
            root / "requests" / f"{args.id}.json"
        ).exists():
            raise ValueError("trial already exists")
        write_json(root / "requests" / f"{args.id}.json", request)
        print(json.dumps({"queued": args.id}))
        return
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("validated launch contract checksum drift")
    source = json.loads(SOURCE.read_text())
    import yaml

    from experiments.b200_nvidia_mxfp8.run import environment

    for path_key, digest_key in (
        ("student_rows", "student_rows_sha256"),
        ("soft_targets", "soft_targets_sha256"),
        ("selection_manifest", "selection_sha256"),
    ):
        if sha(Path(source["job"][path_key])) != source["job"][digest_key]:
            raise ValueError(f"input drift: {path_key}")
    if not source["hardware_packing"] or not source["fused_descale"]:
        raise ValueError("resident baseline requires both validated FP4 fusions")
    root.mkdir(parents=True, exist_ok=False)
    (root / "requests").mkdir()
    logroot = ROOT / "logs/runpod/b200_mlp_gemm" / args.session
    logroot.mkdir(parents=True, exist_ok=False)
    command = [
        x.replace("warmedprofile01", args.session)
        for x in source["command"]
        if x != "++student.training.native_fp4_mlp_profile=true"
    ]
    command[0] = sys.executable
    command.append("++student.training.native_fp4_mlp_resident=true")
    env = environment(
        yaml.safe_load((ROOT / "experiments/b200_mlp_gemm/config.yaml").read_text())
    )
    env.update(
        HF_HOME=str(ROOT / ".cache/huggingface"),
        HF_HUB_CACHE=str(ROOT / ".cache/huggingface/hub"),
        TORCHINDUCTOR_COMPILE_THREADS="16",
        MAX_JOBS="16",
        PYTHONUNBUFFERED="1",
        GLEIPNIR_FP4_RESIDENT_ROOT=str(root),
        GLEIPNIR_FP4_RESIDENT_REFERENCE=str(
            ROOT
            / "results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json"
        ),
        GLEIPNIR_FP4_HARDWARE_PACKING="1",
        GLEIPNIR_FP4_FUSED_DESCALE="1",
        GLEIPNIR_FP4_RUNTIME_REPORT=str(root / "native_runtime.json"),
    )
    for key in ("GLEIPNIR_FP4_WARM_REPORT", "GLEIPNIR_FP4_PROFILE_OUTPUT"):
        env.pop(key, None)
    from gleipnir.serving.runtime import local_serving_runtime

    staged_runtime = local_serving_runtime(ROOT, env)
    if staged_runtime is not None:
        command[0] = staged_runtime["python"]
    files = [
        *Path("experiments/b200_mlp_gemm").glob("*.py"),
        *[
            Path(canonical_source_reference(x))
            for x in source["source_sha256"]
            if not x.startswith("experiments/b200_mlp_gemm/")
        ],
        Path("src/gleipnir/training/startup.py"),
        Path("src/gleipnir/training/adaptive_microbatching.py"),
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path("src/gleipnir/serving/runtime.py"),
    ]
    files = sorted(set(files))
    for file in files:
        target = root / "executed_sources" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / file).read_bytes())
    receipt = {
        "status": "starting",
        "command": command,
        "launch_reference_sha256": SOURCE_SHA,
        "baseline": "combined_native_fp4_mlp_bf16_fa4",
        "timing_only": True,
        "staged_runtime": staged_runtime,
        "cache_paths": {k: v for k, v in env.items() if "CACHE" in k},
        "source_sha256": {str(f): sha(ROOT / f) for f in files},
    }
    write_json(root / "launch.json", receipt)
    # Check reset stability twice; the last trial attributes GEMMs.
    conditions = [
        ("01baseline", "baseline"),
        ("02repeat", "baseline"),
    ]
    if not args.baseline_only:
        conditions.append(("03gemmprofile", "gemmprofile"))
    for name, variant in conditions:
        write_json(root / "requests" / f"{name}.json", {"id": name, "variant": variant})
    with (logroot / "worker.log").open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt.update(pid=process.pid, status="launched")
    write_json(root / "launch.json", receipt)
    print(json.dumps({"pid": process.pid, "root": str(root)}))


if __name__ == "__main__":
    main()
