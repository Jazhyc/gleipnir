"""Selected CPU serving components, separate from GPU compilation identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def selected_score_reference(root: Path) -> dict:
    """Verify the selected score workload, receipts and every saved repeat."""
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    if selection.get("endpoint") != "/v1/monitor/score":
        raise ValueError("selected reference is not a monitoring score server")
    for path, digest in selection["artifact_bindings"].items():
        artifact = root / path
        if not artifact.resolve().is_relative_to(root.resolve() / "results") or (
            hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("selected score reference artifact drift")
    directory = root / selection["results"]
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or not summary["score_audit_passed"]:
        raise ValueError("selected score reference incomplete")
    manifest = root / "data/b200_inference_benchmark/manifest.json"
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != selection["manifest_sha256"]
    ):
        raise ValueError("selected score reference workload drift")
    expected = {
        str((directory / f"c{c}_repeat{i}.json").relative_to(root))
        for c, count in ((1, 3), (128, 6))
        for i in range(count)
    }
    if not expected.issubset(selection["artifact_bindings"]):
        raise ValueError("selected score reference repeats unbound")
    return selection


def selected_serving_default(
    root: Path, *, last_query: bool = True
) -> tuple[dict, list[str]]:
    """Resolve the accepted current recipe separately from frozen comparisons."""
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/serving_default.json").read_text()
    )
    if selection["quality_status"] != "user_accepted_finite":
        raise ValueError("serving default lacks explicit finite-quality acceptance")
    for path, digest in selection["artifact_bindings"].items():
        artifact = root / path
        if not artifact.resolve().is_relative_to(root.resolve() / "results") or (
            hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"serving default artifact drift: {path}")
    report = json.loads((root / selection["recipe_summary"]).read_text())
    if report["status"] != "complete" or report["precision"] != "fp8":
        raise ValueError("serving default is not a completed FP8 projection recipe")
    command = report["command"].copy()
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    condition = additional["serving_condition"]
    if (
        condition["attention_projection_precision"] != "fp8"
        or condition["gdn_projection_precision"] != "fp4"
        or condition["attention_precision"] != "mxfp8"
        or command[command.index("--max-model-len") + 1] != "32768"
        or command[command.index("--runner") + 1] != "pooling"
    ):
        raise ValueError("accepted serving precision/context contract changed")
    # Tests and client launchers do not execute in the model. Their subsequent
    # fixture cleanup retains the measured kernel identity; runtime sources
    # still require exact agreement with the evaluated bytes.
    for path, expected in additional["gleipnir_frost_fp4"].items():
        if (
            Path(path).name.startswith("test_")
            or Path(path).name.endswith(("_canary.py", "_compare.py"))
            or Path(path).name in {"run.py", "fp4_gemm_tune.py"}
            or path == "src/gleipnir/serving/reference.py"
        ):
            continue
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"accepted serving runtime source drift: {path}")
    trim = selection.get("last_query_attention")
    if trim is not None and last_query:
        apply_last_query(root, command, trim, selection["artifact_bindings"])
        bindings = selection["artifact_bindings"]
        route = selection.get("gdn_route")
        if route is not None:
            gate = {"cp_min_tokens": int(route["cp_min_tokens"])}
            apply_worker_layer(root, command, route, bindings, ("gdn_route", gate))
            fast = selection.get("gdn_fast")
            if fast is not None:
                apply_worker_layer(root, command, fast, bindings)
    return selection, command


def apply_worker_layer(
    root: Path,
    command: list[str],
    layer: dict,
    bindings: dict[str, str],
    config: tuple[str, dict] | None = None,
) -> None:
    """Stack one selected worker layer, binding its sources and evidence."""
    worker = command.index("--worker-cls") + 1
    if command[worker] != layer["replaces_worker"]:
        raise ValueError(f"{layer['worker']} replaces an unexpected worker")
    if any(path not in bindings for path in layer["evidence"]):
        raise ValueError("worker layer evidence is not artifact-bound")
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    for path, expected in layer["sources"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"accepted worker layer source drift: {path}")
        additional["gleipnir_frost_fp4"][path] = expected
    if config is not None:
        additional[config[0]] = config[1]
    command[worker] = layer["worker"]
    command[index] = json.dumps(additional, sort_keys=True)


def apply_last_query(
    root: Path, command: list[str], trim: dict, bindings: dict[str, str]
) -> None:
    """Select final-layer last-query attention with its bound receipt/sources.

    Launchers that substitute their own worker class keep the untrimmed recipe.
    """
    worker = command.index("--worker-cls") + 1
    if command[worker] != trim["replaces_worker"]:
        raise ValueError("last-query attention replaces an unexpected worker")
    if trim["validation"] not in bindings:
        raise ValueError("last-query kernel receipt is not artifact-bound")
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    for path, expected in trim["sources"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"accepted last-query source drift: {path}")
        additional["gleipnir_frost_fp4"][path] = expected
    additional["last_query_attention"] = {
        "validation": trim["validation"],
        "validation_sha256": bindings[trim["validation"]],
    }
    command[worker] = trim["worker"]
    command[index] = json.dumps(additional, sort_keys=True)


def selected_host_components(
    root: Path, output: Path, *, resident: dict | None = None
) -> tuple[dict | None, dict | None]:
    """Bind selected receipts and preserve the actual resident frontend path."""
    selection = json.loads(
        (root / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    frontend = selection.get("frontend")
    host = selection.get("host_wrapper")
    for component in (frontend, host):
        if component is not None:
            path = root / component["validation"]
            if not path.resolve().is_relative_to(root.resolve() / "results"):
                raise ValueError("selected host validation must be an artifact")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != component["validation_sha256"]
            ):
                raise ValueError("selected host validation checksum drift")
    if frontend is not None:
        frontend = {k: v for k, v in frontend.items() if k != "validation_sha256"}
        frontend["receipt_path"] = (
            resident["frontend"]["receipt_path"]
            if resident and resident.get("frontend")
            else str(output / "frontend.json")
        )
    if host is not None:
        host = {"validation": host["validation"]}
    return frontend, host
