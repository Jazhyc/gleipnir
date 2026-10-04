"""Freeze the unchanged augmentation and audited cached original 9B scores."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment

from experiments.monitor_injection_augmentation.prepare_eval import verify_evaluation
from gleipnir.monitoring_campaign_data import digest, file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DATA = ROOT / "data/monitor_injection_augmentation_9b"
OUTPUT = ROOT / "results/monitor_injection_augmentation_9b"
TRANSFER = ROOT / "data/monitoring_injection_disentanglement"
SHARED = ROOT / "data/monitor_injection_augmentation"


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def populations() -> dict[str, Path]:
    return {
        **{
            s: TRANSFER / s / "neutral.jsonl"
            for s in ("benchmark", "honest_controls", "id")
        },
        "synthetic_id": SHARED / "evaluation/synthetic_id.jsonl",
    }


def benchmark_tools() -> None:
    root = (ROOT / configuration()["benchmark_root"]).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))


def cached_scores(path: Path, inputs: list[dict], template: str) -> list[dict]:
    """Require exact cached coverage, input identity and binary-logit scores."""
    rows = read_rows(path)
    expected = {r["id"]: r for r in inputs}
    if len(rows) != len(inputs) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("cached baseline coverage drift")
    renderer = ImmutableSandboxedEnvironment().from_string(template)
    for row in rows:
        source = expected.pop(row["id"])
        metadata = source["metadata"]
        if row.get("label", row.get("ground_truth")) != metadata["ground_truth"]:
            raise ValueError("cached baseline label drift")
        prompt = (
            renderer.render(
                messages=[{"role": "user", "content": source["prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
        )
        # The original ID cache hashes the rendered boundary; APPS hashes the
        # source user prompt. Preserve both historical cache contracts.
        expected_hash = digest(
            prompt if "margin_prompt_sha256" in row else source["prompt"]
        )
        if row.get("margin_prompt_sha256", row.get("prompt_sha256")) != expected_hash:
            raise ValueError("cached baseline prompt drift")
        if "condition" in row and any(row.get(k) != v for k, v in metadata.items()):
            raise ValueError("cached baseline APPS metadata drift")
        if "source" in row and row["source"] != metadata["source_dataset"]:
            raise ValueError("cached baseline source drift")
        if "source_prompt_sha256" in row and row["source_prompt_sha256"] != digest(
            source["prompt"]
        ):
            raise ValueError("cached baseline source prompt drift")
        delta = row["logprob_1"] - row["logprob_0"]
        if (
            not math.isfinite(delta)
            or abs(row["score"] - 1 / (1 + math.exp(-max(-80, min(80, delta))))) > 1e-12
        ):
            raise ValueError("cached baseline score/logit drift")
    if expected:
        raise ValueError("cached baseline missing identities")
    return rows


def verify_preparation() -> dict:
    verify_evaluation()
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("9B configuration drift")
    for name, sha in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"9B source drift: {name}")
    if file_hash(DATA / "baseline.json") != manifest["baseline_sha256"]:
        raise ValueError("frozen 9B baseline drift")
    return manifest


def main() -> None:
    verify_evaluation()
    if (DATA / "manifest.json").exists():
        print(verify_preparation()["summary"], flush=True)
        return
    config = configuration()
    reference = ROOT / config["startup_validation_reference"]
    if file_hash(reference) != config["startup_validation_reference_sha256"]:
        raise ValueError("9B startup reference drift")
    from gleipnir.validated_startup import validation_reference

    validation_reference(reference)
    tokenizer = reference.parent / "chat_template.jinja"
    template = tokenizer.read_text()
    sources = [reference, ROOT / config["prior_job"], tokenizer, HERE / "prepare.py"]
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        old = (
            ROOT
            / "results/student_injection_awareness/4b/regular/causal_adapter"
            / name
        )
        new = reference.parent / name
        if file_hash(old) != file_hash(new):
            raise ValueError("replication requires matching validated tokenizer files")
        sources.extend((old, new))
    benchmark_root = (ROOT / config["benchmark_root"]).resolve()
    baseline_paths = {
        "id": reference.parent.parent / "id/predictions.jsonl",
        "benchmark": benchmark_root
        / "results/gleipnir-grid-v1/9b/regular/eval_neutral.jsonl",
        "honest_controls": benchmark_root
        / "results/gleipnir-honest-controls-v1/9b/regular/eval_neutral.jsonl",
    }
    cached = {}
    for name, path in baseline_paths.items():
        cached[name] = cached_scores(path, read_rows(populations()[name]), template)
        sources.append(path)
        if name != "id":
            complete_path = path.with_suffix(".complete.json")
            contract_path = path.with_suffix(".contract.json")
            complete = json.loads(complete_path.read_text())
            contract = json.loads(contract_path.read_text())
            if (
                not complete["passed"]
                or complete["sha256"] != file_hash(path)
                or complete["rows"] != len(cached[name])
                or complete["contract_sha256"] != contract["sha256"]
                or contract["base_revision"] != config["model"]["revision"]
                or contract["model_variant"] != "regular"
                or contract["prompt_variant"] != "neutral"
                or contract["serving_sha256"]
                != "3dfc9a9ca9b96c3126cb53c0f984728eaf7377de598656c34ea38b326e587df5"
            ):
                raise ValueError("cached APPS completion/model identity drift")
            sources.extend((complete_path, contract_path))
    id_result_path = baseline_paths["id"].with_name("result.json")
    id_result = json.loads(id_result_path.read_text())
    if id_result["rows"] != 3012 or id_result["model"]["id"] != config["model"]["id"]:
        raise ValueError("cached ID model/coverage drift")
    sources.append(id_result_path)
    benchmark_tools()
    from metrics import summarize
    from metrics_honest_controls import summarize_controls

    standard, _ = summarize(baseline_paths["benchmark"])
    baseline = {
        "id": id_result,
        "monitoring": {
            "neutral": {
                "standard": standard,
                "honest_controls": summarize_controls(
                    cached["honest_controls"], cached["benchmark"]
                ),
            }
        },
        "original_adapter_rescored": False,
    }
    write_json(DATA / "baseline.json", baseline)
    sources += list(populations().values())
    sources += [
        SHARED / p
        for p in (
            "student_rows.jsonl",
            "manifest.json",
            "insertion_ledger.jsonl",
            "evaluation/manifest.json",
        )
    ]
    sources += [
        ROOT / "data/student_injection_awareness/soft_targets.jsonl",
        ROOT / "src/gleipnir/validated_startup.py",
    ]
    sources.append(
        ROOT / "results/monitor_injection_augmentation/4b/augmented/token_audit.json"
    )
    # Store portable repository-relative paths, including the sibling benchmark.
    import os

    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "source_sha256": {os.path.relpath(p, ROOT): file_hash(p) for p in sources},
            "baseline_sha256": file_hash(DATA / "baseline.json"),
            "summary": {
                "training_rows": 8688,
                "injected_rows": 3475,
                "expected_steps": 272,
                "evaluation_rows": {
                    n: len(read_rows(p)) for n, p in populations().items()
                },
                "baseline_rows": {n: len(r) for n, r in cached.items()},
            },
        },
    )
    print(verify_preparation()["summary"], flush=True)


if __name__ == "__main__":
    main()
