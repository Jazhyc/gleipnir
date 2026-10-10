"""Publish the completed CAFT adapter and verify its injection-collection entry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.data.monitoring import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "results/caft-regular-sdpa02"
OUTPUT = ROOT / "results/concept_ablation_training"
NAME = "Gleipnir-4B-ToolTrajectories-CAFT"
COLLECTION = "Jazhyc/gleipnir-prompt-injection-monitoring-6ac287c39057bb51ee492009"
MASTER_SHA = "601a6c9b96ecffe9807fec8552564ac0ec4d1412c752bbd6863374b9370ee8e6"
SERVING_SHA = "2fdbbbe2c2a069c55db0362f727b67322c6f5b6bd7c99bac7d946a10162224d4"


def read(path: Path) -> dict:
    """Read one retained JSON receipt."""
    return json.loads(path.read_text())


def stage() -> Path:
    """Stage only verified weights, prompt, summaries, model card and license."""
    adapter = CAMPAIGN / "4b/monitor"
    if read(CAMPAIGN / "runner.json")["status"] != "complete":
        raise ValueError("campaign is incomplete")
    complete = read(adapter / "complete.json")
    rebase = read(adapter / "model/rebase_manifest.json")
    parity = read(CAMPAIGN / "optimized_parity.json")
    merge_parity = read(CAMPAIGN / "merged_parity.json")
    audit = read(OUTPUT / "independent_audit.json")
    if not all(r["passed"] for r in (parity, merge_parity, audit)):
        raise ValueError("publication requires successful parity and audit")
    if parity["serving_precision"] != "bf16" or audit["rows"] != 12126:
        raise ValueError("evaluation contract mismatch")
    for kind, expected, field in (
        ("causal_adapter", MASTER_SHA, "source_sha256"),
        ("model", SERVING_SHA, "destination_sha256"),
    ):
        weights = adapter / kind / "adapter_model.safetensors"
        if file_hash(weights) != expected or rebase[field] != expected:
            raise ValueError(f"{kind} weight identity drift")
        with safe_open(weights, framework="pt", device="cpu") as handle:
            if len(handle.keys()) != 256 or any(
                handle.get_slice(k).get_dtype() != "F32" for k in handle.keys()
            ):
                raise ValueError(f"{kind} must preserve all 256 FP32 LoRA tensors")
    if (
        complete["master_sha256"] != MASTER_SHA
        or complete["serving_sha256"] != SERVING_SHA
        or complete["steps"] != 272
        or merge_parity["master_sha256"] != MASTER_SHA
    ):
        raise ValueError("checkpoint identity drift")
    release = OUTPUT / "release" / NAME
    release.mkdir(parents=True, exist_ok=True)
    for kind, relative in (("causal_adapter", ""), ("model", "vllm")):
        destination = release / relative
        destination.mkdir(exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source = adapter / kind / name
            expected = read(CAMPAIGN / "runner.json")["stages"]["train"][
                "files_sha256"
            ][f"/workspace/gleipnir/{source.relative_to(ROOT).as_posix()}"]
            if file_hash(source) != expected:
                raise ValueError(f"training artifact drift: {source.name}")
            shutil.copy2(source, destination / name)
    shutil.copy2(ROOT / "LICENSE", release / "LICENSE")
    template = load_prompt_set().student
    (release / "student_prompt.txt").write_text(template.instruction)
    (release / "README.md").write_text(
        (HERE / "model_card.md")
        .read_text()
        .replace("{{STUDENT_INSTRUCTION}}", template.instruction)
    )
    write_json(
        release / "prompt_contract.json",
        {
            "prompt_set_id": template.prompt_set_id,
            "template_sha256": template.template_sha256,
            "trajectory_open": template.trajectory_open,
            "trajectory_close": template.trajectory_close,
            "decision_prefix": template.decision_prefix,
            "negative_label": template.negative_label,
            "positive_label": template.positive_label,
            "enable_thinking": False,
            "role": "single user message",
            "scoring": "sigmoid(logit_1 - logit_0) at assistant Prediction: prefix",
        },
    )
    write_json(
        release / "evaluation_results.json",
        {
            "id": read(OUTPUT / "id_metrics.json")["caft"],
            "apps": read(OUTPUT / "heldout_apps.json")["metrics"]["caft"],
            "evaluation_backend": "merged BF16 vLLM 0.31, ordinary unablated inference",
            "qualification": (
                "One seed; cached control differs in precision, backend and host."
            ),
        },
    )
    write_json(
        release / "parity_summary.json",
        {
            "master_sha256": MASTER_SHA,
            "serving_layout_sha256": SERVING_SHA,
            "master_to_bf16_serving": parity,
            "master_to_merge": {k: v for k, v in merge_parity.items() if k != "scores"},
        },
    )
    write_json(
        release / "release_manifest.json",
        {
            "campaign": "caft-regular-sdpa02",
            "base_model": "Qwen/Qwen3.5-4B",
            "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "config_sha256": file_hash(HERE / "config.yaml"),
            "files_sha256": {
                p.relative_to(release).as_posix(): file_hash(p)
                for p in release.rglob("*")
                if p.is_file() and p.name != "release_manifest.json"
            },
        },
    )
    return release


def publish(release: Path) -> None:
    """Upload the staged allowlist, verify remote hashes, then add to collection."""
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")
    api = HfApi(token=token)
    namespace = api.whoami()["name"]
    if namespace != "Jazhyc":
        raise ValueError("publication account differs from the requested collection")
    collection = api.get_collection(COLLECTION)
    if collection.title != "Gleipnir Prompt Injection Monitoring" or collection.private:
        raise ValueError("collection identity or visibility drift")
    repo_id = f"{namespace}/{NAME}"
    # Existing repositories are never overwritten unless created by this release.
    created = OUTPUT / "publication_created.json"
    if not created.exists():
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=False, exist_ok=False
        )
        write_json(created, {"repo_id": repo_id, "collection": COLLECTION})
    elif read(created) != {"repo_id": repo_id, "collection": COLLECTION}:
        raise ValueError("publication destination drift")
    manifest = read(release / "release_manifest.json")
    files = {
        **manifest["files_sha256"],
        "release_manifest.json": file_hash(release / "release_manifest.json"),
    }
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=release,
        allow_patterns=list(files),
        commit_message="Release regular-data CAFT adapter and negative-result card",
    )
    remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
    siblings = {item.rfilename: item for item in remote.siblings}
    if remote.sha != commit.oid or set(siblings) != set(files) | {".gitattributes"}:
        raise ValueError("remote revision or file coverage mismatch")
    for name, expected in files.items():
        if siblings[name].lfs:
            actual = siblings[name].lfs.sha256
        else:
            actual = file_hash(
                Path(hf_hub_download(repo_id, name, revision=commit.oid, token=token))
            )
        if actual != expected:
            raise ValueError(f"remote checksum mismatch: {name}")
    api.add_collection_item(COLLECTION, repo_id, "model", exists_ok=True)
    updated = api.get_collection(COLLECTION)
    if not any(i.item_id == repo_id and i.item_type == "model" for i in updated.items):
        raise ValueError("collection membership verification failed")
    write_json(
        OUTPUT / "upload.json",
        {
            "repo_id": repo_id,
            "revision": commit.oid,
            "collection": COLLECTION,
            "remote_verified": True,
            "collection_verified": True,
            "files_sha256": files,
        },
    )
    print(f"uploaded https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    release = stage()
    print(f"staged {release}", flush=True)
    if args.upload:
        publish(release)
