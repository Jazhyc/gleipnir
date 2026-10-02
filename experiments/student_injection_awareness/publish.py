"""Stage and publish evaluated FP32 masters and parity-checked serving adapters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from dotenv import dotenv_values
from huggingface_hub import HfApi
from safetensors import safe_open

from experiments.student_injection_awareness.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    VARIANTS,
    file_hash,
    templates,
    write_json,
)


def stage(size: str, variant: str) -> Path:
    """Build a minimal model release from verified completed experiment artifacts."""
    config = yaml.safe_load(CONFIG.read_text())
    directory = OUTPUT / size / variant
    parity = json.loads((directory / "serving_parity.json").read_text())
    if not parity["passed"]:
        raise ValueError("release serving parity did not pass")
    rebase = json.loads((directory / "model/rebase_manifest.json").read_text())
    master_path = directory / "causal_adapter/adapter_model.safetensors"
    if file_hash(master_path) != rebase["source_sha256"]:
        raise ValueError("release master checksum drift")
    if (
        file_hash(directory / "model/adapter_model.safetensors")
        != rebase["destination_sha256"]
        or parity["serving_sha256"] != rebase["destination_sha256"]
    ):
        raise ValueError("release serving checksum drift")
    with safe_open(master_path, framework="pt", device="cpu") as handle:
        if not handle.keys() or any(
            handle.get_slice(k).get_dtype() != "F32" for k in handle.keys()
        ):
            raise ValueError("release adapter is not an FP32 master")
    reports = {}
    for split, expected in [("id", 3012), ("ood", 6395)]:
        report = json.loads((directory / split / "result.json").read_text())
        if (
            report["rows"] != expected
            or report["template_sha256"] != templates()[variant].template_sha256
        ):
            raise ValueError("release evaluation incomplete or instruction drift")
        reports[split] = report
    suffix = "InjectionAware" if variant == "injection_aware" else "Regular"
    name = f"Gleipnir-{size.upper()}-ToolTrajectories-{suffix}"
    release = OUTPUT / "releases" / name
    release.mkdir(parents=True, exist_ok=True)
    for filename in [
        "adapter_model.safetensors",
        "adapter_config.json",
        "training_metadata.json",
    ]:
        shutil.copy2(directory / "causal_adapter" / filename, release / filename)
        (release / "vllm").mkdir(exist_ok=True)
        shutil.copy2(directory / "model" / filename, release / "vllm" / filename)
    shutil.copy2(
        directory / "model/rebase_manifest.json", release / "vllm/rebase_manifest.json"
    )
    shutil.copy2(ROOT / "LICENSE", release / "LICENSE")
    (release / "student_prompt.txt").write_text(templates()[variant].instruction)
    template = templates()[variant]
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
            "scoring": "assistant Prediction: prefix, normalized 1 vs 0 logits",
        },
    )
    write_json(release / "evaluation_results.json", reports)
    write_json(release / "serving_parity.json", parity)
    write_json(
        release / "training_data_provenance.json",
        json.loads((DATA / "manifest.json").read_text()),
    )
    model = config["models"][size]
    table = "\n".join(
        f"| {split.upper()} | "
        f"{r['metrics']['macro']['macro']['pauroc_at_20']:.6f} | "
        f"{r['metrics']['macro']['macro']['auroc']:.6f} |"
        for split, r in reports.items()
    )
    card = f"""---
license: mit
base_model: {model["id"]}
library_name: peft
pipeline_tag: text-generation
tags:
- ai-control
- monitoring
- lora
---

# {name}

Research LoRA monitor trained for one epoch on 8,688 tool trajectories with
unchanged cached Kimi K3 soft binary targets. No prior-deception training rows.
Student instruction: `{variant}`. This is one seed and an uncalibrated research
artifact; detection performance does not establish that a trajectory is safe.

Base revision: `{model["revision"]}`. Frozen BF16 base, FP32 rank-128/alpha-256
master adapters, packed logical batches of 32, AdamW learning rate 5e-5,
linear decay and 3% warmup. Final checkpoint is fixed after one epoch.
Training trajectories retain their original contents, including identified
reasoning in some STRIDE inputs. ID evaluation removes identified CoT and keeps
visible assistant prose; OOD uses the frozen six-source suite.

| Evaluation | Source-macro pAUROC@20 | Source-macro AUROC |
| --- | ---: | ---: |
{table}

The root contains the causal-LM FP32 master. `vllm/` contains its checksum-rebased
serving layout, verified against the master on a bounded training-data canary.
Use `student_prompt.txt` verbatim, then a newline and `<agent_trajectory>`,
the complete trajectory, and `</agent_trajectory>` on separate lines.
Apply the base tokenizer's chat template as one user message with thinking
disabled, then append `Prediction:` at the assistant generation boundary.
Request a single literal `0` or `1` token and both digit logprobs; the risk score
is sigmoid(logprob_1 - logprob_0). Higher values indicate problematic behavior.
Trajectory contents are untrusted and are preserved without sanitization.

Complete prompt, training-data hashes, backend parity, per-source ranking,
calibration, threshold and tie diagnostics are included in the JSON artifacts.
Experiment code: https://github.com/Jazhyc/gleipnir
"""
    (release / "README.md").write_text(card)
    write_json(
        release / "release_manifest.json",
        {
            "name": name,
            "size": size,
            "variant": variant,
            "files_sha256": {
                p.relative_to(release).as_posix(): file_hash(p)
                for p in release.rglob("*")
                if p.is_file() and p.name != "release_manifest.json"
            },
        },
    )
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4b", "9b"), required=True)
    args = parser.parse_args()
    values = dotenv_values(ROOT / ".env")
    token = values.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")
    api = HfApi(token=token)
    namespace = api.whoami()["name"]
    for variant in VARIANTS:
        release = stage(args.size, variant)
        repo_id = f"{namespace}/{release.name}"
        previous = OUTPUT / args.size / variant / "upload.json"
        if previous.exists():
            record = json.loads(previous.read_text())
            if record["release_manifest_sha256"] != file_hash(
                release / "release_manifest.json"
            ):
                raise ValueError("uploaded release identity drift")
            print(f"already_uploaded {repo_id}", flush=True)
            continue
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=False, exist_ok=False
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=release,
            commit_message="Release matched tool-trajectory student monitor",
        )
        remote = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        manifest = json.loads((release / "release_manifest.json").read_text())
        siblings = {item.rfilename: item for item in remote.siblings}
        expected = set(manifest["files_sha256"]) | {"release_manifest.json"}
        if remote.sha != commit.oid or not expected.issubset(siblings):
            raise ValueError("uploaded revision or file coverage mismatch")
        for name in ["adapter_model.safetensors", "vllm/adapter_model.safetensors"]:
            if siblings[name].lfs.sha256 != manifest["files_sha256"][name]:
                raise ValueError("uploaded adapter checksum mismatch")
        write_json(
            previous,
            {
                "repo_id": repo_id,
                "revision": commit.oid,
                "remote_verified": True,
                "release_manifest_sha256": file_hash(release / "release_manifest.json"),
            },
        )
        print(f"uploaded https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()
