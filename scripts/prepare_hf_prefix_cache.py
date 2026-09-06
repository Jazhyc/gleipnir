"""Stage only the approved, validated public prefix-cache release files."""

import hashlib
import json
import shutil
from pathlib import Path

from gleipnir.prefix_cache import validate_resume


def main() -> None:
    cache = Path("results/monitoring_prefix_supervision/qwen35_flashinfer_cache")
    output = Path("results/hf_release/Gleipnir-Prefix-Teacher-Cache")
    contract = json.loads((cache / "contract.json").read_text())
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    refs = {
        r["id"]: r for r in map(
            json.loads,
            Path("data/monitoring_prefix_supervision/prefix_references.jsonl").open(),
        )
    }
    if validate_resume(cache / "logits.jsonl", refs, digest) != set(refs):
        raise ValueError("incomplete cache")
    if len(refs) != 133947:
        raise ValueError("unexpected release coverage")
    allowed = {
        "id", "parent_prompt_id", "lineage_group", "source", "prefix_ordinal",
        "prefix_count", "end_character", "trajectory_sha256", "prefix_sha256",
        "rendered_user_prompt_sha256", "format_warnings", "contract_sha256",
        "logprob_0", "logprob_1", "score", "timestamp_unix", "prompt_tokens",
        "cached_tokens", "completion_tokens", "serving_prompt_sha256",
    }
    for row in map(json.loads, (cache / "logits.jsonl").open()):
        if set(row) != allowed:
            raise ValueError("unreviewed release fields")
    files = {name: cache / name for name in (
        "logits.jsonl", "contract.json", "complete.json",
        "fresh_audit.json", "failure_replay.json",
    )}
    experiment = Path("experiments/monitoring_prefix_supervision")
    files.update({
        "README.md": experiment / "dataset_card.md",
        "teacher_prefix.txt": experiment / "teacher_prefix.txt",
    })
    output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, source in files.items():
        shutil.copy2(source, output / name)
        checksums[name] = {
            "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            "bytes": (output / name).stat().st_size,
        }
    (output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"staged": str(output), "files": sorted(checksums)}))


if __name__ == "__main__":
    main()
