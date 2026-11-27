"""Materialize paired parent/prefix rows only from a complete verified cache."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gleipnir.prefix_audit import validate_fresh_audit
from gleipnir.prefix_cache import validate_resume
from gleipnir.prefix_sampling import attach_sampled_prefix, sample_parent_prefixes


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_training(
    cache_dir: Path, references_path: Path, output: Path, *, seed: int = 0,
    numerical_exception: dict | None = None,
) -> dict:
    """Keep the original parent population and sample one prefix for epoch zero."""
    contract = json.loads((cache_dir / "contract.json").read_text())
    contract_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    completed = json.loads((cache_dir / "complete.json").read_text())
    manifest = contract["manifest"]
    source = Path(manifest["source"])
    if digest(source) != manifest["source_sha256"]:
        raise ValueError("parent source drift")
    if digest(references_path) != manifest["references_sha256"]:
        raise ValueError("prefix reference drift")
    refs = {}
    grouped = defaultdict(list)
    for line in references_path.open():
        row = json.loads(line)
        if row["id"] in refs:
            raise ValueError("duplicate prefix reference")
        refs[row["id"]] = row
        grouped[row["parent_prompt_id"]].append(row["id"])
    cached_path = cache_dir / "logits.jsonl"
    verified = validate_resume(cached_path, refs, contract_hash)
    if (
        verified != set(refs)
        or len(verified) != manifest["rows"]
        or completed["rows"] != len(verified)
        or completed["contract_sha256"] != contract_hash
    ):
        raise ValueError("teacher cache is not complete")
    selected = sample_parent_prefixes(grouped, seed=seed, epoch=0)
    selected_ids = set(selected.values())
    selected_cache = {}
    all_cached = {}
    for line in cached_path.open():
        row = json.loads(line)
        all_cached[row["id"]] = row
        if row["id"] in selected_ids:
            # Endpoints and parent identities come from authoritative references.
            selected_cache[row["id"]] = {**row, **refs[row["id"]]}
    audit_path = cache_dir / "fresh_audit.json"
    if numerical_exception is not None and (
        numerical_exception.get("audit_sha256") != digest(audit_path)
        or numerical_exception.get("contract_sha256") != contract_hash
        or not numerical_exception.get("user_authorization")
    ):
        raise ValueError("numerical exception identity or authorization missing")
    audit_passed = validate_fresh_audit(
        json.loads(audit_path.read_text()),
        list(refs.values()),
        all_cached,
        contract_hash,
        allow_numerical_failure=numerical_exception is not None,
    )
    parents = [json.loads(line) for line in source.open()]
    parent_ids = [row["prompt_id"] for row in parents]
    if len(set(parent_ids)) != len(parent_ids) or set(grouped) - set(parent_ids):
        raise ValueError("ambiguous or missing parent identity")
    paired = [
        attach_sampled_prefix(row, selected_cache.get(selected.get(row["prompt_id"])))
        for row in parents
    ]
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        for row in paired:
            stream.write(json.dumps(row) + "\n")
    result = {
        "parent_rows": len(parents),
        "parents_with_prefix": len(selected),
        "seed": seed,
        "epoch": 0,
        "sampling": "uniform_per_parent_v1",
        "source_sha256": manifest["source_sha256"],
        "cache_contract_sha256": contract_hash,
        "cache_sha256": digest(cached_path),
        "fresh_audit_sha256": digest(audit_path),
        "fresh_audit_passed": audit_passed,
        "numerical_exception": numerical_exception,
        "references_sha256": manifest["references_sha256"],
        "output_sha256": digest(output),
        "full_supervision": "unchanged original fields; no parent duplication",
    }
    with sidecar.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("data/monitoring_prefix_supervision/prefix_references.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_training(
                args.cache_dir, args.references, args.output, seed=args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
