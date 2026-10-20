"""Archive checked v0.2 derivatives of the frozen QAD training/reference stages.

On top of the shared selected-arithmetic transform, only the canary reference
(ids, base scores, prompt hashes) and the starting optimized-server canary
move from the FP4/FA4 baseline's receipts to v0.2's.
"""

from pathlib import Path

import yaml

from experiments.native_fp8_qat.generate import transform as reference_transform
from gleipnir.data.monitoring import file_hash, write_json

OLD_REFERENCE = (
    "root\n            / "
    '"results/filtered-fp4-fa4-baseline01/4b/monitor/parity_reference.json"'
)
OLD_SERVER = 'root / "results/filtered-fp4-fa4-optimized-id01/canary_repeat1.json"'


def transform(source: str, stage: str, **variant) -> tuple[str, list[dict]]:
    source, receipts = reference_transform(source, stage, **variant)
    replacements = [(OLD_REFERENCE, 'root / config["canary_reference"]', 1)]
    if stage == "train":
        replacements.append((OLD_SERVER, 'root / config["initial_server_canary"]', 2))
    for old, new, expected in replacements:
        count = source.count(old)
        if count != expected:
            raise ValueError(f"v0.2 substitution count drift: {old!r} x{count}")
        source = source.replace(old, new)
        receipts.append({"old": old, "new": new, "count": count})
    compile(source, f"qad_v02/{stage}.py", "exec")
    return source, receipts


def generate(root: Path, output: Path, **variant) -> dict[str, Path]:
    directory = output / "stage_sources"
    directory.mkdir(parents=True, exist_ok=True)
    stages, receipts = {}, {}
    for stage in ("train", "reference"):
        original = root / f"experiments/quantization_recovery/{stage}.py"
        generated, changes = transform(original.read_text(), stage, **variant)
        (directory / f"original_{stage}.py").write_text(original.read_text())
        target = directory / f"{stage}.py"
        target.write_text(generated)
        stages[stage] = target
        receipts[stage] = {
            "original_sha256": file_hash(original),
            "generated_sha256": file_hash(target),
            "replacements": changes,
        }
    write_json(output / "stage_generation.json", receipts)
    return stages


def config(root: Path) -> dict:
    return yaml.safe_load((root / "experiments/qad_v02/config.yaml").read_text())
