"""Freeze the first Qwen3.8-27B monitor campaign on the default 27B recipe.

Copies the strict-filtered BF16/SDPA 4B campaign (same rows, Kimi K3 targets,
LR, logical batch, one epoch, ID evaluation) at seed 0, the seed of the
27B initial adapter, and changes only the model, profile, initial adapter,
base snapshot binding and runtime slots.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = "teacher-seed-sdpa-s1-01"
ARM = "qwen38-27b-s0-01"
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SNAPSHOT = f".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/{REVISION}"
ADAPTER = "results/qwen38_27b_initial_adapter"
BASE_FILES = "results/qwen38_27b_training_speed/base_files.json"
# Repository documentation, not model or tokenizer assets; the merge omits it.
DOCUMENTATION = {".gitattributes", "LICENSE", "README.md"}
FIELDS = {
    "hypothesis": (
        "Distilling Kimi K3 soft targets into Qwen3.8-27B with the 4B recipe "
        "raises held-out ID macro pAUROC@20 above both the unadapted 27B (0.8869) "
        "and the seed-0 4B student (0.8959)."
    ),
    "intervention": (
        "Seed-0 strict-filtered BF16/SDPA campaign with Qwen3.8-27B and the "
        "qwen38_27b_b200_default recipe; data, targets, LR and schedule fixed."
    ),
    "selection": (
        "Final one-epoch checkpoint only; descriptive ID comparison with the "
        "unadapted 27B and the 4B seeds; no promotion; OOD outside the campaign."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, failed startup gates, nonfinite or missing "
        "gradients, OOM, failed master/merge/serving parity, incomplete scoring "
        "or completion."
    ),
}


def arm_config(control: dict, gpu_uuid: str, adapter: dict, base_files: dict) -> dict:
    config = copy.deepcopy(control)
    config.update(
        campaign_id=ARM,
        seed=0,
        profile="qwen38_27b_b200_default",
        full_startup_validation=True,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARM}",
        base_model=SNAPSHOT,
        base_files=base_files,
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        **FIELDS,
    )
    config["model"] = {
        "id": MODEL,
        "revision": REVISION,
        "initial_adapter": ADAPTER,
        "initial_tensor_sha256": adapter["initial_master_sha256"],
    }
    config["inputs"].pop("startup_reference")
    for name, file in (
        ("initial_adapter_config", "adapter_config.json"),
        ("initial_weights", "adapter_model.safetensors"),
    ):
        config["inputs"][name] = {
            "path": f"{ADAPTER}/{file}",
            "sha256": adapter["files_sha256"][file],
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    adapter = json.loads((ROOT / ADAPTER / "initialization_receipt.json").read_text())
    for name, sha in adapter["files_sha256"].items():
        if file_hash(ROOT / ADAPTER / name) != sha:
            raise ValueError(f"initial adapter drift: {name}")
    base_files = {
        name: sha
        for name, sha in json.loads((ROOT / BASE_FILES).read_text()).items()
        if name not in DOCUMENTATION
    }
    data = ROOT / "data" / ARM
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    data.mkdir(parents=True)
    config = arm_config(control, args.gpu_uuid, adapter, base_files)
    (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(data / "config.yaml")


if __name__ == "__main__":
    main()
