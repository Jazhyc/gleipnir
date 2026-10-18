"""Score the trained 27B on the frozen OOD workload (6,395 rows).

Prepares an OOD-only scoring context from the campaign config, as the 4B
seed-variation runs did, then reuses their scorer: re-merge the FP32 master
into a fresh slot (byte-identical to the campaign merge, whose parity is
reused), BF16 serving parity on the canary, OOD scoring and scorer retirement.
"""

from __future__ import annotations

import argparse
import copy
import json

import yaml

from experiments.qwen38_27b_campaign.prepare import ARM, ROOT
from experiments.teacher_seed_variation.prepare import OOD, OOD_MANIFEST
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

NAME = f"{ARM.removesuffix('-01')}-ood01"
PARENT = f"results/{ARM}"


def scoring_config(campaign: dict) -> dict:
    config = copy.deepcopy(campaign)
    config.update(
        campaign_id=NAME,
        merged_model=f"/tmp/gleipnir-merged/{NAME}",
        parent_campaign=PARENT,
        score_splits=["ood"],
        evaluations=[],
    )
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, (path, _) in OOD.items():
        config["inputs"][key] = {"path": path, "sha256": file_hash(ROOT / path)}
    return config


def prepare() -> None:
    state = json.loads((ROOT / PARENT / "runner.json").read_text())
    if state["status"] != "complete":
        raise ValueError("the parent campaign has not completed")
    for key, (path, expected) in OOD.items():
        if expected and file_hash(ROOT / path) != expected:
            raise ValueError(f"OOD input drift: {key}")
    frozen = json.loads((ROOT / OOD_MANIFEST).read_text())["files_sha256"]
    if frozen[OOD["ood_workload"][0]] != file_hash(ROOT / OOD["ood_workload"][0]):
        raise ValueError("frozen OOD workload drift")
    campaign = yaml.safe_load((ROOT / "data" / ARM / "config.yaml").read_text())
    data = ROOT / "data" / NAME
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    data.mkdir(parents=True)
    path = data / "config.yaml"
    path.write_text(yaml.safe_dump(scoring_config(campaign), sort_keys=False))
    ctx = Campaign(ROOT, path, yaml.safe_load(path.read_text()))
    write_json(
        data / "manifest.json",
        {
            "files_sha256": {
                str(path.relative_to(ROOT)): file_hash(path),
                **{s["path"]: s["sha256"] for s in ctx.config["inputs"].values()},
            },
            "source_sha256": ctx.sources(),
            "parent": PARENT,
        },
    )
    print(path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "score"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        from experiments.teacher_seed_variation.run import score

        score(NAME)
        print("qwen38_27b_ood_complete", flush=True)


if __name__ == "__main__":
    main()
