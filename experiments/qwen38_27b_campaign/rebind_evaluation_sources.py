"""Re-bind only the scoring-stage sources of the paused 27B campaign.

A prepared campaign pins every executed source. The 27B BF16 serving fixes
(commit 1b66597) landed after preparation and change only code that the
evaluate stage runs. This re-binds exactly those files, after verifying that
every completed stage's bound outputs are unchanged, archives the previous
manifest and records the change in the new one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from experiments.qwen38_27b_campaign.prepare import ARM, ROOT
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import file_hash, write_json

# Scoring-stage code only; none of it runs in train, merge or parity.
ALLOWED = {
    "src/gleipnir/campaigns/monitoring/evaluation.py",
    "src/gleipnir/serving/bf16_monitor.py",
    "src/gleipnir/serving/bf16_worker.py",
}
COMPLETED = ("train", "master-reference", "merge", "merged-reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    campaign = Campaign.load(ROOT, ROOT / "data" / ARM / "config.yaml")
    state = json.loads((campaign.output / "runner.json").read_text())
    if state["status"] != "paused_after_stage" or "evaluate" in state["stages"]:
        raise ValueError("re-binding requires a campaign paused before evaluation")
    for stage in COMPLETED:
        record = state["stages"][stage]
        if record["status"] != "complete":
            raise ValueError(f"{stage} is not complete")
        for path, expected in record["files_sha256"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"completed {stage} output changed: {path}")
    manifest_path = campaign.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bound, current = manifest["source_sha256"], campaign.sources()
    if set(bound) != set(current):
        raise ValueError("source set changed; only edits to bound files are allowed")
    changed = {n for n in bound if bound[n] != current[n]}
    if not changed or changed - ALLOWED:
        raise ValueError(f"unexpected source changes: {sorted(changed - ALLOWED)}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    shutil.copy2(manifest_path, campaign.data / f"manifest.pre_rebind_{stamp}.json")
    executed = campaign.output / "executed_sources"
    for name in sorted(changed):
        archived = campaign.output / f"executed_sources.pre_rebind_{stamp}" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executed / name, archived)
        (executed / name).write_bytes((campaign.source_root / name).read_bytes())
    manifest["source_sha256"] = current
    manifest.setdefault("source_rebindings", []).append(
        {
            "utc": stamp,
            "commit": args.commit,
            "reason": "27B BF16 serving geometry fixes after preparation; "
            "scoring-stage code only, completed stage outputs verified",
            "completed_stages_verified": list(COMPLETED),
            "files": {n: {"old": bound[n], "new": current[n]} for n in sorted(changed)},
        }
    )
    write_json(manifest_path, manifest)
    campaign.check()
    print(json.dumps(manifest["source_rebindings"][-1], indent=2))


if __name__ == "__main__":
    main()
