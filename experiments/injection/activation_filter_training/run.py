"""Run the two frozen data-removal conditions sequentially on the existing B200."""

from pathlib import Path

from experiments.injection.activation_filter_training.analyze import analyze
from gleipnir.campaigns.monitoring.__main__ import execute
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.repository import repo_root

ROOT = repo_root(__file__)
HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    for name in ["ranked", "random"]:
        ctx = Campaign.load(ROOT, HERE / f"{name}.yaml")
        print(
            f"activation_filter_campaign_start {ctx.config['campaign_id']}", flush=True
        )
        execute(ctx)
    analyze()
