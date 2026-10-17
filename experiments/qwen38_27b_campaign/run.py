"""Run the 27B campaign through a chosen stage (resumable)."""

from __future__ import annotations

import argparse

from experiments.qwen38_27b_campaign.prepare import ARM, ROOT
from gleipnir.campaigns.monitoring.__main__ import STAGES, execute
from gleipnir.campaigns.monitoring.contract import Campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=STAGES, default="evaluate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    campaign = Campaign.load(ROOT, ROOT / "data" / ARM / "config.yaml")
    state = execute(campaign, through=args.through, resume=args.resume)
    print(f"qwen38_27b_campaign status={state['status']}", flush=True)


if __name__ == "__main__":
    main()
