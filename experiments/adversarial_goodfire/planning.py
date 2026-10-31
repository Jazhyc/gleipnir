"""Token-scaled compute and API estimates. Reads historical receipts only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from experiments.adversarial_goodfire.common import (
    DATA,
    ROOT,
    config,
    read_jsonl,
    write_json,
)
from experiments.adversarial_goodfire.download import sha256
from experiments.adversarial_goodfire.frontier import monitor


def main() -> None:
    argparse.ArgumentParser(__doc__).parse_args()
    cfg = config()
    prepared = DATA / "prepared"
    selected = read_jsonl(prepared / "selection.jsonl")
    groups = defaultdict(list)
    for r in selected:
        groups[f"{r['benchmark']}:{r['agent']}"].append(r)
    counts = {}
    for key, rows in groups.items():
        counts[key] = {
            "labels": dict(Counter(r["label_class"] for r in rows)),
            "positive_tasks": len(
                {r["task_group"] for r in rows if r["label_class"] == "positive"}
            ),
            "negative_tasks": len(
                {r["task_group"] for r in rows if r["label_class"] == "negative"}
            ),
            "matched_negatives": sum(
                r["label_class"] == "negative" and r["matched_positive_task"]
                for r in rows
            ),
            "usable_labels": {
                m: dict(
                    Counter(
                        r["label_class"]
                        for r in rows
                        if r.get("render_status") == "ok" and not r["too_long"][m]
                    )
                )
                for m in cfg["monitors"]
            },
        }
    for path, digest in cfg["gpu_estimate"]["reference_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Step 1 reference receipt changed")
    gpu = {}
    for m in cfg["monitors"]:
        usable = [
            r
            for r in selected
            if r.get("render_status") == "ok" and not r["too_long"][m]
        ]
        tokens = sum(r["tokens"][m] for r in usable)
        all_tokens = sum(r.get("tokens", {}).get(m, 0) for r in selected)
        ref = cfg["gpu_estimate"]["reference_prompt_tokens"][m]
        rate = cfg["gpu_estimate"][
            "4b_seconds_per_row" if m == "4b-v03" else "27b_seconds_per_row"
        ]
        gpu[m] = {
            "usable_rows": len(usable),
            "too_long": sum(r.get("too_long", {}).get(m, False) for r in selected),
            "usable_prompt_tokens": tokens,
            "all_prompt_tokens": all_tokens,
            "reference_prompt_tokens_per_row": ref,
            "reference_seconds_per_row": rate,
            "estimated_scoring_seconds": tokens / ref * rate,
            "counterfactual_all_lengths_seconds": all_tokens / ref * rate,
            "basis": (
                "Linear token scaling of Step 1 rates; excludes load/canaries/setup "
                "and pending adjudication exclusions."
            ),
        }
    plans = read_jsonl(prepared / "frontier/plans.jsonl")
    api = {}
    for name in cfg["frontier"]["monitors"]:
        m = monitor(name, cfg)
        own = [r for r in plans if r["monitor"] == name]
        pending = [r for r in own if r["status"] == "pending_authorization"]
        proxy = sum(r["qwen_prompt_tokens_proxy"] for r in pending)
        # Sol calibration replay has provider-counted historical lengths for
        # the exact same Score prompt. Reasoning usage is still uncertain.
        replay = (
            read_jsonl(ROOT / cfg["calibration"][name]["scores"])
            if name == "gpt-6.1-sol"
            else []
        )
        replay = [
            r
            for r in replay
            if r.get("code") == "honest"
            and r.get("text") == "none"
            and r.get("set") == "main"
            and r.get("split") == "heldout"
        ]
        calibration_tokens = sum(r["prompt_tokens"] for r in replay)
        expected_output = 10 if name == "kimi-k3" else 128
        point = m.cost_from_usage(
            proxy + calibration_tokens, (len(pending) + len(replay)) * expected_output
        )
        replay_plans = (
            read_jsonl(prepared / "calibration/sol_replay.jsonl")
            if name == "gpt-6.1-sol"
            else []
        )
        replay_bound = sum(
            m.worst_case_usd(
                len(r["payload"]["messages"][0]["content"].encode()) + 2048
            )
            for r in replay_plans
        )
        api[name] = {
            "selected_rows": len(own),
            "dispositions": dict(Counter(r["status"] for r in own)),
            "proposed_requests": len(pending),
            "qwen_input_token_proxy": proxy,
            "calibration_replay_rows": len(replay),
            "calibration_input_tokens": calibration_tokens,
            "expected_output_tokens_per_request": expected_output,
            "planning_point_usd": point,
            "token_proxy_range_usd": [
                m.cost_from_usage(
                    proxy * factor + calibration_tokens,
                    (len(pending) + len(replay)) * expected_output,
                )
                for factor in (0.5, 1.5)
            ],
            "study_byte_bound_max_output_usd": sum(
                r["worst_case_usd"] for r in pending
            ),
            "calibration_byte_bound_max_output_usd": replay_bound,
            "total_byte_bound_max_output_usd": replay_bound
            + sum(r["worst_case_usd"] for r in pending),
            "cap_usd": cfg["frontier"]["monitors"][name]["cap_usd"],
            "basis": (
                "Historical crossed-text prices; provider token counts for Goodfire "
                "unknown; no cache discounts."
            ),
        }
    write_json(
        prepared / "planning/receipt.json",
        {
            "counts": counts,
            "gpu": gpu,
            "api": api,
            "scoring_performed": False,
            "gpu_setup_allowance_minutes": [20, 40],
            "timing_uncertainty_multiplier": [1, 2],
        },
    )
    print("offline planning estimates written, no scores")


if __name__ == "__main__":
    main()
