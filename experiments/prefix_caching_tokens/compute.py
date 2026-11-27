"""Prompt work of monitoring every action, with and without prefix caching.

A monitor that checks before every action scores the transcript up to that
action, so check ``k`` of a trajectory is the prompt prefix through the end of
its ``k``-th tool-calling assistant turn, followed by the closing suffix of the
prompt (envelope end, chat template, decision prefix). Without prefix caching
every check processes its whole prompt; with it a check processes only the
text after the previous check's prefix plus the suffix. Counts are in
characters: the ratios are what matters, and characters stand in for tokens
within one prompt.

``python -m experiments.prefix_caching_tokens.compute`` writes
``results/prefix_caching_tokens/summary.json`` and prints a table.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
OOD = ROOT / "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl"
SESSION = ROOT / "data/b200_prefix_replay/session.json"
OUTPUT = ROOT / "results/prefix_caching_tokens/summary.json"
OPEN, CLOSE = "<agent_trajectory>", "</agent_trajectory>"


def action_ends(trajectory: str) -> list[int]:
    """End offsets of the tool-calling assistant turns of one trajectory.

    Three renderings occur: ``ASSISTANT:`` turns with ``Tool calls:`` (OOD
    suite), ``[ASSISTANT]`` blocks with ``[TOOL CALLS]`` (BashBench, STRIDE,
    Gloom) and ``<assistant>`` blocks with ``<function>`` (BashArena).
    """
    ends = []
    if "<assistant>" in trajectory:
        for match in re.finditer(r"<assistant>(.*?)</assistant>", trajectory, re.S):
            if "<function>" in match.group(1):
                ends.append(match.end())
        return ends
    if "[ASSISTANT]" in trajectory:
        starts = [m.start() for m in re.finditer(r"\[ASSISTANT\]\n", trajectory)]
        next_turn = re.compile(r"\n\[(?:TOOL|USER|ASSISTANT)\]\n")
    else:
        starts = [m.start() for m in re.finditer(r"(?m)^ASSISTANT:", trajectory)]
        next_turn = re.compile(r"\n\n(?:TOOL|USER|ASSISTANT):")
    for start in starts:
        found = next_turn.search(trajectory, start + 1)
        end = found.start() if found else len(trajectory)
        turn = trajectory[start:end]
        if "[TOOL CALLS]" in turn or "Tool calls:" in turn:
            ends.append(end)
    return ends


def work(prompt: str) -> dict | None:
    """Characters processed for one trajectory under three policies."""
    if prompt.count(OPEN) != 1 or prompt.count(CLOSE) != 1:
        raise ValueError("prompt must contain one trajectory envelope")
    body_start = prompt.index(OPEN) + len(OPEN)
    body_end = prompt.index(CLOSE)
    suffix = len(prompt) - body_end
    ends = [body_start + e for e in action_ends(prompt[body_start:body_end])]
    if not ends:
        return None
    return {
        "checks": len(ends),
        "isolated": len(prompt),
        "uncached": sum(e + suffix for e in ends),
        "cached": ends[-1] + len(ends) * suffix,
    }


def session_work() -> dict:
    """The same counts for the synthetic session that ``b200_prefix_replay`` timed."""
    from experiments.b200_prefix_replay.session import content, load, policies

    session = load(SESSION)
    checks = policies(session["steps"])["every_action"]
    lengths = [len(content(session, i)) for i in checks]
    tail = len("\n</agent_trajectory>\n")
    return {
        "checks": len(checks),
        "uncached": sum(lengths),
        "cached": lengths[-1] + (len(lengths) - 1) * tail,
        "measured_time_ratio": 12.9,
    }


def summarize(rows: list[dict]) -> dict:
    ratio = [r["uncached"] / r["cached"] for r in rows]
    return {
        "trajectories": len(rows),
        "checks_median": statistics.median(r["checks"] for r in rows),
        "checks_mean": statistics.fmean(r["checks"] for r in rows),
        "uncached_over_cached": sum(r["uncached"] for r in rows)
        / sum(r["cached"] for r in rows),
        "uncached_over_cached_median": statistics.median(ratio),
        "uncached_over_isolated": sum(r["uncached"] for r in rows)
        / sum(r["isolated"] for r in rows),
        "cached_over_isolated": sum(r["cached"] for r in rows)
        / sum(r["isolated"] for r in rows),
    }


def benchmark_rows() -> list[tuple[str, str]]:
    from experiments.qwen38_27b_serving.bench import workloads
    from experiments.serving_concurrency.run import agent_only

    return [(r["dataset"], r["prompt"]) for r in agent_only(workloads()["full"])]


def ood_rows() -> list[tuple[str, str]]:
    with OOD.open() as handle:
        for line in handle:
            row = json.loads(line)
            yield row["metadata"]["source_dataset"], row["prompt"]


def main() -> None:
    out: dict = {"unit": "characters", "workloads": {}}
    for name, rows in (("benchmark96", benchmark_rows()), ("ood", ood_rows())):
        groups, skipped = defaultdict(list), 0
        for source, prompt in rows:
            counted = work(prompt)
            if counted is None:
                skipped += 1
                continue
            groups[source].append(counted)
        everything = [r for g in groups.values() for r in g]
        out["workloads"][name] = {
            "all": summarize(everything),
            "without_tool_calls": skipped,
            "by_source": {s: summarize(g) for s, g in sorted(groups.items())},
        }
    out["synthetic_session"] = session_work()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    for name, data in out["workloads"].items():
        print(f"== {name} (skipped without tool calls: {data['without_tool_calls']})")
        for source, s in {"ALL": data["all"], **data["by_source"]}.items():
            ratio, median = s["uncached_over_cached"], s["uncached_over_cached_median"]
            print(
                f"{source:32} n={s['trajectories']:5}"
                f" checks med {s['checks_median']:6.0f} mean {s['checks_mean']:6.1f}"
                f"  uncached/cached {ratio:6.1f} (median {median:5.1f})"
                f"  cached/isolated {s['cached_over_isolated']:.2f}"
            )
    s = out["synthetic_session"]
    print(
        f"synthetic session: {s['checks']} checks, uncached/cached "
        f"{s['uncached'] / s['cached']:.1f} (measured time ratio 12.9)"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
