"""Compare changed-input replay and padded outputs to the same live model."""

import hashlib
import json

from experiments.b200_inference_benchmark.run import trial, write


async def graph_state(client, enabled=None):
    kwargs = {} if enabled is None else {"enabled": enabled}
    response = await client.post(
        "/collective_rpc",
        json={"method": "prefill_graph_state", "kwargs": kwargs, "timeout": 180},
    )
    response.raise_for_status()
    result = response.json()["results"]
    if len(result) != 1:
        raise ValueError("graph canary requires one worker")
    return result[0]


async def canary(client, ids, output):
    lengths = (127, 512, 1023, 1535, 1536, 2047, 4095, 4096, 4097, 8191, 16383, 32760)
    rows = []
    for token in (100, 200):
        for n in lengths:
            prompt = [token] * n
            rows.append(
                dict(
                    id=f"graph_{token}_{n}",
                    prompt=prompt,
                    prompt_tokens=n,
                    prompt_sha256=hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
                )
            )
    results = {}
    try:
        await graph_state(client, "false")
        results["uncaptured"], _ = await trial(client, rows, ids, 1, "base")
        await graph_state(client, "true")
        for mode in ("replay", "repeat"):
            results[mode], _ = await trial(client, rows, ids, 1, "base")
    finally:
        state = await graph_state(client, "true")
    errors = [
        abs(a["score"] - b["score"])
        for a, b in zip(results["uncaptured"], results["replay"], strict=True)
    ]
    repeats = [
        abs(a["score"] - b["score"])
        for a, b in zip(results["replay"], results["repeat"], strict=True)
    ]
    logprob_errors = [
        abs(a[key] - b[key])
        for a, b in zip(results["uncaptured"], results["replay"], strict=True)
        for key in ("logprob_0", "logprob_1")
    ]
    covered = {
        "padded_graph": any(
            d["mode"] == "PIECEWISE" and d["padded"] > d["rows"]
            for d in state["decisions"]
        ),
        "large_graph": any(
            d["mode"] == "PIECEWISE" and d["rows"] > 16384 for d in state["decisions"]
        ),
        "bypass": any(d["reason"] == "canary_bypass" for d in state["decisions"]),
        "padding_fallback": any(
            d["reason"] == "padding_or_precision_band" for d in state["decisions"]
        ),
    }
    report = {
        "passed": max(errors) <= 0.005
        and max(repeats) <= 1e-6
        and all(covered.values()),
        "max_score_difference": max(errors),
        "max_repeat_difference": max(repeats),
        "limits": {"score": 0.005, "repeat": 1e-6},
        "rows": len(rows),
        "results": results,
        "coverage": covered,
        "state": state,
        "finite_outputs": all(
            0 <= r["score"] <= 1 for v in results.values() for r in v
        ),
        "quality_selection": False,
    }
    report["passed"] &= report["finite_outputs"]
    report["max_logprob_difference"] = max(logprob_errors)
    report["limits"]["logprob"] = 0.125
    report["passed"] &= max(logprob_errors) <= 0.125
    write(output / "prefill_graph_canary.json", report)
    if not report["passed"]:
        raise ValueError("prefill graph padding/replay canary failed")
    return report
