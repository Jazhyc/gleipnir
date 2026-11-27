"""Check graph replay and padding with changed token-ID classifier inputs."""

import math

from experiments.b200_attention_gdn_serving.prefill_graph_canary import graph_state
from experiments.b200_inference_benchmark.run import write
from gleipnir.serving.monitor_score import score_payload


async def canary(client, output):
    lengths = (127, 512, 1023, 1535, 1536, 2047, 4095, 4096, 4097, 8191, 16383, 32760)
    results = {}
    try:
        for mode, enabled in (
            ("bypass", "false"),
            ("replay", "true"),
            ("repeat", "true"),
        ):
            await graph_state(client, enabled)
            values = []
            for token in (100, 200):
                for length in lengths:
                    response = await client.post(
                        "/classify",
                        json={"model": "monitor", "input": [token] * length},
                    )
                    response.raise_for_status()
                    raw = response.json()
                    if raw["usage"]["prompt_tokens"] != length or len(raw["data"]) != 1:
                        raise ValueError("graph fixture truncated or split")
                    values.append(
                        {
                            "id": f"{token}_{length}",
                            **score_payload(raw["data"][0]["probs"], length),
                        }
                    )
            results[mode] = values
    finally:
        state = await graph_state(client, "true")
    score_error = max(
        abs(a["score"] - b["score"])
        for a, b in zip(results["bypass"], results["replay"], strict=True)
    )
    repeat_error = max(
        abs(a["score"] - b["score"])
        for a, b in zip(results["replay"], results["repeat"], strict=True)
    )
    logit_error = max(
        abs(x - y)
        for a, b in zip(results["bypass"], results["replay"], strict=True)
        for x, y in zip(a["logits"], b["logits"], strict=True)
    )
    covered = {
        "padded_graph": any(
            d["mode"] == "PIECEWISE" and d["padded"] > d["rows"]
            for d in state["decisions"]
        ),
        "large_graph": any(
            d["mode"] == "PIECEWISE" and d["rows"] > 16384 for d in state["decisions"]
        ),
        "bypass": any(d["reason"] == "canary_bypass" for d in state["decisions"]),
        "fallback": any(
            d["reason"] == "padding_or_precision_band" for d in state["decisions"]
        ),
    }
    finite = all(
        math.isfinite(x)
        for values in results.values()
        for v in values
        for x in [v["score"], *v["logits"]]
    )
    report = dict(
        passed=finite
        and score_error <= 0.005
        and repeat_error <= 1e-6
        and logit_error <= 0.125
        and all(covered.values()),
        max_score_difference=score_error,
        max_repeat_difference=repeat_error,
        max_logit_difference=logit_error,
        finite_outputs=finite,
        coverage=covered,
        state=state,
        results=results,
        limits=dict(score=0.005, repeat=1e-6, logit=0.125),
        quality_selection=False,
    )
    write(output / "graph_canary.json", report)
    if not report["passed"]:
        raise ValueError("graph padding/replay gate failed")
