"""Real-engine checks for projection, span reduction and answer-mass readout."""

import json
from pathlib import Path

import numpy as np
import torch

from gleipnir.data.monitoring import write_json
from gleipnir.serving.lens import MonitorLensClient

ROOT = Path(__file__).resolve().parents[2]


def main():
    client = MonitorLensClient("http://127.0.0.1:8010")
    rows = json.loads((ROOT / "data/b200_inference_benchmark/canary.json").read_text())
    row = min(rows, key=lambda r: r["prompt_tokens"])
    prompt = row["prompt"]
    last = row["prompt_tokens"] - 1
    base = client.score(
        prompt,
        capture_layers=[31],
        capture_span_positions=[0, 1, last],
        full_readout=True,
    )
    allpos = client.score(prompt, capture_layers=[31], capture_positions=[0, 1, last])
    mean = allpos["activations"]["residual_stream"].float().mean(1)
    torch.testing.assert_close(
        base["activations"]["residual_span_mean"], mean, atol=1e-6, rtol=1e-6
    )
    if (
        max(
            abs(a - b)
            for a, b in zip(base["logits"], base["readout_logits"], strict=True)
        )
        > 0.25
        or not 0 < base["p01"] <= 1
    ):
        raise ValueError("full answer-mass readout failed")
    d = (np.ones(2560, np.float32) / np.sqrt(2560)).astype(np.float32)
    edit = {
        "direction": d.tolist(),
        "layer_indices": [31],
        "beta": 0.0,
        "decision_centers": [0.0] * 32,
        "span_centers": [0.0] * 32,
    }
    zero = client.score(prompt, capture_layers=[31], directional_edits=[edit])
    if zero["logits"] != base["logits"] or not torch.equal(
        zero["activations"]["residual_stream"], base["activations"]["residual_stream"]
    ):
        raise ValueError("beta-zero changed results")
    project = client.score(
        prompt, capture_layers=[31], directional_edits=[edit | {"beta": 1.0}]
    )
    before = base["activations"]["residual_stream"][0, 0].float()
    after = project["activations"]["residual_stream"][0, 0].float()
    t = torch.from_numpy(d)
    target = before - (before @ t) * t
    torch.testing.assert_close(after, target, atol=0.04, rtol=0.02)
    plain = client.score(prompt, capture_layers=[31])
    assert plain["logits"] == base["logits"] and torch.equal(
        plain["activations"]["residual_stream"], base["activations"]["residual_stream"]
    )
    info = client.info()
    assert (
        info["capture_requests"]
        == info["steering_requests"]
        == info["projection_requests"]
        == 0
    )
    write_json(
        ROOT
        / "results/b200_injection_direction/sdpa_injection_direction01"
        / "operator_smoke.json",
        {
            "passed": True,
            "no_op_exact": True,
            "span_mean_matches_raw": True,
            "full_readout_p01": base["p01"],
            "full_readout_max_logit_drift": max(
                abs(a - b)
                for a, b in zip(base["logits"], base["readout_logits"], strict=True)
            ),
            "projection_reference_max_error": float(abs(after - target).max()),
            "plain_restored_exact": True,
            "state_after": info,
        },
    )
    client.close()
    print("projection_operator_smoke_passed", flush=True)


if __name__ == "__main__":
    main()
