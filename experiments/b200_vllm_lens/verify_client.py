"""Check the public client, full capture, norm matching and disconnect cleanup."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import torch
from vllm_lens import SteeringVector

from experiments.b200_vllm_lens.start import ROOT, write
from gleipnir.serving.lens import MonitorLensClient


async def verify(out: Path) -> None:
    canary = json.loads(
        (ROOT / "data/b200_inference_benchmark/canary.json").read_text()
    )
    row = min(canary, key=lambda r: r["prompt_tokens"])
    client = MonitorLensClient("http://127.0.0.1:8010")
    try:
        full = client.score(
            row["prompt"], capture_layers=[0, 31], capture_positions="all"
        )
        h = full["activations"]["residual_stream"]
        if h.shape != (2, row["prompt_tokens"], 2560):
            raise ValueError("public client full-token shape failed")
        last = client.score(row["prompt"], capture_layers=[0, 31])
        if not torch.equal(h[:, -1:], last["activations"]["residual_stream"]):
            raise ValueError("full/last capture values disagree")
        vector = SteeringVector(
            activations=torch.randn(
                1, 1, 2560, generator=torch.Generator().manual_seed(1)
            ),
            layer_indices=[31],
            position_indices=[row["prompt_tokens"] - 1],
            scale=0.1,
            norm_match=True,
        )
        steered = client.score(
            row["prompt"], capture_layers=[31], steering_vectors=[vector]
        )
        changed = steered["activations"]["residual_stream"][0, 0].float()
        original = h[-1, -1].float()
        ratio = float((changed - original).norm() / original.norm())
        if not 0.08 <= ratio <= 0.12:
            raise ValueError(
                "actual norm matching does not reference the full residual"
            )
    finally:
        client.close()
    rows = json.loads((ROOT / "data/b200_inference_benchmark/full.json").read_text())
    longest = max(rows, key=lambda r: r["prompt_tokens"])
    timed_out = False
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False, timeout=300
    ) as async_client:
        try:
            await async_client.post(
                "/v1/monitor/lens",
                json={
                    "prompt": longest["prompt"],
                    "capture_layers": [0, 31],
                    "capture_positions": "all",
                },
                timeout=httpx.Timeout(0.03, write=10, connect=10),
            )
        except httpx.ReadTimeout:
            timed_out = True
        if not timed_out:
            raise ValueError("disconnect was not exercised")
        await asyncio.sleep(0.5)
        for _ in range(25):
            response = await async_client.get("/v1/monitor/lens/info")
            response.raise_for_status()
            state = response.json()
            if state["capture_requests"] == state["steering_requests"] == 0:
                break
            await asyncio.sleep(0.2)
        else:
            raise ValueError("disconnected request retained Lens state")
        clean = await async_client.post(
            "/v1/monitor/score", json={"prompt": row["prompt"]}
        )
        clean.raise_for_status()
        if clean.json()["logits"] != full["logits"]:
            raise ValueError("scorer did not recover unchanged after disconnect")
    write(
        out / "client_verification.json",
        {
            "passed": True,
            "full_capture_shape": list(h.shape),
            "full_last_values_exact": True,
            "norm_matched_relative_addition": ratio,
            "requested_scale": 0.1,
            "disconnect_exercised": timed_out,
            "state_after_disconnect": state,
            "plain_score_restored_exact": True,
        },
    )
    print("monitor_lens_client_verified", ratio, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    asyncio.run(verify(args.directory))


if __name__ == "__main__":
    main()
