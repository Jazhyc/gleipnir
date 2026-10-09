"""Frozen correctness checks for the real eager pooling monitor and Lens bridge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import torch
from vllm_lens import SteeringVector
from vllm_lens._helpers._serialize import deserialize_tensor

from experiments.b200_vllm_lens.start import ROOT, write
from gleipnir.serving.monitor_score import validate_score_response
from gleipnir.serving.reference import selected_serving_default


async def smoke(out: Path, *, reproduction: dict | None = None) -> dict:
    from experiments.b200_monitor_score.run import trial

    canary = json.loads(
        (ROOT / "data/b200_inference_benchmark/canary.json").read_text()
    )
    if reproduction is None:
        selection, _ = selected_serving_default(ROOT)
        values, _ = await trial(canary, 4, {"port": 8010, "timeout_seconds": 300})
        previous = json.loads((ROOT / selection["canary_predictions"]).read_text())
        if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
            (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in previous
        ]:
            raise ValueError("eager canary token/prompt identity drift")
        scores, expected = (
            np.array([r["score"] for r in values]),
            np.array([r["score"] for r in previous]),
        )
        mae = float(np.abs(scores - expected).mean())
        correlation = float(np.corrcoef(scores, expected)[0, 1])
        master = json.loads((ROOT / selection["master_canary"]).read_text())
        effect = float(np.abs(scores - np.array(master["base"])).max())
        reproduction = {
            "score_mae": mae,
            "correlation": correlation,
            "adapter_effect": effect,
            "passed": bool(np.isfinite(scores).all())
            and mae <= 0.005
            and correlation >= 0.995
            and effect > 0,
        }
        write(out / "canary_predictions.json", values)
        reproduction["research_eager_finite"] = (
            bool(np.isfinite(scores).all()) and effect > 0
        )
        reproduction["mode"] = "user_authorized_research_eager"
        write(out / "canary.json", reproduction)
        if not reproduction["research_eager_finite"]:
            raise ValueError(
                "eager monitor failed finite output or nonzero adapter-effect checks"
            )
    row = min(canary, key=lambda r: r["prompt_tokens"])
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False, timeout=300
    ) as client:

        async def request(*, layers=None, positions="last", vectors=None, prompt=row):
            payload = {
                "model": "monitor",
                "prompt": prompt["prompt"],
                "capture_layers": layers or [],
                "capture_positions": positions,
                "steering_vectors": [v.model_dump(mode="json") for v in vectors or []],
            }
            response = await client.post("/v1/monitor/lens", json=payload)
            response.raise_for_status()
            result = response.json()
            validate_score_response(result, prompt["prompt_tokens"])
            h = (
                deserialize_tensor(result["activations"]["residual_stream"])
                if result["activations"]
                else None
            )
            if h is not None and (
                h.dtype != torch.bfloat16 or not bool(torch.isfinite(h).all())
            ):
                raise ValueError("capture native dtype/finite check failed")
            return result, h

        base, h = await request(layers=[0, 3, 31])
        if h.shape != (3, 1, 2560):
            raise ValueError("hybrid decoder capture geometry failed")
        ordinary = await client.post(
            "/v1/monitor/score", json={"model": "monitor", "prompt": row["prompt"]}
        )
        ordinary.raise_for_status()
        if ordinary.json()["logits"] != base["logits"]:
            raise ValueError("no-op capture changed eager classifier logits")
        zero = SteeringVector(activations=torch.zeros(1, 2560), layer_indices=[31])
        z, hz = await request(layers=[0, 3, 31], vectors=[zero])
        if z["logits"] != base["logits"] or not torch.equal(h, hz):
            raise ValueError("zero steering changed scores or captures")
        generator = torch.Generator().manual_seed(0)
        vector = SteeringVector(
            activations=torch.randn(1, 1, 2560, generator=generator),
            layer_indices=[31],
            position_indices=[row["prompt_tokens"] - 1],
            scale=0.5,
        )
        changed, hc = await request(layers=[0, 3, 31], vectors=[vector])
        if (
            changed["logits"] == base["logits"]
            or torch.equal(hc[-1], h[-1])
            or not torch.equal(hc[:2], h[:2])
        ):
            raise ValueError(
                "nonzero steering had no effect or changed preceding layers"
            )
        restored, hr = await request(layers=[0, 3, 31])
        if restored["logits"] != base["logits"] or not torch.equal(hr, h):
            raise ValueError("steering leaked into the next plain request")
        mixed = await asyncio.gather(
            request(layers=[31], vectors=[vector]), request(layers=[31])
        )
        plain_mixed, hm = mixed[1]
        if abs(plain_mixed["score"] - base["score"]) > 0.005 or not torch.allclose(
            hm[0], h[-1], atol=0.03, rtol=0.01
        ):
            raise ValueError("mixed steered/plain request isolation failed")
        selected, hs = await request(
            layers=[31], positions=[0, row["prompt_tokens"] - 1]
        )
        first_only = vector.model_copy(update={"position_indices": [0]})
        first_changed, hf = await request(
            layers=[31], positions=[0, row["prompt_tokens"] - 1], vectors=[first_only]
        )
        if (
            first_changed["logits"] != selected["logits"]
            or not torch.equal(hf[:, -1], hs[:, -1])
            or torch.equal(hf[:, 0], hs[:, 0])
        ):
            raise ValueError("absolute position steering changed a neighboring token")
        full = json.loads(
            (ROOT / "data/b200_inference_benchmark/full.json").read_text()
        )
        longest = max(full, key=lambda r: r["prompt_tokens"])
        chunked = await asyncio.gather(
            *(
                request(
                    layers=[0, 31],
                    positions=[0, longest["prompt_tokens"] - 1],
                    prompt=longest,
                )
                for _ in range(8)
            )
        )
        if not any(len(result["capture_chunks"]["31"]) > 1 for result, _ in chunked):
            raise ValueError("chunked prefill was not exercised")
        if any(h.shape != (2, 2, 2560) for _, h in chunked):
            raise ValueError("chunked absolute-position capture geometry failed")
        bad = await client.post(
            "/v1/monitor/lens",
            json={
                "prompt": row["prompt"],
                "capture_layers": [31],
                "capture_positions": [row["prompt_tokens"]],
            },
        )
        if bad.status_code != 400:
            raise ValueError("invalid positions were not rejected before scheduling")
        state = (await client.get("/v1/monitor/lens/info")).json()
        if state["capture_requests"] or state["steering_requests"]:
            raise ValueError("completed Lens requests retained state")
        torch.save(
            {
                "baseline": hr,
                "steered": hc,
                "vector": vector.activations,
                "prompt_sha256": row["prompt_sha256"],
            },
            out / "sample_activations.pt",
        )
        return {
            "passed": True,
            "canary": reproduction,
            "capture_layers": [0, 3, 31],
            "capture_shape": [3, 1, 2560],
            "no_op_exact": True,
            "zero_steering_exact": True,
            "nonzero_logit_change": [
                b - a for a, b in zip(base["logits"], changed["logits"], strict=True)
            ],
            "next_request_restored_exact": True,
            "mixed_plain_score_difference": plain_mixed["score"] - base["score"],
            "absolute_position_isolation": True,
            "chunked_requests": [result["capture_chunks"] for result, _ in chunked],
            "state_after": state,
        }
