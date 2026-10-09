"""Request contracts and an optional Lens client for causal monitor scoring."""

from __future__ import annotations

import math
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from gleipnir.serving.monitor_score import MonitorScoreRequest, validate_score_response

ENDPOINT = "/v1/monitor/lens"
KEY = "gleipnir_lens"
MAX_CAPTURE_BYTES = 512 * 1024 * 1024


class DirectionalEdit(BaseModel):
    """A single unit residual direction, clamped at selected decoder layers."""

    model_config = ConfigDict(extra="forbid")
    direction: list[float]
    layer_indices: list[StrictInt]
    beta: float = 1.0
    decision_centers: list[float]
    span_centers: list[float]


class LensScoreRequest(MonitorScoreRequest):
    """Capture and steer only this complete rendered monitor prompt."""

    model_config = ConfigDict(extra="forbid")
    capture_layers: list[StrictInt] = Field(default_factory=list)
    capture_positions: Literal["all", "last"] | list[StrictInt] = "last"
    steering_vectors: list[dict[str, Any]] = Field(default_factory=list)
    directional_edits: list[DirectionalEdit] = Field(default_factory=list, max_length=1)
    capture_span_positions: list[StrictInt] = Field(default_factory=list)
    full_readout: bool = False

    @field_validator("capture_layers", "capture_positions", "capture_span_positions")
    @classmethod
    def unique_nonnegative(cls, value):
        if isinstance(value, list) and (
            len(set(value)) != len(value) or any(i < 0 for i in value)
        ):
            raise ValueError("layers and positions must be unique nonnegative integers")
        return sorted(value) if isinstance(value, list) else value


def validate_intervention(
    request: LensScoreRequest, *, layers: int, hidden: int, tokens: int
) -> list:
    """Validate vector geometry and capture size before admitting a request."""
    import torch
    from vllm_lens import SteeringVector

    if any(i >= layers for i in request.capture_layers):
        raise ValueError("capture layer outside model")
    positions = request.capture_positions
    if request.capture_layers and positions == []:
        raise ValueError("capture positions must be nonempty")
    if isinstance(positions, list) and any(i >= tokens for i in positions):
        raise ValueError("capture position outside prompt")
    n = tokens if positions == "all" else 1 if positions == "last" else len(positions)
    if len(request.capture_layers) * n * hidden * 2 > MAX_CAPTURE_BYTES:
        raise ValueError("capture exceeds 512 MiB; select fewer layers or positions")
    if request.capture_span_positions and (
        not request.capture_layers
        or any(i >= tokens for i in request.capture_span_positions)
    ):
        raise ValueError("span capture requires valid positions and layers")
    if request.full_readout and (
        layers - 1 not in request.capture_layers or positions != "last"
    ):
        raise ValueError("full readout requires final-layer last-token capture")
    if request.directional_edits and request.steering_vectors:
        raise ValueError("composition of projection and addition is not supported")
    for edit in request.directional_edits:
        d = torch.tensor(edit.direction, dtype=torch.float32)
        if (
            d.shape != (hidden,)
            or not bool(torch.isfinite(d).all())
            or abs(float(d.norm()) - 1.0) > 1e-3
            or not math.isfinite(edit.beta)
            or not 0 <= edit.beta <= 1
            or not edit.layer_indices
            or len(set(edit.layer_indices)) != len(edit.layer_indices)
            or any(not 0 <= i < layers for i in edit.layer_indices)
            or len(edit.decision_centers) != layers
            or len(edit.span_centers) != layers
            or not all(
                math.isfinite(c) for c in edit.decision_centers + edit.span_centers
            )
        ):
            raise ValueError("invalid unit direction, clamp centers or beta")
    vectors = []
    allowed = {
        "activations",
        "layer_indices",
        "scale",
        "norm_match",
        "position_indices",
    }
    for value in request.steering_vectors:
        if set(value) - allowed:
            raise ValueError("unsupported steering fields")
        vector = SteeringVector.model_validate(value)
        act = vector.activations
        if (
            not vector.layer_indices
            or len(set(vector.layer_indices)) != len(vector.layer_indices)
            or any(
                type(i) is not int or not 0 <= i < layers for i in vector.layer_indices
            )
            or act.shape[-1] != hidden
            or not act.is_floating_point()
            or not bool(torch.isfinite(act).all())
            or not math.isfinite(vector.scale)
        ):
            raise ValueError("invalid steering geometry or nonfinite values")
        if act.dim() == 2 and vector.position_indices is not None:
            raise ValueError("position-specific steering requires a 3D vector")
        if act.dim() == 3:
            indices = vector.position_indices
            if indices is None:
                indices = list(range(act.shape[1]))
            if (
                not indices
                or len(indices) != act.shape[1]
                or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < tokens for i in indices)
            ):
                raise ValueError("steering positions outside prompt or wrong count")
        vectors.append(vector)
    return vectors


class MonitorLensClient:
    """Use Lens vectors and decode its native tensor format on the score endpoint."""

    def __init__(self, base_url: str, *, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def score(
        self,
        prompt: str,
        *,
        model: str = "monitor",
        capture_layers: list[int] | None = None,
        capture_positions: Literal["all", "last"] | list[int] = "last",
        steering_vectors: list | None = None,
        directional_edits: list[dict] | None = None,
        capture_span_positions: list[int] | None = None,
        full_readout: bool = False,
    ) -> dict[str, Any]:
        """Score one prompt, optionally returning selected residual activations."""
        from vllm_lens._helpers._serialize import deserialize_tensor

        request = LensScoreRequest(
            prompt=prompt,
            model=model,
            capture_layers=capture_layers or [],
            capture_positions=capture_positions,
            directional_edits=directional_edits or [],
            capture_span_positions=capture_span_positions or [],
            full_readout=full_readout,
            steering_vectors=[
                v.model_dump(mode="json") for v in steering_vectors or []
            ],
        )
        response = self.session.post(
            self.base_url + ENDPOINT,
            json=request.model_dump(mode="json"),
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        validate_score_response(result, result["prompt_tokens"])
        result["activations"] = {
            k: deserialize_tensor(v) for k, v in result.get("activations", {}).items()
        }
        return result

    def info(self) -> dict[str, Any]:
        """Read the active research model shape and remaining request-state counts."""
        response = self.session.get(
            self.base_url + ENDPOINT + "/info", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()
