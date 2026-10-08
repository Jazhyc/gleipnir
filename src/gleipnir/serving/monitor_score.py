"""Two-logit monitoring through vLLM's causal classification runner."""

from __future__ import annotations

import copy
import importlib.metadata
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

DECISION_TOKENS = ["0", "1"]
DECISION_IDS = [15, 16]
ENDPOINT = "/v1/monitor/score"


class MonitorScoreRequest(BaseModel):
    """One complete, already rendered monitoring prompt; no generation options."""

    model_config = ConfigDict(extra="forbid")
    model: StrictStr = "monitor"
    prompt: StrictStr = Field(min_length=1)


def classification_overrides(hf_config: dict[str, Any]) -> dict[str, Any]:
    """Preserve Qwen's complete text config when adding the two-row classifier."""
    text = copy.deepcopy(hf_config.get("text_config", hf_config))
    text.update(
        classifier_from_token=DECISION_TOKENS.copy(),
        method="no_post_processing",
        num_labels=2,
        is_causal=True,
    )
    result = {
        "classifier_from_token": DECISION_TOKENS.copy(),
        "method": "no_post_processing",
        "head_dtype": "model",
        "is_causal": True,
        "num_labels": 2,
    }
    if "text_config" in hf_config:
        result["text_config"] = text
    return result


def score_payload(logits: list[float], prompt_tokens: int) -> dict[str, Any]:
    """Normalize exactly two finite logits, without a vocabulary softmax."""
    if len(logits) != 2 or not all(math.isfinite(x) for x in logits):
        raise ValueError("monitoring requires two finite decision logits")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("monitoring requires a nonempty untruncated prompt")
    margin = float(logits[1]) - float(logits[0])
    if not math.isfinite(margin):
        raise ValueError("nonfinite monitoring margin")
    exp = math.exp(-abs(margin))
    score = 1.0 / (1.0 + exp) if margin >= 0 else exp / (1.0 + exp)
    return {
        "score": score,
        "margin": margin,
        "logits": logits,
        "prompt_tokens": prompt_tokens,
    }


def validate_score_response(response: dict[str, Any], prompt_tokens: int) -> dict:
    """Reject truncation, malformed logits and inconsistent probabilities."""
    if response["prompt_tokens"] != prompt_tokens:
        raise ValueError("monitor prompt truncation or token mismatch")
    expected = score_payload(response["logits"], response["prompt_tokens"])
    for key in ("score", "margin"):
        if not math.isfinite(response[key]) or not math.isclose(
            response[key], expected[key], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"inconsistent monitor {key}")
    return {"score": response["score"], "margin": response["margin"]}


def install_score_api() -> None:
    """Register a score-only response around the supported classification API."""
    if importlib.metadata.version("vllm") not in {"0.24.0", "0.31.0"}:
        raise ValueError("monitor scoring requires vLLM 0.24.0 or 0.31.0")

    from fastapi import Depends, HTTPException, Request
    from fastapi.responses import JSONResponse, Response
    from vllm.entrypoints.pooling import factories
    from vllm.entrypoints.pooling.classify.protocol import (
        ClassificationCompletionRequest,
    )
    from vllm.entrypoints.pooling.classify.serving import ServingClassification
    from vllm.entrypoints.serve.utils.api_utils import (
        load_aware_call,
        validate_json_request,
        with_cancellation,
    )
    from vllm.exceptions import VLLMNotFoundError

    if getattr(factories, "_gleipnir_monitor_score", False):
        raise RuntimeError("monitor score API already installed")

    class MonitorServing(ServingClassification):
        request_id_prefix = "monitor-score"

        def _build_response(self, ctx):
            if len(ctx.final_res_batch) != 1:
                raise RuntimeError("monitor endpoint requires one complete prompt")
            result = ctx.final_res_batch[0]
            logits = result.outputs.data.tolist()
            return JSONResponse(score_payload(logits, len(result.prompt_token_ids)))

    original_init = factories.init_pooling_state
    original_register = factories.register_pooling_api_routers

    def initialize(engine_client, state, args, request_logger, supported_tasks):
        original_init(engine_client, state, args, request_logger, supported_tasks)
        classification = state.serving_classification
        if classification is None:
            raise ValueError("monitor endpoint requires a classification runner")
        state.gleipnir_monitor_score = MonitorServing(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template_config=classification.chat_template_config,
        )

    # Resolve local imports before FastAPI evaluates annotations (future annotations).
    async def score(request, raw_request):
        internal = ClassificationCompletionRequest(
            model=request.model,
            input=request.prompt,
            add_special_tokens=False,
            use_activation=False,
        )
        try:
            return await raw_request.app.state.gleipnir_monitor_score(
                internal, raw_request
            )
        except VLLMNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    score.__annotations__ = {
        "request": MonitorScoreRequest,
        "raw_request": Request,
        "return": Response,
    }
    wrapped = with_cancellation(load_aware_call(score))

    def register(app, supported_tasks, model_config=None):
        original_register(app, supported_tasks, model_config)
        if "classify" not in supported_tasks:
            raise ValueError("monitor endpoint requires classify support")
        app.post(ENDPOINT, dependencies=[Depends(validate_json_request)])(wrapped)

    factories.init_pooling_state = initialize
    factories.register_pooling_api_routers = register
    factories._gleipnir_monitor_score = True
