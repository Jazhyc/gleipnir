"""Opt-in Lens HTTP routing around the audited monitor classification endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import uuid

from gleipnir.serving.lens import ENDPOINT, KEY, LensScoreRequest, validate_intervention


def install_lens_api() -> None:
    """Add request-scoped research routes without patching the standard score API."""
    if importlib.metadata.version("vllm") != "0.31.0":
        raise ValueError("monitor Lens bridge requires vLLM 0.31.0")
    if importlib.metadata.version("vllm-lens") != "1.3.0":
        raise ValueError("monitor Lens bridge requires Lens 1.3.0")
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.pooling import factories
    from vllm.entrypoints.pooling.classify.protocol import (
        ClassificationCompletionRequest,
    )
    from vllm.entrypoints.serve.utils.api_utils import with_cancellation
    from vllm.exceptions import VLLMNotFoundError

    from gleipnir.serving.monitor_score import install_score_api

    # The existing implementation/source bindings remain untouched.
    if not getattr(factories, "_gleipnir_monitor_score", False):
        install_score_api()
    original_init = factories.init_pooling_state
    original_register = factories.register_pooling_api_routers

    class LensClassificationRequest(ClassificationCompletionRequest):
        lens_config: dict

        def to_pooling_params(self):
            params = super().to_pooling_params()
            params.extra_kwargs = {KEY: self.lens_config}
            params.skip_reading_prefix_cache = True
            return params

    def initialize(engine_client, state, args, request_logger, supported_tasks):
        original_init(engine_client, state, args, request_logger, supported_tasks)
        state.gleipnir_lens_lock = asyncio.Lock()
        state.gleipnir_lens_ready = False

    async def prepare(state):
        async with state.gleipnir_lens_lock:
            engine = state.gleipnir_monitor_score.engine_client
            if not state.gleipnir_lens_ready:
                info = await engine.collective_rpc("lens_install")
                if len(info) != 1:
                    raise ValueError("monitor Lens supports one worker")
                state.gleipnir_lens_shape = info[0]
                state.gleipnir_lens_ready = True
            return engine, state.gleipnir_lens_shape

    async def info(raw_request):
        engine, _ = await prepare(raw_request.app.state)
        return (await engine.collective_rpc("lens_info"))[0]

    async def score(request, raw_request):
        state = raw_request.app.state
        engine, shape = await prepare(state)
        service = state.gleipnir_monitor_score
        key = uuid.uuid4().hex
        try:
            # Use the same native renderer as the scorer to validate token bounds
            # before enqueueing; do not change or truncate the rendered prompt.
            token_ids = service.renderer._encode(
                request.prompt, add_special_tokens=False
            )
            vectors = validate_intervention(
                request,
                layers=len(shape["layers"]),
                hidden=shape["hidden_size"],
                tokens=len(token_ids),
            )
            if len(token_ids) > service.max_model_len:
                raise ValueError("monitor Lens prompt exceeds the context limit")
            config = {
                "request_id": key,
                "capture_layers": request.capture_layers,
                "capture_positions": request.capture_positions,
                "capture_span_positions": request.capture_span_positions,
                "full_readout": request.full_readout,
                "directional_edits": [
                    e.model_dump(mode="json") for e in request.directional_edits
                ],
                "steering_vectors": [v.model_dump(mode="json") for v in vectors],
            }
            config["projection_identity"] = hashlib.sha256(
                json.dumps(config["directional_edits"], sort_keys=True).encode()
            ).hexdigest()
            internal = LensClassificationRequest(
                model=request.model,
                input=request.prompt,
                add_special_tokens=False,
                use_activation=False,
                lens_config=config,
            )
            response = await service(internal, raw_request)
            if response.status_code != 200:
                return response
            result = json.loads(response.body)
            captured = (
                await engine.collective_rpc(
                    "lens_collect", args=(key, request.full_readout)
                )
            )[0]
            if captured["activation_layers"] != request.capture_layers:
                raise RuntimeError("requested Lens layers did not all capture")
            positions = request.capture_positions
            expected = (
                list(range(len(token_ids)))
                if positions == "all"
                else [len(token_ids) - 1]
                if positions == "last"
                else positions
            )
            if request.capture_layers and captured["activation_positions"] != expected:
                raise RuntimeError("requested Lens token positions did not all capture")
            if request.capture_span_positions and captured.get(
                "span_capture_count"
            ) != len(request.capture_span_positions):
                raise RuntimeError("span mean capture coverage changed")
            result.update(captured)
            result["lens_request_id"] = key
            return JSONResponse(result)
        except VLLMNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            # Cancellation must finish vLLM's generator abort before this service
            # releases its worker buffers; shielding lets cleanup finish on disconnect.
            await asyncio.shield(engine.collective_rpc("lens_clear", args=(key,)))

    score.__annotations__ = {"request": LensScoreRequest, "raw_request": Request}
    info.__annotations__ = {"raw_request": Request}

    def register(app, supported_tasks, model_config=None):
        original_register(app, supported_tasks, model_config)
        if "classify" not in supported_tasks:
            raise ValueError("monitor Lens requires classify support")
        app.post(ENDPOINT)(with_cancellation(score))
        app.get(ENDPOINT + "/info")(info)

    factories.init_pooling_state = initialize
    factories.register_pooling_api_routers = register
