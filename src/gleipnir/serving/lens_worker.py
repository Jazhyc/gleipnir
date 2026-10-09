"""Pooling request routing for Lens's native residual steering and tensor format."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.forward_context import is_forward_context_available
from vllm_lens import SteeringVector
from vllm_lens._helpers._serialize import serialize_tensor
from vllm_lens._worker_ext import _apply_steering, _discover_layer_modules

from gleipnir.serving.lens import KEY
from gleipnir.serving.lens_projection import apply_projection, apply_projection_batch


def residual_stream(output: Any) -> torch.Tensor:
    """Match Lens's post-layer semantics for Qwen's deferred residual addition."""
    if isinstance(output, tuple):
        return output[0] if output[1] is None else output[0] + output[1]
    return output


def intervene_slice(
    output: Any, vectors: list, layer: int, start: int, end: int, absolute_start: int
) -> Any:
    """Apply upstream steering to one request, preserving every neighboring row."""
    norm_ref = residual_stream(output)
    modified = (
        (output[0].clone(), output[1]) if isinstance(output, tuple) else output.clone()
    )
    target = modified[0] if isinstance(modified, tuple) else modified
    _apply_steering(vectors, layer, target, start, end, absolute_start, norm_ref)
    return modified


class MonitorLensExtension:
    """Explicit TP=PP=1 research extension; no generation plugin patches."""

    def lens_install(self) -> dict:
        """Install post-layer hooks once on the audited eager pooling backbone."""
        if getattr(self, "_monitor_lens_installed", False):
            return self.lens_info()
        if (
            not self.model_config.enforce_eager
            or self.model_config.runner_type != "pooling"
            or self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
            or self.vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError(
                "monitor Lens requires eager single-GPU pooling, prefix cache off"
            )
        layer_map = _discover_layer_modules(self)
        self._lens_layers = sorted(layer_map)
        self._lens_buffers = {}
        self._lens_vectors = {}
        self._lens_edits = {}
        self._lens_handles = []
        for index, module in layer_map.items():

            def hook(_module, _inputs, output, *, layer=index):
                return self._lens_forward(layer, output)

            self._lens_handles.append(module.register_forward_hook(hook))
        self._monitor_lens_installed = True
        return self.lens_info()

    def lens_info(self) -> dict:
        return {
            "worker_pid": os.getpid(),
            "layers": self._lens_layers,
            "hidden_size": self.model_config.get_hidden_size(),
            "eager": True,
            "capture_requests": len(self._lens_buffers),
            "steering_requests": len(self._lens_vectors),
            "projection_requests": len(getattr(self, "_lens_edits", {})),
            "semantics": "post-layer residual stream before final normalization",
        }

    def _lens_forward(self, layer: int, output: Any) -> Any:
        if not is_forward_context_available():
            return None
        runner = self.model_runner
        active = []
        for i, request_id in enumerate(
            runner.input_batch.req_ids[: runner.input_batch.num_reqs]
        ):
            state = runner.requests[request_id]
            params = state.pooling_params
            config = (
                (params.extra_kwargs or {}).get(KEY) if params is not None else None
            )
            if config and (
                layer in config["capture_layers"]
                or any(layer in v["layer_indices"] for v in config["steering_vectors"])
                or any(
                    layer in e["layer_indices"]
                    for e in config.get("directional_edits", [])
                )
            ):
                active.append((i, state, config))
        if not active:
            return None
        # These pinned CPU buffers also construct V1's actual input positions.
        # Optimized attention metadata may omit token lengths or count KV pages.
        count = runner.input_batch.num_reqs
        boundaries = runner.query_start_loc.np[: count + 1].tolist()
        offsets = runner.input_batch.num_computed_tokens_cpu[:count].tolist()
        lengths = [offsets[i] + boundaries[i + 1] - boundaries[i] for i in range(count)]
        source = residual_stream(output)
        modified = output
        projected = set()
        projected_active = [
            (i, state, cfg)
            for i, state, cfg in active
            if any(
                layer in e["layer_indices"] for e in cfg.get("directional_edits", [])
            )
        ]
        if (
            len(projected_active) == count
            and len({cfg["projection_identity"] for _, _, cfg in projected_active}) == 1
        ):
            first = projected_active[0][2]
            key = first["request_id"]
            if key not in self._lens_edits:
                self._lens_edits[key] = [
                    e
                    | {
                        "direction": torch.tensor(
                            e["direction"], device=source.device, dtype=torch.float32
                        )
                    }
                    for e in first["directional_edits"]
                ]
            edit = self._lens_edits[key][0]
            finals = []
            for i, state, cfg in projected_active:
                self._lens_edits[cfg["request_id"]] = self._lens_edits[key]
                last = len(state.prompt_token_ids) - 1
                if offsets[i] <= last < lengths[i]:
                    finals.append(boundaries[i] + last - offsets[i])
                projected.add(cfg["request_id"])
            modified = (
                (output[0].clone(), output[1])
                if isinstance(output, tuple)
                else output.clone()
            )
            target = modified[0] if isinstance(modified, tuple) else modified
            apply_projection_batch(
                target,
                source,
                edit["direction"],
                edit["beta"],
                edit["decision_centers"][layer],
                edit["span_centers"][layer],
                boundaries[-1],
                finals,
            )
        for index, _state, config in active:
            key = config["request_id"]
            start, end = boundaries[index : index + 2]
            absolute_start = offsets[index]
            if absolute_start < 0 or end > source.shape[0]:
                raise RuntimeError("invalid pooling chunk boundaries")
            edits = config.get("directional_edits", [])
            if key not in projected and any(layer in e["layer_indices"] for e in edits):
                if key not in self._lens_edits:
                    self._lens_edits[key] = [
                        e
                        | {
                            "direction": torch.tensor(
                                e["direction"],
                                device=source.device,
                                dtype=torch.float32,
                            )
                        }
                        for e in edits
                    ]
                if modified is output:
                    modified = (
                        (output[0].clone(), output[1])
                        if isinstance(output, tuple)
                        else output.clone()
                    )
                target = modified[0] if isinstance(modified, tuple) else modified
                for edit in self._lens_edits[key]:
                    if layer in edit["layer_indices"]:
                        apply_projection(
                            target,
                            source,
                            edit["direction"],
                            edit["beta"],
                            edit["decision_centers"][layer],
                            edit["span_centers"][layer],
                            start,
                            end,
                            absolute_start,
                            len(_state.prompt_token_ids) - 1,
                        )
            if any(layer in v["layer_indices"] for v in config["steering_vectors"]):
                if key not in self._lens_vectors:
                    vectors = [
                        SteeringVector.model_validate(v)
                        for v in config["steering_vectors"]
                    ]
                    self._lens_vectors[key] = [
                        v.model_copy(
                            update={
                                "activations": v.activations.to(
                                    device=source.device, dtype=source.dtype
                                )
                            }
                        )
                        for v in vectors
                    ]
                if modified is output:
                    modified = (
                        (output[0].clone(), output[1])
                        if isinstance(output, tuple)
                        else output.clone()
                    )
                target = modified[0] if isinstance(modified, tuple) else modified
                _apply_steering(
                    self._lens_vectors[key],
                    layer,
                    target,
                    start,
                    end,
                    absolute_start,
                    source,
                )
        hidden = source if modified is output else residual_stream(modified)
        for index, state, config in active:
            key = config["request_id"]
            start, end = boundaries[index : index + 2]
            absolute_start = offsets[index]
            if layer not in config["capture_layers"]:
                continue
            buffer = self._lens_buffers.setdefault(key, {})
            record = buffer.setdefault(
                layer, {"positions": [], "values": [], "chunks": []}
            )
            record["chunks"].append([absolute_start, lengths[index]])
            if config.get("full_readout") and layer == self._lens_layers[-1]:
                final_position = len(state.prompt_token_ids) - 1
                if absolute_start <= final_position < lengths[index]:
                    at = start + final_position - absolute_start
                    delta = modified[0] if isinstance(modified, tuple) else modified
                    record["readout_delta"] = delta[at : at + 1].detach().cpu()
                    if isinstance(modified, tuple) and modified[1] is not None:
                        record["readout_residual"] = (
                            modified[1][at : at + 1].detach().cpu()
                        )

            span = [
                p
                for p in config.get("capture_span_positions", [])
                if absolute_start <= p < lengths[index]
            ]
            if span:
                indices = torch.tensor(
                    [start + p - absolute_start for p in span], device=hidden.device
                )
                total = hidden.index_select(0, indices).float().sum(0).detach().cpu()
                record["span_sum"] = record.get("span_sum", 0) + total
                record["span_count"] = record.get("span_count", 0) + len(span)

            positions = config["capture_positions"]
            if positions == "last":
                positions = [len(state.prompt_token_ids) - 1]
            elif positions == "all":
                positions = list(range(absolute_start, lengths[index]))
            positions = [p for p in positions if absolute_start <= p < lengths[index]]
            if positions:
                indices = torch.tensor(
                    [start + p - absolute_start for p in positions],
                    device=hidden.device,
                )
                values = hidden.index_select(0, indices).detach().cpu()
                if not bool(torch.isfinite(values).all()):
                    raise RuntimeError("nonfinite Lens capture")
                record["positions"].extend(positions)
                record["values"].append(values)
        return modified if modified is not output else None

    def lens_collect(self, key: str, full_readout: bool = False) -> dict:
        """Transfer this request's native BF16 captures and remove its state."""
        records = self._lens_buffers.pop(key, {})
        self._lens_vectors.pop(key, None)
        getattr(self, "_lens_edits", {}).pop(key, None)
        if not records:
            return {
                "activations": {},
                "activation_layers": [],
                "activation_positions": [],
            }
        layers = sorted(records)
        positions = records[layers[0]]["positions"]
        if (
            not positions
            or len(set(positions)) != len(positions)
            or any(records[i]["positions"] != positions for i in layers)
        ):
            raise RuntimeError(
                "missing, repeated or mismatched captured token positions"
            )
        tensor = torch.stack([torch.cat(records[i]["values"], dim=0) for i in layers])
        activations = {"residual_stream": serialize_tensor(tensor)}
        extra = {}
        counts = [records[i].get("span_count", 0) for i in layers]
        if any(counts):
            if len(set(counts)) != 1 or not all(counts):
                raise RuntimeError("span mean coverage differs across layers")
            mean = torch.stack([records[i]["span_sum"] / counts[0] for i in layers])
            if not bool(torch.isfinite(mean).all()):
                raise RuntimeError("nonfinite span means")
            activations["residual_span_mean"] = serialize_tensor(mean)
            extra["span_capture_count"] = counts[0]
        if full_readout:
            if layers[-1] != self._lens_layers[-1] or tensor.shape[1] != 1:
                raise ValueError("full readout requires final residual")
            model = self.model_runner.get_model()
            norms = [
                m
                for n, m in model.named_modules()
                if n.endswith(".norm") and ".layers." not in n
            ]
            embeddings = [
                m for n, m in model.named_modules() if n.endswith("embed_tokens")
            ]
            if len(norms) != 1 or len(embeddings) != 1:
                raise ValueError("ambiguous full-vocabulary readout")
            embedding = embeddings[0].weight
            with torch.inference_mode():
                record = records[layers[-1]]
                delta = record["readout_delta"].to(embedding.device)
                residual = record.get("readout_residual")
                normalized = (
                    norms[0](delta)
                    if residual is None
                    else norms[0](delta, residual.to(embedding.device))[0]
                )
                logits = torch.nn.functional.linear(
                    normalized, embedding[: self.model_config.hf_text_config.vocab_size]
                ).float()[0]
                answer = logits[[15, 16]]
                extra["p01"] = float((answer.logsumexp(0) - logits.logsumexp(0)).exp())
                extra["readout_logits"] = answer.cpu().tolist()
                extra["readout_method"] = "exact_fused_final_norm_tied_full_embedding"
        return {
            **extra,
            "activations": activations,
            "activation_layers": layers,
            "activation_positions": positions,
            "capture_chunks": {str(i): records[i]["chunks"] for i in layers},
        }

    def lens_clear(self, key: str) -> None:
        self._lens_buffers.pop(key, None)
        self._lens_vectors.pop(key, None)
        getattr(self, "_lens_edits", {}).pop(key, None)
