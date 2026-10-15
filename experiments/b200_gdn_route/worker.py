"""Workers that install the length-gated FlashInfer GDN prefill route."""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_last_layer_trim.worker import (
    LastLayerTrimWorker,
    LastQueryPrecisionWorker,
)


def route_threshold(additional: dict) -> int | None:
    """The selected default binds the threshold; experiment runs use the env."""
    selected = additional.get("gdn_route")
    if selected is not None:
        return int(selected["cp_min_tokens"])
    value = os.environ.get("GLEIPNIR_GDN_CP_MIN_TOKENS")
    return int(value) if value else None


class GdnRouteMixin:
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving import gdn_route

        threshold = route_threshold(self.vllm_config.additional_config)
        if threshold is None:
            raise ValueError("GDN route worker requires a token threshold")
        if os.environ.get("GLEIPNIR_FLASHINFER_GDN_CP") != "auto":
            raise ValueError("GDN route gate composes only with the auto route")
        gdn_route.install(threshold)
        self.gdn_route_threshold = threshold
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.precision["gdn_route"] = {"cp_min_tokens": threshold}
        write(
            "gdn_route.json",
            {"worker_pid": os.getpid(), "cp_min_tokens": threshold},
        )

    def gdn_route_audit(self) -> dict:
        from gleipnir.serving import gdn_route

        return {"cp_min_tokens": self.gdn_route_threshold, **gdn_route.audit()}

    def set_gdn_route(self, min_tokens: int) -> dict:
        from gleipnir.serving import gdn_route

        gdn_route.set_threshold(min_tokens)
        return gdn_route.audit()


class GatedTrimWorker(GdnRouteMixin, LastLayerTrimWorker):
    """Long-context telemetry worker with last-query attention and gated GDN."""


class GatedPrecisionWorker(GdnRouteMixin, LastQueryPrecisionWorker):
    """The selected default with last-query attention and gated GDN."""
