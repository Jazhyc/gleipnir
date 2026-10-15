"""Adopted default worker with lean dispatch for gleipnir custom ops."""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_gdn_fast.worker import FastGdnWorker


class LeanDispatchMixin:
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from gleipnir.serving import lean_dispatch

        super().load_model(load_dummy_weights=load_dummy_weights)
        ops = lean_dispatch.install()
        if not ops:
            raise ValueError("no gleipnir custom ops found for lean dispatch")
        write("lean_dispatch.json", {"worker_pid": os.getpid(), "ops": ops})

    def set_lean_dispatch(self, enabled: bool) -> dict:
        from gleipnir.serving import lean_dispatch

        return lean_dispatch.set_enabled(enabled)

    def lean_dispatch_audit(self) -> dict:
        from gleipnir.serving import lean_dispatch

        return lean_dispatch.audit()


class LeanDispatchWorker(LeanDispatchMixin, FastGdnWorker):
    """Default stack plus lean custom-op dispatch."""
