"""Use the already-pinned D256/paged FA4 kernel behind vLLM's forward API."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch

from experiments.b200_attention_gdn_serving.combined_worker import (
    MixedFa4ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT
from gleipnir.serving_fa4 import make_paged_fa4_forward


class ExternalFa4ServingAuditWorker(MixedFa4ServingAuditWorker):
    def _install_attention_audit(self) -> None:
        import flash_attn.cute.interface as external
        import vllm.v1.attention.backends.flash_attn as backend

        path = Path(external.__file__).resolve()
        if (
            version("flash-attn-4") != "4.0.0b33"
            or not path.is_relative_to(ROOT / ".cache/kernels/fa4")
            or torch.cuda.get_device_capability() != (10, 0)
        ):
            raise ValueError("external FA4 source/version/hardware drift")
        validation = ROOT / self.condition["fa4_validation"]
        receipt = json.loads(validation.read_text())
        if not receipt["passed"] or len(receipt["checks"]) != 6 or any(
            not row["finite"] or row["relative_l2"] > 0.01
            for row in receipt["checks"]
        ):
            raise ValueError("external FA4 page-layout validation failed")
        for source, checksum in receipt["sources"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != checksum:
                raise ValueError("external FA4 validation source drift")
        backend.get_flash_attn_version = lambda **kwargs: 4
        backend.flash_attn_varlen_func = make_paged_fa4_forward(
            external._flash_attn_fwd
        )
        self.external_fa4 = {
            "version": version("flash-attn-4"),
            "interface": str(path),
            "interface_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dedicated_hd256_paged": True,
            "hybrid_page_subdivision": "zero_copy_nhd_128",
            "validation": str(validation),
            "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        }
        super()._install_attention_audit()

    def audit_serving_state(self) -> None:
        from vllm.model_executor.layers.attention import Attention

        for layer in self.model_runner.get_model().modules():
            if isinstance(layer, Attention) and any(
                value != 1.0
                for value in (
                    layer._q_scale_float,
                    layer._k_scale_float,
                    layer._v_scale_float,
                )
            ):
                raise ValueError("external BF16 FA4 requires unit layer descales")
        self.precision["external_fa4"] = self.external_fa4
        super().audit_serving_state()
