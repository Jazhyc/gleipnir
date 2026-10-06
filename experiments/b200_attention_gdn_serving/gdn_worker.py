"""Select a canary-validated GDN prefill backend without changing decode."""

import hashlib
import json
from pathlib import Path

from experiments.b200_attention_gdn_serving.worker import ROOT, ServingAuditWorker


class GdnServingAuditWorker(ServingAuditWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        condition = self.vllm_config.additional_config["serving_condition"]
        selected = condition["gdn_backend"]
        path = ROOT / condition["gdn_validation"]
        validation = json.loads(path.read_text())
        if (
            not validation["passed"]
            or validation["backend"] != selected
            or validation["output_relative_l2_limit"] != 0.03
            or validation["state_relative_l2_limit"] != 0.03
            or validation["device_capability"] != [10, 0]
            or len(validation["checks"]) != 6
        ):
            raise ValueError("missing matched GDN forward/state validation")
        for source, expected in validation["sources"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"GDN validation source drift: {source}")
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

        runtime = None
        if selected == "flashqla":
            from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

            from gleipnir.flashqla_training import load_flashqla
            from gleipnir.serving_gdn_kernels import make_flashqla_prefill

            _, runtime = load_flashqla()
            gdn.fi_chunk_gated_delta_rule = make_flashqla_prefill(
                chunk_gated_delta_rule_fwd, gdn.l2norm_fwd, auto_cp=False
            )
        elif selected == "cutedsl":
            from cutlass._mlir.dialects import nvvm

            from gleipnir.serving_gdn_kernels import install_nvvm_compatibility

            runtime = {"compatibility": install_nvvm_compatibility(nvvm)}
        else:
            raise ValueError("unsupported alternative GDN backend")
        super().load_model(load_dummy_weights=load_dummy_weights)
        operators = {}
        for name, layer in self.model_runner.get_model().named_modules():
            if isinstance(layer, gdn.ChunkGatedDeltaRule):
                expected = "flashinfer" if selected == "flashqla" else "cutedsl"
                if layer.gdn_prefill_backend != expected:
                    raise ValueError(f"GDN backend fell back: {name}")
                operators[name] = {
                    "routing_backend": layer.gdn_prefill_backend,
                    "forward_method": layer._forward_method.__name__,
                    "effective_kernel": selected,
                }
        if len(operators) != 24:
            raise ValueError("incomplete alternative GDN backend coverage")
        self.precision["gdn_prefill"] = {
            "backend": selected,
            "operators": operators,
            "runtime": runtime,
            "forward_only": True,
            "auto_cp": False if selected == "flashqla" else None,
            "validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self.audit_serving_state()
        print(f"gdn_prefill_audit_passed backend={selected} layers=24", flush=True)
