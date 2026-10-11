"""Audit the unchanged BF16 backbone, then replace only head computation."""

from __future__ import annotations

import hashlib
import types

import torch

from gleipnir.serving.bf16_worker import Bf16Worker

from .head import project


class Fp32HeadWorker(Bf16Worker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model()
        heads = {id(m.score): m.score for m in model.modules() if hasattr(m, "score")}
        head = next(iter(heads.values()))
        head.register_buffer(
            "fp32_decision_rows", head.weight.detach().float(), persistent=False
        )
        if not torch.equal(head.fp32_decision_rows.to(torch.bfloat16), head.weight):
            raise ValueError("promoted head weights changed")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.precision["head_computation"] = {
            "passed": False,
            "stored_weight_dtype": str(head.weight.dtype),
            "compute_weight_dtype": str(head.fp32_decision_rows.dtype),
            "promoted_weights_bitwise_identical": True,
            "tf32_allowed": False,
            "compute_weight_sha256": hashlib.sha256(
                head.fp32_decision_rows.cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        audit = self

        def forward(module, hidden):
            output = project(hidden, module.fp32_decision_rows)
            if torch.backends.cuda.matmul.allow_tf32:
                raise ValueError("TF32 enabled during FP32 head computation")
            if output.dtype != torch.float32:
                raise ValueError("head output rounded before pooling")
            if not audit.precision["head_computation"]["passed"]:
                audit.precision["head_computation"].update(
                    passed=True,
                    input_dtype=str(hidden.dtype),
                    output_dtype=str(output.dtype),
                    input_shape=list(hidden.shape),
                    output_shape=list(output.shape),
                )
                audit.save()
                print("fp32_monitor_head_actual_dispatch_passed", flush=True)
            return output

        head.forward = types.MethodType(forward, head)
        self.save()
