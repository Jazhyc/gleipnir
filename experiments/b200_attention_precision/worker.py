"""Audit all changed attention projections and reuse the remaining native checks."""

import json
import math
import os

from experiments.b200_attention_gdn_serving.worker import ROOT, write
from experiments.b200_vllm031.worker import MigrationWorker
from gleipnir.serving.fp4.attention import EXPECTED, ROWS, SHAPES, projection_identity
from gleipnir.serving.sources import recorded_source_path
from gleipnir.serving.vllm.attention_precision import (
    AttentionBf16Method,
    AttentionFp8Method,
    install_audit,
)


def validate_native(receipt: dict, precision: str) -> None:
    checks = [c for c in receipt.get("checks", []) if c["precision"] == precision]
    if (
        not receipt.get("passed")
        or receipt.get("state") != "completed"
        or {(c["projection"], c["rows"]) for c in checks}
        != {(p, m) for p in SHAPES for m in ROWS}
        or len(checks) != 16
    ):
        raise ValueError("incomplete attention precision native receipt")
    for c in checks:
        if (
            not all(
                c.get(k)
                for k in [
                    "passed",
                    "finite",
                    "zero_row_exact",
                    "unchanged_rows_exact",
                    "replay_changed",
                ]
            )
            or tuple(c["shape"]) != SHAPES[c["projection"]]
            or any(
                not math.isfinite(c[k]) or c[k] > 0.01
                for k in ("relative_l2", "replay_relative_l2")
            )
        ):
            raise ValueError("failed attention precision native check")


class AttentionPrecisionWorker(MigrationWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        import torch

        from experiments.b200_inference_benchmark.run import sha

        condition = self.vllm_config.additional_config["serving_condition"]
        precision = condition["attention_projection_precision"]
        path = condition["attention_precision_validation"]
        receipt = json.loads((ROOT / path).read_text())
        validate_native(receipt, precision)
        for source, expected in receipt["sources"].items():
            if sha(recorded_source_path(ROOT, source)) != expected:
                raise ValueError(f"attention precision source drift: {source}")
        if receipt["gpu"] != torch.cuda.get_device_name():
            raise ValueError("attention precision native hardware changed")
        super().load_model(load_dummy_weights=load_dummy_weights)
        seen, calls, linears = set(), [], []
        expected_method = {"bf16": AttentionBf16Method, "fp8": AttentionFp8Method}[
            precision
        ]
        for name, layer in self.model_runner.get_model().named_modules():
            identity = projection_identity(name)
            if identity is None:
                continue
            if identity not in EXPECTED or not isinstance(
                layer.quant_method, expected_method
            ):
                raise ValueError("attention precision actual layer coverage changed")
            linears.append(
                {
                    "layer": name,
                    "identity": list(identity),
                    "dtype": str(layer.weight.dtype),
                    "method": type(layer.quant_method).__name__,
                    "weight_relative_l2": getattr(
                        layer, "_attention_fp8_weight_error", 0.0
                    ),
                }
            )
        if len(linears) != 16:
            raise ValueError("attention precision requires exactly 16 projections")
        audit = {
            "passed": False,
            "worker_pid": os.getpid(),
            "precision": precision,
            "validation_path": path,
            "validation_sha256": sha(ROOT / path),
            "linears": linears,
            "calls": calls,
            "outputs": "BF16",
        }

        def observed(name, rows, mode):
            if name in seen:
                return
            if (
                projection_identity(name) not in EXPECTED
                or mode != precision
                or not 0 < rows <= 32768
            ):
                raise ValueError("unexpected actual attention projection dispatch")
            seen.add(name)
            calls.append({"layer": name, "rows": rows, "precision": mode})
            audit["passed"] = len(seen) == 16
            write("native_attention_projections.json", audit)

        install_audit(observed)
        self.precision["attention_projection_precision"] = {
            "precision": precision,
            "projection_count": 16,
            "outputs": "BF16",
            "validation_sha256": audit["validation_sha256"],
        }
        self.audit_serving_state()
        write("native_attention_projections.json", audit)
        print("attention_precision_installed", precision, len(linears), flush=True)
