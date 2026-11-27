"""Explicit optimized-serving candidate using validated typed KV coordinates."""

from experiments.teachers.filtered_fp4_baseline.compiler_repair import install_repair
from gleipnir.serving.stack.workers.attention_precision import AttentionPrecisionWorker


class TypedCoordinateWorker(AttentionPrecisionWorker):
    def _install_attention_audit(self) -> None:
        if self.condition.get("mxfp8_coordinate_repair") != "typed_int32":
            raise ValueError("typed-coordinate worker requires explicit recipe binding")
        install_repair()
        super()._install_attention_audit()
