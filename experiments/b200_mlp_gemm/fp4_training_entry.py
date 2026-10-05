"""Scoped FP4 MLP arithmetic in the existing ordinary FA4 Trainer."""

import json
import os
import runpy
from pathlib import Path
from unittest.mock import patch

from gleipnir.bf16_lora import bf16_lora_metadata
from gleipnir.cudnn_fp4_mlp import cache_metadata, install_fp4_mlp


def main() -> None:
    def validate_and_install(model):
        original = bf16_lora_metadata(model)
        if not hasattr(model, "_gleipnir_fp4_mlp_installation"):
            installation = install_fp4_mlp(
                model,
                hardware_packing=os.environ.get("GLEIPNIR_FP4_HARDWARE_PACKING", "0")
                == "1",
                fused_descale=os.environ.get("GLEIPNIR_FP4_FUSED_DESCALE", "0") == "1",
            )
            if len(installation["modules"]) != 32:
                raise ValueError("native FP4 screen requires exactly 32 Qwen MLPs")
            model._gleipnir_fp4_mlp_installation = installation
            print(f"native_fp4_mlp_installation={installation}", flush=True)
        return {**original, "native_fp4_mlp": model._gleipnir_fp4_mlp_installation}

    try:
        with patch("gleipnir.bf16_lora.bf16_lora_metadata", validate_and_install):
            runpy.run_path(
                "experiments/deception_distillation/train_student_sft.py",
                run_name="__main__",
            )
    finally:
        path = os.environ.get("GLEIPNIR_FP4_RUNTIME_REPORT")
        if path:
            Path(path).write_text(json.dumps(cache_metadata(), indent=2) + "\n")


if __name__ == "__main__":
    main()
