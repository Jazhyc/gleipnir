"""Scoped merged gate/up intervention inside the existing ordinary Trainer."""

import runpy
from unittest.mock import patch

from gleipnir.bf16_lora import bf16_lora_metadata
from gleipnir.mlp_gemm import install_merged_mlp


def main() -> None:
    def validate_and_install(model):
        metadata = bf16_lora_metadata(model)
        if not getattr(model, "_gleipnir_mlp_installed", False):
            # The recorded default compiles decoder shells including MLPs.
            installation = install_merged_mlp(model, compile_mlp=False)
            if len(installation["modules"]) != 32:
                raise ValueError("Qwen3.5-4B requires 32 decoder MLPs")
            model._gleipnir_mlp_installed = installation
            print(f"merged_mlp_installation={installation}", flush=True)
        return {**metadata, "mlp_gemm": model._gleipnir_mlp_installed}

    with patch("gleipnir.bf16_lora.bf16_lora_metadata", validate_and_install):
        runpy.run_path(
            "experiments/deception_distillation/train_student_sft.py",
            run_name="__main__",
        )


if __name__ == "__main__":
    main()
