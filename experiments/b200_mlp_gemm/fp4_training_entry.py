"""Scoped FP4 MLP arithmetic in the existing ordinary FA4 Trainer."""

import json
import os
import runpy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from gleipnir.bf16_lora import bf16_lora_metadata
from gleipnir.cudnn_fp4_mlp import cache_metadata, install_fp4_mlp


def install_graph_step_boundary(model: torch.nn.Module) -> None:
    """Keep all decoder graph segments in one physical forward/backward step."""
    if hasattr(model, "_gleipnir_graph_step_boundary"):
        raise ValueError("model graph step boundary is already installed")

    def begin_step(module, args):
        torch.compiler.cudagraph_mark_step_begin()

    model._gleipnir_graph_step_boundary = model.register_forward_pre_hook(begin_step)


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
            install_graph_step_boundary(model)
            installation["cudagraph_step_boundary"] = "physical_model_forward"
            model._gleipnir_fp4_mlp_installation = installation
            print(f"native_fp4_mlp_installation={installation}", flush=True)
        return {**original, "native_fp4_mlp": model._gleipnir_fp4_mlp_installation}

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch("gleipnir.bf16_lora.bf16_lora_metadata", validate_and_install)
            )
            warm_report = os.environ.get("GLEIPNIR_FP4_WARM_REPORT")
            profile_output = os.environ.get("GLEIPNIR_FP4_PROFILE_OUTPUT")
            resident_root = os.environ.get("GLEIPNIR_FP4_RESIDENT_ROOT")
            if profile_output or resident_root:
                from experiments.b200_mlp_gemm.full_model_profile import (
                    profile_validation_reference,
                )

                stack.enter_context(
                    patch(
                        "gleipnir.validated_startup.validation_reference",
                        profile_validation_reference,
                    )
                )
            if resident_root:
                from transformers import Trainer

                from experiments.b200_mlp_gemm.resident_worker import resident_train

                stack.enter_context(
                    patch.object(
                        Trainer,
                        "train",
                        resident_train(Trainer.train, Path(resident_root)),
                    )
                )

            if warm_report:
                from transformers import Trainer

                from experiments.b200_mlp_gemm.warmed_training import (
                    install_warmed_train,
                )

                stack.enter_context(
                    patch.object(
                        Trainer,
                        "train",
                        install_warmed_train(
                            Trainer.train,
                            Path(os.environ["GLEIPNIR_FP4_WARM_REFERENCE"]),
                            Path(warm_report),
                        ),
                    )
                )
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
