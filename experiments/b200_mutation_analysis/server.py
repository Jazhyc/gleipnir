"""Enable validated stride metadata before the API or spawned worker loads models."""

import os
import runpy
from pathlib import Path

import experiments.b200_monitor_score.server  # noqa: F401
from gleipnir.serving.triton_mutation import install

install(
    Path(__file__).resolve().parents[2],
    validation=os.environ["GLEIPNIR_STRIDE_VALIDATION"],
)

if __name__ == "__main__":
    from gleipnir.serving.monitor_score import install_score_api
    from gleipnir.serving_compile_cache import install_compile_identity
    from gleipnir.serving_gigatoken import install_native_encoder

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
