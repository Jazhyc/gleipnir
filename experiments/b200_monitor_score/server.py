"""Register selected kernels and a true pooling endpoint before vLLM startup."""

import runpy
from pathlib import Path

# The selected frontend initializes registries and optional host binding control
# at import time, including in the spawned GPU process.
import experiments.b200_inference_benchmark.frontend_server  # noqa: F401

if __name__ == "__main__":
    from gleipnir.serving.monitor_score import install_score_api
    from gleipnir.serving_compile_cache import install_compile_identity
    from gleipnir.serving_gigatoken import install_native_encoder

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
