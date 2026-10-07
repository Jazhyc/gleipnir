"""Native text encoding around the unchanged selected GPU registry loader."""

import runpy
from pathlib import Path

# Import at module scope so spawned engines inherit registry initialization.
import experiments.b200_attention_gdn_serving.server  # noqa: F401

if __name__ == "__main__":
    from gleipnir.serving_compile_cache import install_compile_identity
    from gleipnir.serving_gigatoken import install_native_encoder

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
