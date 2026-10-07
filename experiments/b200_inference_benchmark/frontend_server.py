"""Native text encoding around the unchanged selected GPU registry loader."""

import os
import runpy
from pathlib import Path

# Import at module scope so spawned engines inherit registry initialization.
import experiments.b200_attention_gdn_serving.server  # noqa: F401

if validation := os.environ.get("GLEIPNIR_FROST_WRAPPER_VALIDATION"):
    from gleipnir.serving_frost_wrappers import enable_worker_control

    enable_worker_control(Path(__file__).resolve().parents[2], validation)

if os.environ.get("GLEIPNIR_PROMPT_ONLY_TRACE"):
    from experiments.b200_attention_gdn_serving.prompt_only_trace import enable_trace

    enable_trace(Path(__file__).resolve().parents[2])

if __name__ == "__main__":
    import json
    import sys

    from gleipnir.serving_compile_cache import install_compile_identity
    from gleipnir.serving_gigatoken import install_native_encoder

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    if "--additional-config" in sys.argv:
        additional = json.loads(sys.argv[sys.argv.index("--additional-config") + 1])
        if additional.get("serving_condition", {}).get("prompt_only"):
            from gleipnir.serving_prompt_only import install_api_guard

            install_api_guard()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
