"""Native text encoding around the unchanged selected GPU registry loader."""

import os
import runpy

# Import at module scope so spawned engines inherit registry initialization.
import gleipnir.serving.stack.registry  # noqa: F401
from gleipnir.repository import repo_root

if validation := os.environ.get("GLEIPNIR_FROST_WRAPPER_VALIDATION"):
    from gleipnir.serving.frost_wrappers import enable_worker_control

    enable_worker_control(
        repo_root(__file__),
        validation,
    )

if os.environ.get("GLEIPNIR_PROMPT_ONLY_TRACE"):
    from gleipnir.serving.stack.prompt_only_trace import enable_trace

    enable_trace(repo_root(__file__))

if __name__ == "__main__":
    import json
    import sys

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder

    install_compile_identity(repo_root(__file__))
    install_native_encoder()
    if "--additional-config" in sys.argv:
        additional = json.loads(sys.argv[sys.argv.index("--additional-config") + 1])
        if additional.get("serving_condition", {}).get("prompt_only"):
            from gleipnir.serving.prompt_only import install_api_guard

            install_api_guard()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
