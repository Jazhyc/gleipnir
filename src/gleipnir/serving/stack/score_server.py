"""Register selected kernels and a true pooling endpoint before vLLM startup."""

import runpy

# The selected frontend initializes registries and optional host binding control
# at import time, including in the spawned GPU process.
import gleipnir.serving.stack.frontend  # noqa: F401
from gleipnir.repository import repo_root

if __name__ == "__main__":
    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(repo_root(__file__))
    install_native_encoder()
    install_score_api()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
