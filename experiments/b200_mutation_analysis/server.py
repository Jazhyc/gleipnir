"""Enable validated stride metadata before the API or spawned worker loads models."""

import os
import runpy

import experiments.b200_monitor_score.server  # noqa: F401
from gleipnir.repository import repo_root
from gleipnir.serving.triton_mutation import install

install(
    repo_root(__file__),
    validation=os.environ["GLEIPNIR_STRIDE_VALIDATION"],
)

if __name__ == "__main__":
    from gleipnir.serving.monitor_score import install_score_api
    from gleipnir.serving_compile_cache import install_compile_identity
    from gleipnir.serving_gigatoken import install_native_encoder

    install_compile_identity(repo_root(__file__))
    install_native_encoder()
    install_score_api()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
