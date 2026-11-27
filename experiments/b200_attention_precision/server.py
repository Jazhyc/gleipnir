"""Register precision variants before the unchanged vLLM 0.31 API startup."""

import experiments.b200_vllm031.server  # noqa: F401
import gleipnir.serving.vllm.attention_precision  # noqa: F401
from gleipnir.repository import repo_root

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(repo_root(__file__))
    install_native_encoder()
    install_score_api()
    main()
