"""Register precision variants before the unchanged vLLM 0.31 API startup."""

from pathlib import Path

import experiments.b200_vllm031.server  # noqa: F401
import gleipnir.serving.vllm.attention_precision  # noqa: F401

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    main()
