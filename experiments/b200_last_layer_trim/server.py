"""Precision server with the last-query tail registered as a splitting op."""

import experiments.b200_attention_precision.server  # noqa: F401
from gleipnir.repository import repo_root
from gleipnir.serving.last_query_tail import register

# Before engine arguments become a VllmConfig, so the default splitting ops
# (copied from CompilationConfig._attention_ops) include the tail.
register()

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(repo_root(__file__))
    install_native_encoder()
    install_score_api()
    main()
