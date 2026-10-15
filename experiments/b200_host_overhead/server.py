"""The selected default's server, with py-spy attach permitted (diagnostic only)."""

from pathlib import Path

import experiments.b200_attention_precision.server  # noqa: F401
from experiments.b200_host_overhead.tracing import allow_profiler_attach

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    allow_profiler_attach()
    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    main()
