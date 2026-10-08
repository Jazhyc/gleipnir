"""Start the migrated score API without the obsolete Torch 2.11 repair."""

from pathlib import Path

import experiments.b200_monitor_score.server  # noqa: F401
from experiments.b200_vllm031.backend_config import install

install()

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    main()
