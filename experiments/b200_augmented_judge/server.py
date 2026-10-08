"""Start the selected BF16 or optimized recipe with an explicit A/B head."""

import importlib
import os
from pathlib import Path

precision = os.environ["GLEIPNIR_JUDGE_PRECISION"]
modules = {
    "bf16": "gleipnir.serving.bf16_server",
    "optimized": "experiments.b200_attention_precision.server",
}
importlib.import_module(modules[precision])

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    main()
