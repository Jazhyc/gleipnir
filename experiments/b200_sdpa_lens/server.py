"""Opt-in eager Lens routes on the ordinary unquantized BF16 monitor."""

import os
from pathlib import Path

os.environ["VLLM_LENS_DISABLE"] = "1"

import gleipnir.serving.bf16_server  # noqa: E402,F401

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.lens_api import install_lens_api

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_lens_api()
    main()
