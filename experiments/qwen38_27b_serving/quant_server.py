"""BF16 monitor server entrypoint with the geometry FP8/FP4 config registered."""

from pathlib import Path

from experiments.b200_vllm031.backend_config import install

install()

if __name__ == "__main__":
    from vllm.entrypoints.launchers.api_server.entry import main

    import gleipnir.serving.quant_stack  # noqa: F401  (registers the config)
    from gleipnir.serving.compile_cache import install_compile_identity
    from gleipnir.serving.gigatoken import install_native_encoder
    from gleipnir.serving.monitor_score import install_score_api

    install_compile_identity(Path(__file__).resolve().parents[2])
    install_native_encoder()
    install_score_api()
    main()
