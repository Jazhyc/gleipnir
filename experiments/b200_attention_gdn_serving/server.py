"""Register mixed FROST/GDN precision before starting the standard vLLM API."""

import runpy
from pathlib import Path

from gleipnir.vllm_frost_attention_fp4 import FrostAttentionFp4Config  # noqa: F401
from gleipnir.vllm_frost_gdn import FrostGdnConfig  # noqa: F401
from gleipnir.vllm_frost_gdn_fp4 import FrostGdnFp4Config  # noqa: F401

if __name__ == "__main__":
    from gleipnir.serving_compile_cache import install_compile_identity

    install_compile_identity(Path(__file__).resolve().parents[2])
    print("mixed_gdn_registry_import_passed fp8=true fp4=true", flush=True)
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
