"""Register mixed FROST/GDN precision before starting the standard vLLM API."""

import runpy

from gleipnir.vllm_frost_attention_fp4 import FrostAttentionFp4Config  # noqa: F401
from gleipnir.vllm_frost_gdn import FrostGdnConfig  # noqa: F401
from gleipnir.vllm_frost_gdn_fp4 import FrostGdnFp4Config  # noqa: F401

if __name__ == "__main__":
    print("mixed_gdn_registry_import_passed fp8=true fp4=true", flush=True)
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
