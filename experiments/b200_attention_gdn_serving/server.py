"""Register mixed FROST/GDN precision before starting the standard vLLM API."""

import runpy

from gleipnir.vllm_frost_gdn import FrostGdnConfig  # noqa: F401

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
