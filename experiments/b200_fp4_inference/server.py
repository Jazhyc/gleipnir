"""Register online precision before the standard HTTP server parses arguments."""

import runpy

from gleipnir.vllm_online_nvfp4 import OnlineNvFp4Config  # noqa: F401

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
