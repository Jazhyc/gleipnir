"""Ensure the custom quantizer is registered in vLLM's spawned GPU process."""

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_nvfp4 import GleipnirNvFp4Config


class Fp4Worker(Worker):
    """Use the standard GPU worker with explicit quantizer registration."""

    def __init__(self, *args, **kwargs):
        assert get_quantization_config("gleipnir_nvfp4") is GleipnirNvFp4Config
        super().__init__(*args, **kwargs)
