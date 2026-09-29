"""Ensure the custom quantizer is registered in vLLM's spawned GPU process."""

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_nvfp4 import GleipnirNvFp4Config


class Fp4Worker(Worker):
    """Use the standard GPU worker with explicit quantizer registration."""

    def __init__(self, *args, **kwargs):
        assert get_quantization_config("gleipnir_nvfp4") is GleipnirNvFp4Config
        super().__init__(*args, **kwargs)


class Fp4ProfileWorker(Fp4Worker):
    """Initialize the diagnostic CUPTI subscriber with custom registration intact."""

    def init_device(self) -> None:
        super().init_device()
        from experiments.local_inference.profile_worker import probe_cupti

        probe_cupti(self.device)
