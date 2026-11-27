"""Register the custom mixed FP8 config in the spawned GPU worker."""

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_mixed_fp8 import GleipnirMixedFp8Config


class MixedFp8Worker(Worker):
    def __init__(self, *args, **kwargs):
        assert get_quantization_config("gleipnir_mixed_fp8") is GleipnirMixedFp8Config
        super().__init__(*args, **kwargs)
