"""Ensure the custom quantizer is registered in vLLM's spawned GPU process."""

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.v1.worker.gpu_worker import Worker

from gleipnir.vllm_fp32_logits import install_fp32_logits
from gleipnir.vllm_nvfp4 import GleipnirNvFp4Config, audit_layer_fallback

logger = init_logger("vllm.gleipnir.precision")


class Fp4Worker(Worker):
    """Use the standard GPU worker with explicit quantizer registration."""

    def __init__(self, *args, **kwargs):
        assert get_quantization_config("gleipnir_nvfp4") is GleipnirNvFp4Config
        super().__init__(*args, **kwargs)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        if getattr(self.vllm_config.quant_config, "fp32_logits", "0") == "1":
            install_fp32_logits(self.model_runner.get_model())
        config = self.vllm_config.quant_config
        if getattr(config, "fp8_layers", set()):
            if config.projections != "down":
                raise ValueError("Loaded fallback audit requires down-only FP4")
            actual = audit_layer_fallback(self.model_runner.get_model(), config)
            logger.info("Loaded down projection precision audit: %s", actual)


class Fp4ProfileWorker(Fp4Worker):
    """Initialize the diagnostic CUPTI subscriber with custom registration intact."""

    def init_device(self) -> None:
        super().init_device()
        from experiments.local_inference.profile_worker import probe_cupti

        probe_cupti(self.device)
