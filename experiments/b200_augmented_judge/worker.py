"""Explicit A/B head binding around unchanged audited backbone recipes."""

import experiments.b200_monitor_score.worker as monitor_audit
from experiments.b200_attention_precision.worker import AttentionPrecisionWorker
from gleipnir.serving.bf16_worker import Bf16Worker

# The old auditor captures these constants at import. Bind only this spawned
# process to A/B; do not edit its checksum-bound source or monitoring defaults.
monitor_audit.DECISION_TOKENS = ["A", "B"]
monitor_audit.DECISION_IDS = [32, 33]


def validate_surface(config) -> None:
    """Reject a monitoring head or any different token ordering."""
    from vllm.tokenizers import get_tokenizer

    tokens = ["A", "B"]
    tokenizer = get_tokenizer(config.model_config.tokenizer)
    if (
        [tokenizer.convert_tokens_to_ids(t) for t in tokens] != [32, 33]
        or config.model_config.hf_text_config.classifier_from_token != tokens
        or config.additional_config["monitor_score"]["token_ids"] != [32, 33]
    ):
        raise ValueError("JudgeDeceiver requires the exact A/B decision surface")


class JudgeOptimizedWorker(AttentionPrecisionWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        validate_surface(self.vllm_config)
        super().load_model(load_dummy_weights=load_dummy_weights)


class JudgeBf16Worker(Bf16Worker):
    decision_ids = (32, 33)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        validate_surface(self.vllm_config)
        super().load_model(load_dummy_weights=load_dummy_weights)
