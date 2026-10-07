"""Audit the two-row head without changing the selected backbone kernels."""

import hashlib
import json
import os

import torch
from safetensors import safe_open
from vllm.tokenizers import get_tokenizer

from experiments.b200_attention_gdn_serving.swiglu_native_output_worker import (
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker,
)
from experiments.b200_attention_gdn_serving.worker import ROOT, write
from gleipnir.serving.monitor_score import DECISION_IDS, DECISION_TOKENS


class MonitorScoreAuditWorker(
    NativeOutputAttentionTunedPreparationMxfp8ServingAuditWorker
):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        config = self.vllm_config.model_config
        scheduler = self.vllm_config.scheduler_config
        model = self.model_runner.get_model()
        if (
            config.runner_type != "pooling"
            or config.attn_type not in {"decoder", "hybrid"}
            or config.pooler_config.get_seq_pooling_type() != "LAST"
            or not scheduler.enable_chunked_prefill
            or self.vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError("monitor score requires causal cached LAST pooling")
        tokenizer = get_tokenizer(config.tokenizer)
        if [
            tokenizer.convert_tokens_to_ids(t) for t in DECISION_TOKENS
        ] != DECISION_IDS:
            raise ValueError("monitor decision token identity changed")
        heads = [m.score for m in model.modules() if hasattr(m, "score")]
        # The pooler can reference the same classifier; deduplicate references.
        heads = list({id(head): head for head in heads}.values())
        if len(heads) != 1:
            raise ValueError("expected exactly one monitoring head")
        head = heads[0]
        if (
            head.weight.shape != (2, config.get_hidden_size())
            or head.weight.dtype != torch.bfloat16
            or head.bias is not None
            or any(hasattr(m, "lm_head") for m in model.modules())
        ):
            raise ValueError("monitor requires only a two-row BF16 output head")
        merged = ROOT / config.model
        index = json.loads((merged / "model.safetensors.index.json").read_text())
        mapping = index["weight_map"]
        keys = [k for k in mapping if k.endswith("lm_head.weight")]
        tied = config.hf_text_config.tie_word_embeddings
        if not keys and tied:
            keys = [k for k in mapping if k.endswith("embed_tokens.weight")]
        if len(keys) != 1:
            raise ValueError("cannot identify merged classifier source weights")
        with safe_open(merged / mapping[keys[0]], framework="pt", device="cpu") as f:
            expected = f.get_slice(keys[0])[DECISION_IDS[0] : DECISION_IDS[1] + 1]
        observed = head.weight.detach().cpu()
        if not torch.equal(observed, expected):
            raise ValueError("classifier weights differ from merged decision rows")
        receipt = {
            "passed": True,
            "worker_pid": os.getpid(),
            "runner": config.runner_type,
            "attention_type": config.attn_type,
            "pooling": "LAST",
            "chunked_prefill": True,
            "prefix_caching": False,
            "decision_token_ids": DECISION_IDS,
            "head_shape": list(observed.shape),
            "head_dtype": str(observed.dtype),
            "weight_source": keys[0],
            "weights_sha256": hashlib.sha256(
                observed.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "full_vocabulary_head_present": False,
            "sampler_bypassed": True,
            "backbone_recipe_unchanged": True,
        }
        self.precision["monitor_score"] = receipt
        self.audit_serving_state()
        write("monitor_score.json", receipt)
        print("monitor_score_audit_passed", receipt, flush=True)
