"""Full long-context stack with prefix caching allowed for the replay only.

The production monitor audit rejects prefix caching because cached
classification has not been validated; this experiment measures exactly that.
The flag is masked only while the inherited audit runs during model loading,
then restored before KV-cache initialization, and the override is recorded.
"""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_gdn_fast.worker import FastGatedTrimWorker


class PrefixReplayWorker(FastGatedTrimWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        cache = self.vllm_config.cache_config
        if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
            raise ValueError("prefix replay requires align-mode prefix caching")
        cache.enable_prefix_caching = False
        try:
            super().load_model(load_dummy_weights=load_dummy_weights)
        finally:
            cache.enable_prefix_caching = True
        write(
            "prefix_replay_worker.json",
            {
                "worker_pid": os.getpid(),
                "enable_prefix_caching": cache.enable_prefix_caching,
                "mamba_cache_mode": cache.mamba_cache_mode,
                "audit_override": "prefix caching masked during inherited audit only",
            },
        )
