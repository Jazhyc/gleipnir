"""Adopted serving workers with opt-in prefix caching.

The production monitor audit rejects prefix caching because cached
classification is not score-identical to uncached scoring (the replay measured
|cached - uncached| up to 0.15). These workers allow it only when the server
was launched with align-mode prefix caching: the flag is masked while the
inherited audit runs during model loading, restored before KV-cache
initialization, and the override is recorded.
"""

import os

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_gdn_fast.worker import FastGatedTrimWorker, FastGdnWorker


class PrefixCachingMixin:
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        cache = self.vllm_config.cache_config
        if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
            raise ValueError("prefix-cached serving requires align-mode caching")
        cache.enable_prefix_caching = False
        try:
            super().load_model(load_dummy_weights=load_dummy_weights)
        finally:
            cache.enable_prefix_caching = True
        write(
            "prefix_replay_worker.json",
            {
                "worker_pid": os.getpid(),
                "worker": type(self).__name__,
                "enable_prefix_caching": cache.enable_prefix_caching,
                "mamba_cache_mode": cache.mamba_cache_mode,
                "audit_override": "prefix caching masked during inherited audit only",
            },
        )


class PrefixReplayWorker(PrefixCachingMixin, FastGatedTrimWorker):
    """Long-context replay worker (full stack plus telemetry)."""


class PrefixCachedFastGdnWorker(PrefixCachingMixin, FastGdnWorker):
    """Opt-in prefix-cached variant of the selected serving default."""
