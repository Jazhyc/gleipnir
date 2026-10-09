# Restore Lens on the SDPA-trained monitor

Hypothesis: the existing request-scoped Lens bridge works on the completed
BF16/SDPA augmented replay with unquantized BF16 serving. Restore the pinned
vLLM 0.31/Lens 1.3.0 runtime, native overlays and checksum-identical merged
checkpoint from the retained NC2 volume. No retraining or dependency upgrades.

Use the archived SDPA BF16 command, changing only its server entrypoint and
adding eager execution and the Lens worker extension. Keep cached causal LAST
0/1 scoring, native tokenizer, corrected synchronous pooling scheduler, prefix
off, chunked prefill, 32K context/token budget and 128 sequences. BF16 eager is
an explicit research baseline for mechanistic analysis; optimized production
selection remains unchanged. Archive stale active-process receipts from the
terminated pod without signaling their PIDs on the new container.

Before startup, freeze input/source/model/runtime identities. Reuse the passing
merge/master receipt for identical restored weights, then run the same 20-row
adapter canary against the archived FP32 master (MAE <=0.020, correlation >=0.99,
finite scores/nonzero adapter effect), compare separately to compiled BF16,
and require actual native BF16 audits. Preserve eager numerical drift. Require
existing Lens smoke checks for no-op/zero/nonzero steering, request and absolute
position isolation, chunked capture, cleanup and the public client. Stop on
identity drift, failed gates/audits, nonfinite outputs, OOM or functional failure;
retire failed workers. Keep a passing scorer warm.

No full ID/APPS evaluation, learned-direction selection, promotion, systems
benchmark, new capacity or determinism study. Frozen ID/APPS controls remain
available for the subsequent intervention study. Use the user-reserved NC2
B200; merged weights stay on its ephemeral disk, with the persistent backup
retained. Monitoring is active-turn only.

After restoration, run the existing runtime wrapper with
`-m experiments.b200_sdpa_lens.start --name sdpa01`. Configuration is frozen in
`config.json`; artifacts use `results/b200_sdpa_lens/sdpa01/`. The HTTP endpoint
and public client are documented in `experiments/b200_vllm_lens/README.md`.
