# First Qwen3.8-27B monitor campaign

**Hypothesis.** Distilling the Kimi K3 soft targets into Qwen3.8-27B with the
4B recipe raises held-out ID macro pAUROC@20 above both the unadapted 27B
(0.8869) and the seed-0 4B student (0.8959)
([27B zero-shot ID](../qwen27b_generations_ood/README.md#qwen38-27b-on-the-id-and-training-rows-2026-10-17)).

**Design (frozen before training).** `prepare.py` copies the strict-filtered
BF16/SDPA campaign `teacher-seed-sdpa-s1-01` (7,836 rows, Kimi K3 targets, LR
5e-5, logical batch 32, one epoch, 245 updates, ID evaluation, BF16 serving)
and changes only:
- seed 0, matching the 27B initial adapter;
- model `Qwen/Qwen3.8-27B` @ `1d4bf0f2…` with `qwen38_27b_b200_default`
  ([decision](../../docs/decisions/qwen38_27b_b200_training_default.md));
- the initial adapter `results/qwen38_27b_initial_adapter`, exported by
  `experiments/qwen38_27b_training_speed/export_initial_adapter.py` and
  digest-matched to the benchmark's fresh seed-0 LoRA (`4093b2aa…`);
- `full_startup_validation: true` (no 27B startup receipt exists) and a
  `base_files` manifest of the 27B snapshot (documentation files excluded).

Selection is the final checkpoint only, with no promotion; OOD runs outside
the campaign. The comparison is descriptive: 27B zero-shot and the three 4B
seeds (0.8959 / 0.8978 / 0.9142).

**Token identity.** `tokenizer_audit.py` (results
`results/qwen38_27b_training_speed/tokenizer_audit.json`): the 27B tokenizer
gives identical ids and counts for all 3,012 frozen ID prompts, and identical
non-thinking renderings and ids for all 7,836 training rows (the trainer
renders with `enable_thinking=False`; only thinking-mode templates differ).

**Execution.** The driver runs through `merged-reference`, then pauses so
the BF16 serving fixes (`1b66597`) can be synced without source drift during
a running driver; `run.py --resume` then runs ID scoring.

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.qwen38_27b_campaign.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_campaign.run --through merged-reference
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_campaign.run --resume
```

## Run log

**Launch (2026-10-17 17:53 UTC).** Pod `ljzun21fdpft92`, `GPU-7a2d0c53…`.
All startup gates passed: FlashQLA model canary finite (gradient relative L2
0.329, as in the benchmark), packing eager and compiled canaries, trim canary,
and the longest-32 memory preflight (154 GiB of 178). The first update ran by
about 18:13; the first eight averaged about 70 s including warm-up (progress
bar ETA 4 h 38 min).
