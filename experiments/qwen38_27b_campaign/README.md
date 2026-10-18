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

**Training (17:53–22:28 UTC).** 245 updates in 15,200 s of updates (train
stage 16,301 s with startup gates), about 62 s per update (~4,990
tokens/s). Mean training loss 0.149: the first 20 updates averaged 0.179 and
the last 50 averaged 0.129, at Kimi's mean target entropy on these rows
(0.128). Completion validation passed.

**Merge and parity.** Master reference 223 s, merge 379 s, merged parity
172 s (correlation 0.99991, MAE 0.0016, maximum 0.030, adapter effect 0.73).
The driver paused after merged parity as planned.

**Scoring-source re-binding.** A prepared campaign pins its sources for its
whole life (`_prepare` only re-checks an existing manifest), so the resume
first stopped on source drift in `evaluation.py`. Nothing ran.
`rebind_evaluation_sources.py` verified every completed stage's bound outputs,
allowed only the three scoring-stage files of `1b66597`
(`evaluation.py`, `bf16_monitor.py`, `bf16_worker.py`) to change, archived the
old manifest and executed sources (`*.pre_rebind_20261017T224617Z*`) and
recorded the change under `source_rebindings` in the new manifest. The
`evaluation.py` change touches only the BF16 attention-call count in the
scoring path, not merged parity.

**ID scoring (1,735 s stage; 1,456 s for 3,012 rows, 33.75M tokens).**
Serving parity against the FP32 master: correlation 0.99999, MAE 0.0007,
maximum 0.010. The BF16 audit passed with the 27B geometry (16 attention
layers, 128/96/32 MLP/GDN/attention projections, 16 attention calls). All
3,012 rows scored. The scorer was retired afterwards.

## Result (`compare.py`, `results/qwen38-27b-s0-01/id_comparison.json`)

| ID, 3,012 rows | macro pAUROC@20 | macro AUROC | Brier | Gloom exfiltration | Test STRIDE |
|---|---:|---:|---:|---:|---:|
| **27B trained, seed 0** | **0.9217** | **0.9822** | 0.0759 | **0.8868** | 0.9567 |
| 27B unadapted | 0.8869 | 0.9730 | 0.0812 | 0.8454 | 0.9284 |
| 4B seed 0 | 0.8959 | 0.9638 | 0.1083 | 0.8285 | 0.9633 |
| 4B seed 1 | 0.8978 | 0.9627 | 0.0862 | 0.8168 | 0.9788 |
| 4B seed 2 | 0.9142 | 0.9734 | 0.0687 | 0.8619 | 0.9665 |

- **The hypothesis holds on ID:** +3.5 points over the unadapted 27B and +2.6
  over the seed-0 4B. Against the best 4B seed (+0.75) the gap is inside the
  4B's seed-to-seed spread (about 1 point), and the 27B has one seed.
- **Where it gains.** Gloom (0.887, the best of any model) and macro AUROC
  (0.982). On Test STRIDE it improves on zero-shot (0.928 → 0.957) but stays
  below all three 4B seeds.
- **Not yet known.** OOD, the frozen test suite where the unadapted 27B
  scores 0.890; fine-tuning toward Kimi could trade some of that away.

**Training (17:53–22:28 UTC).** 245 updates in 15,200 s (4 h 13 min; 57–73 s
per update after warm-up), mean training loss 0.149. The first 20 updates
averaged 0.179 and the last 50 0.129, at Kimi's mean target entropy over these
rows (0.128), the floor of the soft-BCE objective. Completion validation
passed (step count, recipe metadata, trim, initial and final master digests).

**Merge and parity.** Master reference, merge (55 GB BF16) and merged parity
completed by 22:42; merged vs. FP32 master: correlation 0.99991, MAE 0.0016,
max 0.030, adapter effect 0.73.

**Scoring-source re-binding.** The first resume stopped on campaign source
drift: a prepared campaign pins its sources, and the BF16 serving fixes
(`1b66597`) changed `evaluation.py`, `bf16_monitor.py` and `bf16_worker.py`
after preparation. `rebind_evaluation_sources.py` verified every completed
stage's bound outputs, allowed only those three scoring-stage files, archived
the manifest (`manifest.pre_rebind_20261017T224617Z.json`) and recorded the
old/new hashes under `source_rebindings`. No completed stage ran the changed
code.

**ID scoring (BF16 vLLM 0.31, 1,456 s).** Serving vs. FP32 master
correlation 0.99999, MAE 0.0007; the BF16 audit passed with the 27B geometry
(16 attention layers, 32/96/128 attention/GDN/MLP projections, 16 attention
calls). All 3,012 rows scored (`compare.py`;
`results/qwen38-27b-s0-01/id_comparison.json`):

| ID, 3,012 rows | macro pAUROC@20 | macro AUROC | Brier | Gloom | Test STRIDE |
|---|---:|---:|---:|---:|---:|
| **27B trained, seed 0** | **0.9217** | **0.9822** | 0.0759 | **0.8868** | 0.9567 |
| 27B unadapted | 0.8869 | 0.9730 | 0.0812 | 0.8454 | 0.9284 |
| 4B seed 0 | 0.8959 | 0.9638 | 0.1083 | 0.8285 | 0.9633 |
| 4B seed 1 | 0.8978 | 0.9627 | 0.0862 | 0.8168 | 0.9788 |
| 4B seed 2 | 0.9142 | 0.9734 | 0.0687 | 0.8619 | 0.9665 |

**Reading.** Training adds 3.5 points over the unadapted 27B and 2.6 over the
same-seed 4B; against the best 4B seed (+0.75) the gap is within 4B seed
variation (about 1 point). The gain is mostly Gloom (best of any model) and
AUROC; on Test STRIDE the 27B improves on zero-shot but stays below all three
4B seeds. One 27B seed; OOD is not yet scored.

## OOD (`ood.py`, `compare_ood.py`; 2026-10-18 00:16–01:02 UTC)

An OOD-only scoring context (`qwen38-27b-s0-ood01`) reused the 4B
seed-variation scorer: it re-merged the FP32 master into a fresh slot
(byte-identical to the campaign merge, so its parity receipt was reused),
passed BF16 serving parity (correlation 0.99999, MAE 0.0007) and the BF16
audit, scored all 6,395 frozen OOD rows and retired the scorer. All 6,395
stored workload token counts match the 27B tokenizer (`ood_tokenizer_audit.py`).

| strict OOD, pAUROC@20 | 27B trained | 27B unadapted | Kimi K3 | 4B seeds 0 / 1 / 2 |
|---|---:|---:|---:|---|
| **Mean-OOD** | **0.9087** | 0.8901 | 0.9084 | 0.7869 / 0.8125 / 0.7768 |
| AgentDojo | 0.9983 | 0.9657 | 0.9997 | 0.9298 / 0.9248 / 0.8506 |
| CUA-SHADE-Arena | 0.8806 | 0.8648 | 0.8856 | 0.7717 / 0.7640 / 0.7679 |
| IaC | 0.9871 | 0.9678 | 0.9920 | 0.8123 / 0.8958 / 0.8451 |
| MLE-Sabotage | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 / 0.9995 |
| Rogue Eval | 0.9989 | 0.9862 | 0.9986 | 0.9696 / 0.9812 / 0.9765 |
| SHADE-Arena | 0.5875 | 0.5560 | 0.5745 | 0.2380 / 0.3092 / 0.2209 |
| macro AUROC | 0.9679 | 0.9535 | 0.9616 | 0.9222 / 0.9255 / 0.9061 |

The unadapted 27B and Kimi K3 were scored under the full teacher prompt; the
trained 27B and 4B students under the compact student prompt. Kimi's
SHADE-Arena value is implied by its recorded Mean-OOD and other sources.

- Training raises OOD as well as ID (+1.9 points over zero-shot; every
  unsaturated source improves), so fitting Kimi did not erode the 27B's
  generalization.
- The trained 27B ties its teacher on Mean-OOD (+0.0003), leads it on
  SHADE-Arena (0.588 vs. 0.575) and Rogue Eval, and is within 0.5 points on
  the rest; one seed, against a 4B OOD seed SD of 1.8 points.
- It is 9.6–13.2 points above the 4B students, most of all on SHADE-Arena.
