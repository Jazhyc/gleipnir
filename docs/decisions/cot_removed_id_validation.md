# Use CoT-removed ID inputs for future validation

Decision: accepted by the user on 2026-09-08.

Future monitoring validation, hyperparameter comparisons, early stopping, and
checkpoint selection must use the CoT-removed ID dataset. Remove source-typed
agent reasoning and explicit `think` calls while preserving visible assistant
prose, other tool calls/arguments, and tool results. This aligns selection with
monitoring observable behavior without access to identified private reasoning.

The canonical current artifact is `data/id_cot_only_evaluation/prompts.jsonl`:

- 3,012 rows: 946 STRIDE and 2,066 Gloom-Exfiltration.
- Input SHA-256: `7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4`.
- Manifest: `data/id_cot_only_evaluation/evaluation.manifest.json`.
- Manifest SHA-256: `abcc8267a77f6eff0a252ea846a6a59283e8a027cc860210e3984064eec611f1`.

New campaign configs must reference this input and manifest with their hashes,
and use separate result/cache identities. Preserve the original input and frozen
historical configurations for reproducibility; their scores are not the default
selection baseline. The broader assistant-prose-removal dataset is an ablation,
not the validation default. Use source-macro pAUROC@20 and retain per-source
ranking, calibration, threshold, and tie diagnostics.

The standard 4B checkpoint's matched CoT-removed baseline is macro pAUROC@20
0.850083 and AUROC 0.952674. Compare future candidates on the same input contract.
The existing CoT-only experiment config describes a completed fixed-checkpoint
comparison; new selection campaigns must declare their own selection rule.

This decision does not clean existing training data or regenerate teacher targets.
It prevents validation from rewarding access to identified agent CoT, but does
not establish that a model trained on reasoning-bearing inputs never relies on
reasoning. Keep the six-source OOD evaluation outside selection.

Evidence and reproduction:
[CoT-only experiment](../../experiments/id_cot_only_evaluation/README.md),
[ID audit](../findings/id_assistant_prose_and_reasoning.md), and
[OOD audit](../findings/ood_input_reasoning.md).
