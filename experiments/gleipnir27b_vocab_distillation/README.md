# Gleipnir 27B as a full-vocabulary teacher for the 4B

User direction, 2026-10-18: improve the 4B by distilling from the trained
Gleipnir 27B, comparing binary 0/1 logit distillation with full-vocabulary
distillation at the decision boundary, and find out whether the "dark
knowledge" in the probabilities of unrelated tokens helps. Step 1 (this
record) extracts the teacher signal and diagnoses it; training arms come
after, with their own frozen design.

**Why it is possible.** Qwen3.5-4B and Qwen3.8-27B have identical
vocabularies (248,077 tokens, every id the same string, the same 33 added
tokens; both LM heads 248,320 wide, the rest padding). The 4B ties its
embeddings, the 27B does not. The 4B trains on the same 7,836 strict-filtered
tool-trajectory rows that Gleipnir 27B was trained on.

**Prior evidence.** Unadapted Qwen3.8-27B as a *binary* teacher lost 1.86 ID
pAUROC@20 points against the Kimi control
([qwen38_teacher_4b](../qwen38_teacher_4b/README.md)); it was the weaker
teacher on the training rows. Gleipnir 27B was trained on Kimi's soft
targets for these rows, so its binary training-row outputs are a refit of
Kimi rather than independent judgments; the full-vocabulary signal is new.

## Extraction (`extract.py`, `queue01.sh`)

- **Models.** Trained Gleipnir 27B (the merged `qwen38-27b-s0-01`
  checkpoint, byte-identical to the campaign merge) and unadapted
  Qwen3.8-27B at `1d4bf0f2…`.
- **Prompts.** The full teacher prompt for both (user decision: the
  comparison needs the prompt the base model is benchmarked with, as Kimi's
  targets were), rendered exactly as the frozen benchmark renders it (chat
  template, thinking off, `Prediction:`). The 3,012 ID rows are also
  extracted under the compact student prompt (the trained model's interface)
  for a model × prompt comparison.
- **Rows.** The 8,688-row teacher-prompt training pool (the 7,836 student
  rows flagged) and the 3,012 ID rows.
- **Signal.** Stock vLLM 0.31 BF16, embed pooling of the final normed hidden
  state at the last token; logits = hidden @ lm_head.T in **FP32** (user
  request), which removes the BF16 head's 1/8-logit rounding near ±20 and the
  ties it causes. Hidden states (FP32 copies of BF16 activations) are
  stored, so full logits are reconstructable; per row we also keep
  logprob_0/1, the FP32 margin, the probability outside {0, 1} and the top-64
  tokens.
- **Parity.** 64-row smoke test (Gleipnir, student prompt, ID) against the
  campaign's stock BF16 scorer: margin difference 0.041 mean / 0.118 max,
  consistent with BF16 head rounding plus batch noise. The full run repeats
  this on all 3,012 rows, and checks the base teacher-prompt margins against
  `results/qwen27b_generations_ood/qwen38_27b_{train,id}`.

## Diagnostic (frozen before the full extraction finished)

All on CPU/GPU from the stored signal; nothing here trains a student.

1. **Parity and teacher quality.** Margin agreement with the reference
   scorers; ID macro/per-source pAUROC@20 and AUROC for each model × prompt
   with the FP32 margin; score ties at BF16 vs FP32 head precision.
2. **How much mass is outside {0, 1}**, by model, prompt, source and label;
   which tokens carry it.
3. **Does the tail carry label information beyond the margin?** Logistic
   probes trained on the 7,836 student training rows (teacher prompt) and
   evaluated on the 3,012 ID rows (teacher prompt), per model:
   (a) margin only; (b) margin plus centered log-probabilities of the 256
   non-decision tokens with the most training-row mass; (c) the full hidden
   state (a linear-readout upper bound, i.e. feature distillation). Report ID
   macro pAUROC@20/AUROC; (b) − (a) is the tail's added information.
4. **What training changed.** Base vs Gleipnir on the same rows: margin
   correlation, outside mass, tail-distribution divergence, and agreement of
   each with Kimi's targets on the training rows.

**Decision use.** If (b) does not beat (a) on ID, a full-vocabulary KL at the
decision position is unlikely to add over binary distillation, and step 2
should not spend seeds on it; if it does, step 2 freezes a tail loss and
weight from these measurements. Single extraction, no seeds; probe results
are descriptive.
