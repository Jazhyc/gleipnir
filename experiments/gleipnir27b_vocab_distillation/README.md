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
  rows flagged) and the 3,012 ID rows; for Gleipnir also the frozen 6,395-row
  OOD suite under the teacher prompt (`queue02.sh`, user request after the ID
  result below), scored against its student-prompt OOD 0.9087 and the
  unadapted model's teacher-prompt OOD 0.8901.
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

## Interim result: Gleipnir 27B under the teacher prompt (ID)

Gleipnir 27B was trained on the compact student prompt, yet on the 3,012 ID
rows (FP32 head) it ranks better under the full teacher prompt: macro
pAUROC@20 0.9456 vs 0.9216 (AUROC 0.9875 vs 0.9821; Gloom 0.9187 vs 0.8865,
STRIDE 0.9726 vs 0.9567), with 9.9% more prompt tokens. The FP32 head alone
does not change ranking (0.9216 vs the BF16 campaign's 0.9217), though it
separates all 3,012 margins where the BF16 head gives 122 distinct values.
ID has been read many times, so this is not a selection; OOD follows.

## Result: Gleipnir 27B under the teacher prompt (OOD)

On the frozen 6,395-row OOD suite (unweighted six-source mean), Gleipnir 27B
scored under the full teacher prompt reaches **0.9352 Mean-OOD pAUROC@20**
(AUROC 0.9790), against 0.9087 under its compact student prompt, 0.8901 for
the unadapted model under the teacher prompt and 0.9084 for Kimi K3. One
extraction, one seed; ID has been read many times, OOD once for this arm.

| strict OOD, pAUROC@20 | teacher prompt | student prompt | unadapted, teacher prompt |
|---|---:|---:|---:|
| **Mean-OOD** | **0.9352** | 0.9087 | 0.8901 |
| AgentDojo | 0.9990 | 0.9983 | 0.9657 |
| CUA-SHADE-Arena | 0.8911 | 0.8806 | 0.8648 |
| IaC | 0.9931 | 0.9871 | 0.9678 |
| MLE-Sabotage | 1.0000 | 1.0000 | 1.0000 |
| Rogue Eval | 0.9982 | 0.9989 | 0.9862 |
| SHADE-Arena | 0.7298 | 0.5875 | 0.5560 |

Most of the gain is SHADE-Arena (+14.2 points), the source whose score is
dominated by the `american_airlines_2` task
([audit](../../docs/findings/shade_length_audit.md)); the other sources move
by at most 1.1 points.

## Diagnostic result (`diagnostic.json`)

- **Parity.** Gleipnir student-prompt ID vs the campaign scorer: MAE 0.044,
  correlation 0.99988, 17 threshold flips. Base teacher-prompt ID / train vs
  `qwen27b_generations_ood`: MAE 0.112 / 0.110, correlation 0.9997 / 0.9998.
- **Quality (ID macro pAUROC@20).** Gleipnir 0.9216 student / 0.9456 teacher
  prompt; base 0.8478 student / 0.8869 teacher prompt. The teacher prompt
  helps both models by 2.4–3.9 points.
- **Ties.** FP32 head: 3,011 distinct ID margins of 3,012; BF16 head: 133
  (Gleipnir) and 214 (base). Mean BF16–FP32 margin difference 0.06–0.07.
- **Mass outside {0, 1}.** Median 5.4–6.6e-4 for Gleipnir on every job;
  base 1.5e-5 (student prompt) to 1.3e-4 (teacher prompt). The 256 selected
  tail tokens hold 97.7% (Gleipnir) / 99.9% (base) of it.
- **Probes** (train on the 7,836 student rows, teacher prompt):

  | probe | Gleipnir ID | Gleipnir OOD | base ID |
  |---|---:|---:|---:|
  | margin | 0.9456 | 0.9352 | 0.8869 |
  | margin + 256 tail log-probs | 0.9460 | 0.9420 | 0.8596 |
  | hidden state (linear) | 0.9486 | 0.9459 | 0.9117 |

  Gleipnir's grouped-CV training AUROC is 0.9998–1.0 for every arm (it was
  trained on these rows), so C selection is uninformative there. The tail's
  OOD +0.7 points is SHADE-Arena +9.7 offset by AgentDojo −3.2 and
  CUA-SHADE −1.5.
- **What training changed.** Margin correlation with the base 0.951 (train)
  / 0.973 (ID); median |margin| 4.6 vs 5.3 (train) and 3.4 vs 5.4 (ID), so
  Gleipnir is less extreme; top-8 token overlap 0.79. Training-row margin vs
  Kimi logit correlation 0.983 (Gleipnir) vs 0.951 (base).

**Decision (frozen rule).** The tail does not beat the margin on ID (+0.0004
for Gleipnir; −0.027 for the base), so step 2 should not spend seeds on a
full-vocabulary KL at the decision position. The linear hidden-state readout
adds at most 0.3 (ID) / 1.1 (OOD) points over Gleipnir's margin, a small
ceiling for feature distillation. The large effect is the prompt: Gleipnir's
binary margin under the teacher prompt is the strongest teacher signal
measured, on ID and OOD.
