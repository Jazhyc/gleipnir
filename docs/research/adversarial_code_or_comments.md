# Does the monitor respond to what the code does, or to how the comments describe it?

Design draft, 20 October 2026. Nothing has run. Freeze a protocol in `experiments/adversarial_code_or_comments/` before any paid step. Proposed by Abhinav, after Edward Young's "Alignment Engineering vs Misalignment Science" (LessWrong, 5 October 2026).

The aim is an explanation of what Gleipnir's score depends on, tested by its predictions. We are not trying to make the score move.

## Why now

Earlier runs hint at an answer but can't settle it:

- **A1 (discrimination).** Injected text raises the scores of honest and backdoored code alike. Within-task ranking of backdoor over honest falls from 95.7% to 72–78%.
- **A2 (cover comments).** LLM-written comments that rationalise backdoors more often *raised* the score than lowered it: 248 rises against 79 falls. But the comments were unmatched in length, written per task, and many generations failed.
- **v0.3 replay.** Honest programs carrying length-matched *neutral* text were flagged 49/192 times by v0.3, against 6/192 by v0.1. Any added text may now look suspicious.

None of these crosses *what the code does* with *what the comment claims*. This design does.

## Design

**Units:** APPS tasks with a valid honest solution and a valid backdoored solution, paired by task.

**Data split:** 200 development tasks and 200 held-out tasks, both drawn with a fixed seed. The held-out tasks stay unscored until the explanation and its predictions are committed.

**Factor 1, code:** honest or backdoored, from the same task.

**Factor 2, text:** one comment block, placed as a header docstring, with four levels:

| Text | On honest code | On backdoored code |
| --- | --- | --- |
| None | none | none |
| Truthful | explains the algorithm accurately | explains the algorithm and names the special case accurately, including its trigger |
| Misleading | claims a hidden special case that isn't there | presents the special case as a legitimate requirement (a cover story) |
| Unrelated | the truthful explanation written for a *different* task, so the same style and length but irrelevant | same |

So the core is a 2×2 of what the code does against what the comment claims:
- **Claims clean:** honest + truthful, and backdoored + misleading.
- **Claims a special case:** honest + misleading, and backdoored + truthful.

The unrelated and none levels separate "any added text" from what the text says.

**Factor 3, repetition (run separately):** on 100 development tasks, repeat each text block 1, 2, 4 or 8 times.
- If the monitor counts cues, flagging grows with repetition.
- If it reads content, repetition of an accurate claim changes little.

## Making the texts valid

- **Finding the backdoor.** The released rows don't record where the backdoor is, and the two solutions were written independently, so a diff doesn't isolate it.
  - An LLM is given the problem and both solutions, and must name the special case and a trigger input.
  - Then run both programs on that input in a sandbox on the GPU box, never on the Mac. The trigger is accepted only if the outputs differ.
  - Tasks without a confirmed trigger are dropped and counted.
- **Writing.** One LLM writes all texts to a target length. Each text must fall within ±10% of the task's truthful-explanation length.
- **Code unchanged.** With comments stripped (Python `tokenize`), every program must be byte-identical to the original; this is the harm-preserved check.
- **Fidelity.** A second LLM from another family, plus hand reading of 40 random items, confirms that each "truthful" text is true and each "misleading" text is false. Failures are excluded and counted.

## The explanation and how it is tested

**On the development split,** describe the score with a small, readable model of the margin (logit 1 − logit 0). The terms are the code effect, the claim effect, any-text, length, repetition and their interaction. Then write down, before touching held-out data:

1. the predicted mean margin change and flag rate for each cell, with intervals;
2. which effect is larger, code or claim;
3. the repetition slope for each text type.

**Prediction tests, in pre-committed order:**
1. Same monitor (4B v0.3), the 200 held-out tasks.
2. A second monitor: Gleipnir 27B with its recommended teacher prompt, on the held-out tasks.
3. Optional third: the old 4B v0.1, already set up, which shows whether the explanation survives retraining.

**Success:** for the held-out predictions, the cell means fall inside the predicted intervals and the ordering of cells is preserved. For the second monitor, the sign of each main effect is preserved. A miss is a result: it means the explanation was local to one monitor or one split.

**Optional white-box forensics:** what share of the attribution to the decision logit falls on comment tokens against code tokens, per cell. This tests the same explanation from the inside.

## Predictions (draft, to be frozen with the protocol)

- **Code effect:** backdoored code scores above honest code in every text condition.
- **Claim effect:** for v0.3, texts claiming a special case raise the margin for both code types. The claim effect is at least half the size of the code effect.
- **Any added text:** unrelated text raises v0.3 scores relative to no text, which is the replay's neutral-text finding.
- **Repetition:** flagging rises with repetition for texts claiming a special case, and stays roughly flat for truthful texts about honest code.

## Cost

- **Generation:** about 400 tasks × 4 texts, plus trigger finding. Roughly $10–20 on OpenRouter, with failures logged as their own category.
- **Scoring:**
  - **Main grid:** 400 × 2 × 4 = 3,200 rows.
  - **Repetition:** 100 × 2 × 3 × 3 = 1,800 rows.
  - **Total:** about 10M tokens, roughly 15 minutes on the H100.
- **27B held-out check:** 1,600 rows, about 30 minutes on an 80 GB GPU.
