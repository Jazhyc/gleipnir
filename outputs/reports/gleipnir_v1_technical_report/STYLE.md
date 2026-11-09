# Writing style

Rules for all text in this report: body, captions, tables, footnotes and the
abstract. Follow them when drafting or editing.

## Language: simplified technical English

- Write short sentences with one statement each. Aim for at most 20 words;
  split longer sentences.
- Use the active voice and the simple present or past tense. Use "we" for the
  authors' actions.
- Use common words with one meaning. Write "use", not "leverage" or "utilize";
  "show", not "demonstrate"; "about", not "approximately".
- Use one term for one concept throughout. Do not vary terms for style. For
  example, write "monitor" every time, not "monitor", "classifier" and
  "detector".
- Use established field terms when they help readers place the work, for
  example AI control, distillation, LoRA, AUROC, pAUROC@20, out-of-distribution
  (OOD). Define each term or abbreviation at first use.
- Keep one idea per paragraph. Open the paragraph with that idea.
- Connect the sentences of a paragraph. Start a sentence with what the
  previous sentence ended on, and use a linking word ("therefore", "because",
  "in addition") where the logic needs one. Short sentences must still read
  as one line of argument, not as a list of facts. A sentence may pass 20
  words when splitting it would break that line.

## Tone

- Write formally. Do not use contractions, colloquial phrases or rhetorical
  questions.
- Do not use promotional words such as "novel", "remarkable", "powerful",
  "state-of-the-art" or "game-changing". Let the numbers show the result.
- Use "significant" only for a statistical test, and name the test.

## Qualifiers

- Do not hedge with words such as "somewhat", "fairly", "arguably", "may
  potentially" or "it seems". Do not intensify with words such as "very",
  "clearly", "highly", "extremely" or "dramatically".
- Express uncertainty with numbers: confidence intervals, seed spread or sample
  sizes. When a limitation applies, state it once, explicitly, where it
  matters.

## Claims

- Report only what the results show directly. Each claim names the model, the
  evaluation set, the metric and the value.
- Separate observations from interpretation. Mark an interpretation as one
  ("One explanation is ...") and say what would test it.
- Make causal claims only when a controlled comparison or ablation supports
  them. Otherwise, describe the association.
- Give both values in a comparison, plus the difference and its interval when
  available. For example: "Mean OOD pAUROC@20 is 0.xxx for Gleipnir 4B and
  0.yyy for the baseline, a difference of 0.zzz (95% interval 0.aaa to 0.bbb)."
- Report negative and null results with the same detail as positive results.
- Do not generalize beyond the evaluated models, data sources and settings.

## Numbers

- Report AUROC-type metrics with three decimals, and use the same precision
  across a table or comparison.
- Give intervals as 95% bootstrap intervals unless stated otherwise, and say
  what was resampled.
- Record the source and revision of every reported number in
  [sources.md](sources.md).

## Citations

- Citations are numeric. Use `\citep` for a reference in brackets and
  `\citet` when the authors are part of the sentence ("Sinha et al. [2]
  train ...").
- Cite a work once per paragraph, at its first mention. After that, refer to
  the authors by name or as "they".
