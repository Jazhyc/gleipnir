# Batch-composition dependence of monitor scores

User question, 2026-10-15: whether the fused FP4 SwiGLU MLP kernels add noise.
The serving MLP chooses its gate/up + SwiGLU producer by the total rows in the
engine step:

- under 1,536 rows: the reference path;
- up to 4,096: the overhead path;
- above 4,096: direct FP4 output with local block scales.

A request's arithmetic can therefore depend on what it is batched with.

`probe.py` uses the timing worker so engine step sizes are recorded. It scores
short canary prompts alone, then queued behind a 32K blocker together with a
1K or 4K filler, so prompt and filler share the next step (verified from step
token counts in every repeat).

`batching02` (3 repeats per condition, EU-RO-1 B200, adopted default):

- Every condition repeats bitwise within itself. No run-to-run nondeterminism
  was observed.
- 1K-filler steps (1,307–2,015 tokens, reference or overhead band): scores
  equal the alone scores for all 4 prompts.
- 4K-filler steps (4,379–5,087 tokens, direct-output band): 2 of 4 prompts
  change, 0.01243 → 0.00758 and 0.04209 → 0.01591, repeatably.

So scores at concurrency above 1 depend deterministically on batch composition.
This is consistent with the earlier c4-versus-c1 canary differences (up to
0.053) and c128 pass-to-pass differences (MAE 0.0035, max 0.29). The direct FP4
output path above 4,096 rows is the most likely source. It matches the row
threshold and the documented score shift of that path. Other row-band-dependent
components were not isolated individually.
