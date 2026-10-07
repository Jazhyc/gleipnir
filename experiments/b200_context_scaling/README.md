# Fixed 2K reference scaling

Measure the existing warm selected B200 scorer on 320 distinct, exactly 2,048
token text windows derived from longer full320 systems prompts. The user chose
prompts concentrated near 2,048 tokens rather than mixed lengths averaging 2K.
There are no natural full320 prompts in the 1,792–2,304 token interval.

Hypothesis: shorter, uniform contexts change throughput saturation and request
latency relative to the earlier heavy-tailed full320 sweep. The intervention is
input geometry only; preserve the reference process, weights, quantization,
FCFS scheduler, sequence/token budgets, native tokenizer and cache policy.
Use concurrency 1, 2, 4, 8, 16, 32, 64 and 128, with one excluded warmup and three
timed repeats per level. Use the same ordered windows at every level and fresh
HTTP pools. Request latency excludes client semaphore waiting; it includes
HTTP, tokenization and engine waiting. Report input tokens/s and requests/s.

Construct windows deterministically using the checkpoint tokenizer, verify
source token counts, decode/re-encode to exactly 2,048 tokens, require unique
text hashes, then freeze prompts and source/window provenance before timing.
These cropped texts are an artificial systems workload: remove labels, do not
compute AUROC or infer model quality from them. Recheck the unchanged real
adapter canary and all endpoint counts/finite outputs. Save scores for numerical
diagnostics. Prior full320 timing is contextual, not a matched length-only
control: both average length and distribution change. No promotion or held-out
selection. Stop on identity/token/count/canary failure, nonfinite responses,
server failure/OOM or completion. Leave the warm server and capacity intact.

```bash
PYTHONPATH=<frozen-source-bootstrap>:src:. <serving-python> \
  -m experiments.b200_context_scaling.run --name twok01
```

Results and executed sources are under `results/b200_context_scaling/<name>/`.
Measure the missing c8 point on the unchanged original full320 inputs too;
preserve it as an addendum here without rewriting the original sweep.
The runner requires the live process/command to match the completed reference
sweep at `results/b200_score_scaling/scale01`; it never starts or stops a server.
Export tables and figures with `python -m experiments.b200_context_scaling.summarize
results/b200_context_scaling/twok01 results/b200_score_scaling/scale01`.
