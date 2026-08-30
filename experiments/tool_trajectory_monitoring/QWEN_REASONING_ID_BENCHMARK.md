# Qwen3.5 reasoning benchmark on held-out in-distribution data

This campaign repeats the frozen direct-logit, visible-reasoning, and native-
thinking comparison on the paper's two held-out validation distributions:
946 STRIDE-test rows and 2,066 Gloom-Exfiltration rows. It changes only the
evaluation population relative to the completed 100-row source canary.

STRIDE test is the `test` split of public revision
`aksh-n/rich-trajectories-cleaned@92a44c8`. Gloom-Exfiltration is reconstructed
from public revision `Syghmon/gloom-data-exfiltration@ee3c56b`. The latter has
2,190 raw rows; retaining trajectories at or below 27,900 raw Qwen3 tokens
produces the paper's 2,066 rows and 1,035/1,031 class balance. More importantly,
the retained ground-truth sequence matches all 2,066 rows in the paper author's
released benchmark predictions, and the selected source-ID digest is frozen in
the materializer. The 27,900 raw-token boundary is the operational equivalent
of the paper's 28,000-token prepared-input cap after its short evaluation
envelope.

Neither upstream dataset card declares a license. Keep the downloaded files and
derived records ignored, use them only for internal research evaluation, and do
not redistribute them. The manifest records repository revisions, source file
checksums, tokenizer revision, transformation, selected-ID digest, and output
checksum.

Download the two pinned parquet files as `stride_test.parquet` and
`gloom_exfiltration.parquet` under
`data/tool_trajectory_monitoring/source/id_evaluation/`, then materialize:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_qwen_reasoning_id_benchmark
```

Run a four-row backend canary into a disposable output directory, then start the
resumable full campaign:

```bash
python -m experiments.tool_trajectory_monitoring.benchmark_qwen_reasoning \
  --config experiments/tool_trajectory_monitoring/qwen_reasoning_id_benchmark.json \
  --input data/tool_trajectory_monitoring/qwen_reasoning_id/records.jsonl \
  --output results/tool_trajectory_monitoring/qwen_reasoning_id_canary \
  --limit 4

python -m experiments.tool_trajectory_monitoring.benchmark_qwen_reasoning \
  --config experiments/tool_trajectory_monitoring/qwen_reasoning_id_benchmark.json \
  --input data/tool_trajectory_monitoring/qwen_reasoning_id/records.jsonl \
  --output results/tool_trajectory_monitoring/qwen_reasoning_id
```

One persistent text-only vLLM engine runs all three arms. Reasoning generation
keeps the frozen 16,384-token cap and Qwen-recommended sampling family. The run
atomically checkpoints every eight rows and can be resumed without recomputing
completed records.
