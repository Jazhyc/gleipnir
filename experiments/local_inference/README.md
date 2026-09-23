# Local merged Gleipnir 4B baseline

Hypothesis: a BF16 merge of the released Gleipnir 4B LoRA provides an efficient
fixed-weight baseline for later vLLM serving optimizations on the local RTX 4080.
This campaign uses only a frozen 512-row ID subset throughout, as requested on
2026-09-23; neither full-ID confirmation nor strict-OOD selection is in scope.

Reconstruct CoT-removed trajectories from the checksum-pinned STRIDE and Gloom
sources using the existing ID loaders and source-aware CoT remover. Preserve the
compact student prompt and non-thinking `Prediction:` boundary. Stratify by
source, hard label, and length quartile, then select by seeded ID hash. Freeze
membership before scoring. Source labels are evaluation-only; no teacher calls.
All source datasets remain ignored and must not be redistributed (the existing
source inventory records unspecified upstream licenses).

The frozen subset has 5,760,843 prompt tokens, ranging from 1,291 to 29,475
tokens per row, with SHA-256
`f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf`.
Reconstruction matches the recorded 3,012-row population's 33,750,959 prompt
tokens; its JSONL metadata schema is new, so it is not claimed to match the
historical full-file hash. The four short eager canaries have lengths
1,291, 1,354, 1,716, and 2,195; no prompt is shortened for a canary.

Download the published master adapter at a resolved immutable Hub revision and
the pinned Qwen3.5-4B base. Keep the FP32 master unchanged. Merge on CPU with
PEFT safe merging, save BF16 inference weights, and hash every artifact. A
bounded Transformers canary compares unadapted base, FP32-master adapter, and
merged weights; a matched vLLM canary then checks serving agreement. These
eager canary includes one short actual subset row per source/label pair; vLLM
also runs the longest subset input as a separate memory/kernel canary. Require
nonzero adapter effect, mean absolute score error <= 0.02, maximum error <= 0.10,
and correlation >= 0.99 when scores have nonzero variance. Report all flips;
these are baseline-construction tolerances, not future optimization tolerances.
Stop before full-subset evaluation if a gate fails.

Use one persistent vLLM engine, one constrained decision token, explicitly
requested 0/1 logprobs, and three warm timed passes over identical input order.
Disable prefix caching initially to avoid replay-dependent timings. Record raw
decision logprobs, margins, scores, per-source/macro ranking and calibration,
threshold flips, ties, repeat variation, initialization time, and prompt tokens/s.
Stop on missing or nonfinite scores, OOM, truncation, artifact/hash drift, or a
failed parity gate. Do not change data or quality criteria in response to scores.
Future optimizations will be compared against this merged baseline on these
same rows; set their tolerances using baseline repeat noise before tuning.

The workstation has a 16 GB RTX 4080 and 47 GiB RAM. No Slurm or cloud capacity
is involved. No in-chat scheduling tool is available in this session; monitoring
uses active-turn startup checks and continued active-turn checks during scoring,
and cannot promise agent follow-ups after the turn ends.

Startup exposed two launcher/build issues: explicit Python invocation did not
put `.venv/bin/ninja` on PATH, and FlashInfer's sampling Ninja file does not
escape the space in this checkout's `AI Control` directory. The runner now
adds its executable directory to PATH. The frozen environment sets
`VLLM_USE_FLASHINFER_SAMPLER=0`; this workload uses greedy one-token decisions,
and vLLM's native sampling path avoids that optional JIT dependency. FlashAttention
2 and Triton/FLA GDN remain the automatic model-kernel choices on SM89. Both
failed startup attempts and logs are retained separately.

Provenance: reuse `prepare_qwen_reasoning_id_benchmark` source loaders,
`id_cot_only_evaluation.prepare.strip_cot`, the shared student prompt, and shared
binary metric functions. This is a separate local campaign; historical configs
and result contracts are unchanged.

## Run

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python -m experiments.local_inference.fetch
.venv/bin/python -m experiments.local_inference.run
```

The runner prepares the subset, merges the release, checks the eager reference,
and starts the vLLM baseline sequentially. Each phase has a separate log under
`logs/local/local_inference/`; status and results are in
`results/local_inference/`. Failed attempts must be retained before rerunning
the baseline; completed timed passes are not silently overwritten or resumed.

For local temperature/clock diagnostics, run
`.venv/bin/python -m experiments.local_inference.telemetry` in a second process
after `baseline/status.json` exists. It samples every ten seconds and exits when
the campaign completes or fails. This is instrument logging, not scheduled agent
monitoring. The first baseline showed software thermal throttling at 87 C;
compare future configurations under comparable cooling and record telemetry.
Sampling began partway through its first pass, so those samples are not a full
startup or first-pass thermal trace.
