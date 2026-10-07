# B200 serving startup

2026-10-07. The validated unchanged-arithmetic restart now takes **95.44 s**,
versus **680.8 s** for the archived selected Direct FP4 recipe. Use locally
staged dependencies and computation-bound compiler identity by default. Keep
the model, kernel recipe, precision, scoring and selected reference unchanged.

## Changes

The Pod's `/workspace` is network-backed FUSE; `/tmp` is local overlay storage.
`scripts/stage_serving_runtime.py` copies the locked Python environment, pinned
Triton override and vendor dependencies into `/tmp/gleipnir-serving-runtime`.
The launcher selects its interpreter and library paths when the lock,
interpreter config and installed package metadata binding matches. It rejects
a stale staged environment. Shared compiler/native caches remain durable on
the volume. Copying is a one-time provisioning step, with eight independent
groups; the final parallel phase took 73.32 s after reusing a partial copy.
Regenerate the local runtime after Pod replacement or dependency changes.

The CPU import trace found Transformers cumulative import time about 51 s,
including 33.63 s in `import_utils`, and about 70.67 s importing the API after
Torch/Transformers. Local CPU imports plus hash-protocol validation completed
in 25.69 s. These CPU-hidden probes diagnose imports; actual server readiness
is the end-to-end measurement below.

vLLM 0.24 hashes the complete `additional_config` dictionary. Our run labels,
control references, profiler paths and validation locations therefore caused
unnecessary cache misses. `ServingCompileConfig` uses the supported
`SupportsHash` protocol, preserving full dictionary access and pickle support.
Hash computation includes kernel/worker sources, runtime versions, arithmetic
settings, native receipt contents and unknown settings. Exclude labels/control
references/audit locations and canary/compare implementation; moving an
unchanged receipt preserves identity. vLLM adds backend fields during config
construction, so the object supports mutation and updates the hash. The first
immutable-wrapper failure before GPU loading is preserved.

The minimal `startup.py` launcher measures readiness and bounded adapter/native
dispatch parity without repeating the concurrency sweep or full shape probes.
Genuine kernel/source/settings changes still invalidate the compiler key.

## Measured restarts

| Run | Time to readiness | Compile/load | Initial kernel warmup |
| --- | ---: | ---: | ---: |
| Archived selected recipe | 680.8 s | prior recorded cache/config | prior receipt |
| `startup_local02`, first new identity/runtime | 441.92 s | 165.95 s | 152.36 s |
| `startup_local03`, changed run label | **95.44 s** | **21.07 s** | **6.34 s** |
| `startup_local04`, compiler-cache mirror | 105.34 s | 16.56 s | 7.07 s |

The two local non-mirror runs have identical compilation identity
`5a00ce5a0c8398b6b28bd93baa522878dee420dc7c60682c693ae9893bd7fd5d`.
The second explicitly loaded cached graphs and AOT entry
`9947a0bcd13c84d57d1a1635aec1c9a89ff2e8b13594a0c45d3b6f05a234a8f1`.
Graph capture still occurs per process and took nine seconds on the warm run.
The initial relocation/new-identity run includes cold compilation and extra
native warmup; do not advertise 95 s for new arithmetic or empty caches.
These are individual restart measurements, not a startup latency distribution.

All successful variants passed the twenty-example serving canary with exactly
zero difference from the selected serving reference. Existing native receipt
failures/finite acceptance remain recorded; relocation does not change them.

## Further cache mirroring

The follow-up copies existing vLLM/Inductor cache keys to fixed local paths and
persists new/changed entries back without deleting durable keys. Local file
metadata snapshots avoid scanning/transferring the entire shared tree on every
writeback. Shared top-level keys discovered later can fall back through
directory links. Staging took 10.32 s; the measured run persisted 101 changed
entries in 4.69 s. Both trees and their source identities remain recorded.

Cached graph loading fell from 19.26 to 14.71 s, but complete startup regressed
to 105.34 s. No net benefit was established. Keep mirroring **opt-in** with
`GLEIPNIR_COMPILER_MIRROR=1`; defaults use shared compiler caches. Remaining
startup includes imports in the API and spawned engine, model/plan setup,
compiled-graph reconstruction, profiling and process-specific capture. The
limited mirror benefit suggests graph loading includes substantial work beyond
filesystem access. Further import/process changes require their own measured
compatibility test; no multiprocessing or compilation downgrade was selected.

## Quality and retained state

One excluded warmup plus three c128 passes on the same frozen 320-row systems
development workload gave 198904/198443/199017 input tokens/s, median **198904**
(+1.04% against the archived five-repeat reference). Treat this short check as
no observed throughput regression, not a new kernel speedup. Source-macro
AUROC is unchanged; pooled AUROC is 0.885314 (-0.0527 percentage points).
There were zero threshold flips, mean/max score difference 0.001078/0.097038.
Batch scheduling and existing FP4 arithmetic produce repeat variation. Final
ID was excluded; no quality/reference promotion follows.

Artifacts under `results/b200_attention_gdn_serving/`:

- `startup_diagnosis01`: import traces, staging logs, compatibility checks.
- `startup_local01`: preserved pre-GPU immutable-wrapper failure.
- `startup_local02`, `startup_local03`, `startup_local04`: source snapshots,
  readiness, exact parity, compile identities and native receipts.
- `startup_quality01`: three warm throughput/quality passes and all metrics.
- `startup_collection01`: logs, staged/mirror manifests, cache writeback,
  retired server receipts, source-verified health and locally verified checksums.

All 45 condition source hashes matched at collection. Twenty-five focused tests
and Ruff passed, including protocol mutation/pickle, unsafe receipt paths,
dependency staleness, cache identity changes, disjoint package copying and
durable cache writeback. Selected baseline SHA256 remains
`39811c43e0b4bbf574e682d7b21f09e394909af3af4a69f3b398193cace89166`.
Retain the healthy mirror-trial server API/engine **104702/104725** on the
existing NC2 B200, port 8010. Its arithmetic is the selected Direct FP4 recipe;
the next ordinary launch uses the faster non-mirror configuration. Only the
engine owns GPU memory. The staged packages and optional mirror are ephemeral;
important receipts and authoritative caches remain on the network volume.
