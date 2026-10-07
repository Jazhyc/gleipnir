# Large prefill graphs on the repaired monitoring endpoint

Hypothesis: the responsive EU host and repaired mutation analysis allow larger
piecewise graphs to improve end-to-end score serving. Retain the selected
cached two-logit FP4/MXFP8/Gigatoken/direct-host stack and validated mutation fix.
Reuse its saved three c1 and six c128 repeats; no new timing control or final ID.

Capture 1,16,64,128,256,512,1024,1536,2048,4096,8192,16384,32768 tokens.
Keep attention/GDN/KV-update splitting operations eager. Preserve the earlier
12.5% padding limit, FP4 producer-band boundaries and temporary profile-cache
allocation fix. Verify the pinned dispatcher/model-runner source identities.
Retire the existing server before launching its replacement, retaining caches.

Before timing, compare graph bypass/replay/repeated changed-input outputs at
12 boundary lengths for two token fixtures through the supported classifier
API. Require finite outputs, score error <=0.005, logit error <=0.125 and
repeat score error <=1e-6, with padded/large replay and fallback coverage.
The normal endpoint additionally checks exact prompt lengths and the accepted
20-row adapter canary. Stop on any gate, provenance or server failure.

Measure quick64/c1 and full320/c128 with one excluded warmup then three/six
repeats. Report input tokens/s, latency, AUROC in percentage points, calibration,
score drift, ties and flips against the selected mutation03 reference. A separate
profile must confirm actual replay; dispatch counters alone are insufficient.
No automatic promotion: keep the repaired reference unless the user selects
this candidate. Preserve failures and a usable warm service. No capacity changes.

```bash
PYTHONPATH=src:. python -m experiments.b200_score_graphs.run --name graphs01
```

The runner inherits the live score parent's environment in memory, keeps the
existing mutation-validation receipt, and reuses the frozen remote runtime.
