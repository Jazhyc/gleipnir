# Full QAD checkpoint OOD transfer

Evaluate the fixed full-epoch `quantization-recovery01` checkpoint on the same
6,395 frozen OOD examples as the preceding FP4/FA4 evaluation. Test whether
same-model quantization-aware distillation changes transfer, and separately
measure optimized-serving drift against this checkpoint's BF16 scores. There is
no training, teacher call, checkpoint/threshold selection or OOD promotion.
The checkpoint is the original 245-update QAD adapter, not a short-budget or
fresh direct-Kimi variant. Its FP32 master SHA256 is
`eb08e1371e72cde91243b4f0792270aa52ea97e855d9a9099a4b436f85659770`.

Reuse the shared [frozen OOD runner](../filtered_fp4_ood/README.md) with this
directory's explicit configuration. Preserve the old campaign's code archives
and receipts. Verify the completed FP4 OOD run and idle existing NC2 B200 before
launch. Replace only the owned ephemeral merged slot after checking its previous
manifest and every file; reconstruct the original QAD merge exactly from the
preserved key-converted FP32 export and pinned base. Bind the QAD master/config,
original merge, recovery receipt, BF16 master-reference scores and exact canary
prompt order/hashes. Preserve all masters and persistent caches.

Run BF16 first and optimized serving second, retiring each scorer before the
next. Require three fresh master-score canaries per path (MAE <=0.02,
correlation >=0.99, nonzero adapter effect), BF16 native audit or fresh optimized
FP8 projection audit, and the existing source-bound optimized native checks.
Optimized serving uses the original QAD deployment recipe: FP4 MLP/GDN, FP8
full-attention projections and repaired MXFP8 attention. No precision fallback.

Each path has exactly one descriptive c128 pass over unchanged IDs, labels,
rendered prompts and lengths. No prose restoration. Report source-macro raw
normalized pAUROC@20, pooled/per-source AUROC, calibration, threshold diagnostics,
ties and prompt tokens/s. Repeat variation is unmeasured. Compare both cached original FP4 BF16
passes, filtered SDPA, released Gleipnir 4B and regular unfiltered controls.
Same-checkpoint BF16 versus optimized isolates serving effects; historical
training comparisons retain seed/runtime/host confounds. This OOD suite has been
used repeatedly, and higher-level lineage independence remains unestablished.

Stop on identity drift, failed parity/native checks, missing/nonfinite scores,
truncation, backend failure or incomplete coverage. Final checkpoint only;
no selection or promotion. Artifacts: `results/qad-ood01/`, with separate
`bf16/` and `optimized/` receipts; logs: `logs/runpod/qad-ood01/`.
Leave capacity running. Launch with:

```bash
python -m experiments.filtered_fp4_ood.launch --config experiments/qad_ood/config.yaml
```

No in-chat scheduling tool is available. Detached stage automation does not
promise later agent wakeups; verify canaries and advancing outputs during startup.
