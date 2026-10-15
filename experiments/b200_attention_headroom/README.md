# MXFP8 D256 attention headroom on B200

Diagnostic, 2026-10-15, EU-RO-1 B200 `GPU-e5eb9962` (same host as the 1M
baseline, a different GPU), driver 580.126.20. The goal is to decide whether
attention kernel configuration or engineering is the next long-context lever.

**Configuration space.** The pinned cuDNN DSL serves the MXFP8 (256, 256)
prefill kernel only at `cga=1`: `supported_cgas_for` returns `(1,)`.
`split_kv` applies only to decode-style paged queries. The
`sm100/prefill_d256_mxfp8.py` template has no exp2 MUFU/FMA split hook.
NVIDIA's notes list every d256/d512 flavor as unmeasured for that split. Tile
and stage sizes come from a fixed config. There is no configuration screen to
run without editing the kernel template.

**Measurement.** `bench.py` times `paged_forward` for one 32K query chunk at
the end of 32K–1M histories (bottom-right causal), and the three MXFP8
producers separately. It samples SM clock and power with `nvidia-smi` every
100 ms. Kernel FLOPs count causal query/key pairs. The peak assumes 16,384 dense
FP8 FLOP per SM per clock across 148 SMs.

| History | Kernel | Packing share | PFLOP/s | SM clock | Peak power | Fraction of clock-adjusted peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | 3.90 ms | 5.2% | 2.26 | 1,965 MHz | 244 W | 0.47 |
| 64K | 12.09 ms | 2.7% | 2.18 | 1,957 MHz | 365 W | 0.46 |
| 128K | 29.19 ms | 2.0% | 2.11 | 1,965 MHz | 546 W | 0.44 |
| 256K | 63.74 ms | 1.7% | 2.07 | 1,541 MHz | 731 W | 0.55 |
| 512K | 136.0 ms | 1.6% | 2.00 | 1,470 MHz | 997 W | 0.56 |
| 1M | 310.8 ms | 1.8% | 1.78 | 1,237 MHz | 994 W | 0.59 |

The kernel reaches under half of clock-adjusted FP8 peak when unthrottled. At
512K and above, the GPU sits at its 1,000 W limit and the clock falls to
1.2–1.5 GHz. Packing the history is under 2% from 128K up, which caps the gain
from caching converted history.

```bash
... -m experiments.b200_attention_headroom.bench --output results/b200_attention_headroom/bench01.json
```
