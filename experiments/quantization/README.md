# Quantization experiments

Quantization-aware training and distillation (QAT/QAD) and calibration. Add new experiments to this table.
The QAD template experiment `quantization_recovery` stays directly under
`experiments/`; see
[Kept at their recorded paths](../README.md#kept-at-their-recorded-paths).

| Experiment | Question |
| --- | --- |
| [`int4_calibration`](int4_calibration/) | Real-activation INT4 calibration screen |
| [`kimi_direct_qat`](kimi_direct_qat/) | Direct Kimi K3 quantization-aware distillation |
| [`native_fp8_qat`](native_fp8_qat/) | Native FP8 QAT quality test and MXFP8 backward diagnostic |
| [`native_mxfp8_qat`](native_mxfp8_qat/) | Native MXFP8 backward quality test |
| [`qad_budget`](qad_budget/) | Short QAD budgets |
| [`qad_ood`](qad_ood/) | Full QAD checkpoint OOD transfer |
| [`qad_v02`](qad_v02/) | QAD for Gleipnir 4B v0.2 |
| [`qat_training_optimization`](qat_training_optimization/) | Merged-weight QAT training optimization |
