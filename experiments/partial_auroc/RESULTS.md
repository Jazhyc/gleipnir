# Initial launch record

On 2026-10-12, `strict-pauc02` passed preparation and began the single variant
defined in [config.yaml](config.yaml) and [README.md](README.md). The detached
driver launched at 00:14:41 CEST on existing Runpod B200 `rzarg8j8x0so3i` in
NC2. Driver PID 6712, Trainer PID 6881; verify live receipts before using these
historical PIDs. No new capacity or lifecycle action occurred.

At the initial startup check, seven of 245 optimizer updates had completed.
The fresh parameter-gradient replay canary passed with maximum error
`5.551115123125783e-17`; all observed native score/replay margin differences
were zero and gradients were finite. 39 focused tests and lint passed.
Implementation: `d6b1637`. Startup compiled distinct gradient/no-gradient
specializations into the persistent cache; expected recompilation messages and
the existing experimental cuBLAS preference warning are retained in logs.

The earlier `strict-pauc01` attempt failed at offline tokenizer lookup because
the custom launcher omitted persistent HF cache paths. No model loaded and no
optimizer update occurred. Its launch/failure receipts and logs remain intact.
The corrected attempt changes cache/PATH setup and artifact name only; objective,
data, initialization, optimizer and selection settings are unchanged.

Quality results are pending. The automated queue runs final-epoch master/merge/
FP32-head serving checks and complete ID/OOD evaluation, with the archived strict
FP32-head predictions as the reference. It writes the final `comparison.json`
and stops on failure without retraining or waiving parity. The initial estimate
was completion around 01:45–02:00 CEST; this is an estimate, not a completion
receipt or scheduled agent follow-up.

Live state: `results/strict-pauc02/runner.json`; objective checks:
`partial_auc_training.json`; final metrics: `comparison.json` in the same
directory. Logs: `logs/runpod/strict-pauc02/`. Read those receipts for later
status; this document records the initial launch only. The scorer retires after
evaluation and the Pod remains running. Important initial receipts and the
failed startup logs were collected locally. No recurring agent scheduler is
available; the earlier unattended-execution preference applies to the queue.
