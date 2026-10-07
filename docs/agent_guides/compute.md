# Compute and experiment operations

These instructions are required for the tasks routed here by the root
[AGENTS.md](../../AGENTS.md). Standing preferences were moved from that
file on 2026-10-07; their scope and historical acceptance remain intact.

Never terminate an instance or launch billable capacity without the user's
explicit instruction. Collect important artifacts before termination.
Verify live runtime, hardware and connection metadata before using a target;
historical status in documentation is not a live inventory.

## Experiment monitoring

When launching or monitoring a long-running experiment, check startup frequently
(roughly every 30–60 seconds) until model loading, compilation, and any preflight
canaries have passed and actual training steps or evaluation outputs are advancing.
Then schedule agent follow-ups every 10 minutes using the available in-chat
scheduling/heartbeat mechanism. Each follow-up should inspect progress, logs,
GPU health, and failures; report meaningful changes and revise the ETA when
supported by measured throughput. Recheck startup closely for each new queued run.
Stop the recurring follow-ups when the campaign completes, the user asks to stop
monitoring, or a blocker requires user input; collect and summarize final results.

These heartbeats must wake the agent to inspect the experiment. A remote queue
timer, process watchdog, or log message is not a substitute. Verify that scheduling
succeeded before claiming monitoring is active. If this session has no scheduling
tool, explicitly tell the user that limitation; active-turn waiting can support
checks but cannot promise a follow-up after the turn ends. Do not silently replace
agent follow-ups with a remote polling loop.

## Persistent caches

Standing user preference, 2026-10-03: reuse persistent compiler and kernel caches
across compatible training runs on the network volume. Do not create a cold cache
namespace for each experiment. Keep run outputs separate from shared caches, record
the effective cache paths and runtime versions, and let compiler cache keys handle
new kernels, shapes and configurations automatically. Use isolated cold caches only
for an explicitly requested cold-start measurement or a diagnosed cache problem.
Preserve these caches across Pod restarts and changes of physical host.

## Slurm

Use local Slurm GPU jobs for cluster experiments. Default to one `gpushort`
`rtx_pro_6000` GPU, one CPU, and 32 GB RAM unless the workload requires a
documented change. Redirect final logs to `logs/slurm/<experiment>/` and remove
the temporary bootstrap output after redirection. Batch related inference
conditions in one persistent vLLM process when possible.

Run Slurm control and submission commands outside the filesystem/network
sandbox. Restricted shells can block name resolution or controller sockets and
make a healthy `slurm1` appear down. Before diagnosing a controller outage,
repeat `scontrol ping` with sandbox escalation. Use ordinary `sbatch`/`salloc`
jobs to hold GPUs: advanced reservations created with `scontrol` require Slurm
administrator privileges.

## Remote infrastructure

No Lambda Cloud training target is currently reserved. The former
`gleipnir-improvement` instance was terminated on 2026-09-08 after explicit user
authorization and verified artifact collection.
Use `scripts/lambda_cloud.py` for SSH, sync, bootstrap, secret transfer, and
artifact collection on any separately authorized future target. Probe and record
its hardware before freezing a recipe. Never terminate an instance or launch
billable capacity without the user's explicit instruction. Pull important
artifacts before any termination.

Read the relevant platform section of [infrastructure.md](../infrastructure.md)
before remote setup, access, transfers or lifecycle operations. Use
`scripts/runpod_cloud.py` with a refreshed sanitized Pod snapshot for Runpod
SSH and sync; preserve network volumes and compiler/kernel caches.
