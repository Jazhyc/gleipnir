# Persistent compiler caches across training campaigns

User-selected standing preference, 2026-10-03: reuse warm persistent compiler and
kernel caches across compatible training runs. Campaign names identify data,
outputs and logs; they do not justify a cold compiler-cache namespace.

Ordinary `monitoring_campaign_runtime.training_environment` calls now use
`.cache/training/shared`. On the preserved B200 volume, this is a relative alias
to the populated `student_injection_awareness` cache, preserving its original
resolved paths and artifacts. A fresh volume initializes a shared directory.
Inductor, Triton, TileLang and TVM cache paths all use this persistent root.
Explicit cold-start experiments can request `isolated_cache=True`.

The compiler still validates cache keys and graph guards. New graph shapes,
code, precision, hardware or compiler settings can cause cache misses and new
compilation; sharing the directory does not promise zero startup work. See
[PyTorch's cache configuration documentation](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html).
This choice does not change the selected training recipe or its numerical gates.
Preserve historical caches, record effective paths and versions, and never claim
cache hits solely because a directory exists.

The behavior-grounded campaign initially selected a new cache namespace despite
the existing warm volume. Its cold startup is retained in its log and execution
receipt. The user requested this correction before the first optimizer update.
The running worker already captured its environment; changing the shared helper
affects future launches, including its bounded evaluation reference process.

After the user reported low utilization, the initial attempt was intentionally
stopped after ten completed updates. Its logs, input audit and execution receipt
are retained under the campaign's `attempts/cold_cache/` trees. Missing entries
from that newly built cache were merged into the populated legacy cache without
overwriting existing entries. Training was restarted from the same initial
adapter and seed for a complete epoch, using the corrected shared-cache helper;
the original cache directories remain available for embedded artifact paths.
