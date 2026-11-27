"""The optimized B200 monitor-serving runtime.

Promoted on 2026-11-27 from the experiments that built it (see
``docs/decisions/serving/b200_monitor_score_reference.md``). Entry modules:
``server`` (Gleipnir 4B lineage) and ``quant_server`` (Gleipnir 27B). Workers
stack one validated layer each, from ``workers.frost`` up to
``workers.prefix_cache``. ``promotion.json`` maps every recorded source of the
accepted recipes to its promoted path and hash; ``gleipnir.serving.sources``
applies it.
"""
