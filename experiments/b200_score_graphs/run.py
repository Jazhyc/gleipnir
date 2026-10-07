"""Replace the score server and screen bounded large prefill capture."""

import argparse
import asyncio
import json
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write
from experiments.b200_monitor_score.run import measure
from experiments.b200_score_graphs.canary import canary
from gleipnir.serving_prefill_graphs import validate_graph_config


def graph_command(command, settings, hashes):
    command = command.copy()
    if command[command.index("-m") + 1] != "experiments.b200_mutation_analysis.server":
        raise ValueError("graph trial requires the repaired score endpoint")
    if "--compilation-config" in command:
        raise ValueError("parent already overrides graph capture")
    command[command.index("--worker-cls") + 1] = (
        "experiments.b200_score_graphs.worker.GraphMonitorScoreAuditWorker"
    )
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    if not additional.get("qk_mutation_analysis") or "monitor_score" not in additional:
        raise ValueError("parent score/mutation contract missing")
    condition = additional["serving_condition"]
    condition.update({k: settings[k] for k in ("compilation_config", "prefill_graphs")})
    validate_graph_config(condition)
    additional["gleipnir_frost_fp4"].update(
        {
            p: h
            for p, h in hashes.items()
            if p.endswith("/worker.py") or p.endswith("/prefill_graph_worker.py")
        }
    )
    command[index] = json.dumps(additional, sort_keys=True)
    command += ["--compilation-config", json.dumps(settings["compilation_config"])]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    try:
        asyncio.run(
            measure(
                args.name,
                experiment=Path(__file__).parent,
                result_group="b200_score_graphs",
                command_prepare=graph_command,
                ready_prepare=canary,
            )
        )
    except BaseException as error:
        write(
            ROOT / "results/b200_score_graphs" / args.name / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
