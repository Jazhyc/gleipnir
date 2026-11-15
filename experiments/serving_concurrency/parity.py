"""Score the frozen ID and OOD workloads on one 4B serving step."""

from __future__ import annotations

import argparse
import subprocess
import sys

from experiments.serving_concurrency.run import OUT, PORT, ROOT, launch, prepare


def run(step: str, session: str) -> None:
    from gleipnir.data.monitoring import write_json
    from gleipnir.serving import bf16_launch

    name = f"parity_4b_{step}"
    out = OUT / session / name
    out.mkdir(parents=True, exist_ok=False)
    command, env = prepare("4b", step, out)
    process, startup, attempt = launch(command, env, name, out)
    try:
        validate = [sys.executable, "-m", "experiments.release_serving.validate"]
        validate += ["4b", "--port", str(PORT), "--output", str(out / "report.json")]
        validate += ["--rows-output", str(out)]
        code = subprocess.run(validate, cwd=ROOT, check=False).returncode
        write_json(
            out / "condition.json",
            {
                "step": step,
                "command": command,
                "startup_seconds": startup,
                "startup_attempt": attempt,
                "validate_exit": code,
            },
        )
    finally:
        bf16_launch.stop(process)
    print(f"parity_complete {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--session", default="", help="output subdirectory")
    args = parser.parse_args()
    run(args.step, args.session)


if __name__ == "__main__":
    main()
