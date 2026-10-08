"""Run scheduler regressions and native parity before any model-serving launch."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("validation name must be a stem")
    root = Path(__file__).resolve().parents[2]
    out = root / "results/b200_vllm031" / args.name
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "checks": []}

    def save():
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for name, command in (
        (
            "scheduler",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "experiments/b200_vllm031/test_scheduler.py",
                f"--junitxml={out / 'scheduler.xml'}",
            ],
        ),
        (
            "native",
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.native",
                "--output",
                str(out / "native.json"),
            ],
        ),
    ):
        print("validation_start", name, flush=True)
        with (out / f"{name}.log").open("x") as log:
            status = subprocess.call(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
        check = {"name": name, "exit_code": status}
        if name == "scheduler" and status == 0:
            suite = ET.parse(out / "scheduler.xml").getroot().find("testsuite")
            check.update(
                tests=int(suite.get("tests")), skipped=int(suite.get("skipped"))
            )
            if check["tests"] != 3 or check["skipped"]:
                status = 1
                check["exit_code"] = status
        report["checks"].append(check)
        if status:
            report["status"] = "failed"
            save()
            raise SystemExit(status)
        save()
        print("validation_passed", name, flush=True)
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
