"""Summarize frozen artifacts with the pinned training runtime on the Python path."""

from experiments.b200_fp4_full_training.campaign import write_comparison


def main() -> None:
    result = write_comparison()
    print(result["macro_differences"], flush=True)


if __name__ == "__main__":
    main()
