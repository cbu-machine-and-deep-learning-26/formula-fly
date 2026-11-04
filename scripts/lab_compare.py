"""Compare two lab folders' recipes and results (GH-103). Reads only; safe while training.

Run it with either folder's venv::

    .venv-flyvis\\Scripts\\python.exe scripts\\lab_compare.py D:\\formula-fly D:\\formula-fly-vale
    python scripts/lab_compare.py <base folder> <test folder> --run-prefix lab_20261104
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.analysis.lab_compare import format_comparison  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print the comparison; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="the folder running the recipe compared against")
    parser.add_argument("test", type=Path, help="the folder running the test recipe")
    parser.add_argument(
        "--run-prefix",
        default="lab_",
        help="which runs count; a date such as lab_20261104 picks one day's (default: lab_)",
    )
    args = parser.parse_args(argv)
    try:
        print(format_comparison(args.base, args.test, run_prefix=args.run_prefix))
    except FileNotFoundError as error:
        print(f"nothing to compare: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
