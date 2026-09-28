"""Combined AWS lifecycle and remote-run command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import configure, runner

LIFECYCLE_COMMANDS = {"setup", "start", "pause", "destroy"}
RUN_COMMANDS = {"submit", "status", "logs", "artifacts", "download", "cancel"}


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="openalex-aws",
            description="Provision and run OpenAlex commands on AWS.",
        )
        parser.add_argument("command", nargs="?", choices=sorted(LIFECYCLE_COMMANDS | RUN_COMMANDS))
        parser.print_help()
        return 0
    if values[0] in LIFECYCLE_COMMANDS:
        return configure.main(values)
    if values[0] in RUN_COMMANDS:
        return runner.main(values)
    raise SystemExit(f"Unknown AWS command: {values[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
