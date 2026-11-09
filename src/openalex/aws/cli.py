"""Combined AWS lifecycle and remote-run command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from openalex.artifacts import cli as artifact_cli

from . import configure, runner, snapshot_checkpoint, snapshot_import, website

LIFECYCLE_COMMANDS = {"setup", "start", "pause", "workers", "notifications", "destroy"}
RUN_COMMANDS = {"submit", "status", "logs", "artifacts", "download", "cancel"}
PIPELINE_COMMANDS = {"checkpoint-snapshot", "import-snapshot", "website"}
ARTIFACT_COMMANDS = {"hydrate", "inspect", "pin", "publish"}


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="openalex-aws",
            description="Provision and run OpenAlex commands on AWS.",
        )
        parser.add_argument(
            "command",
            nargs="?",
            choices=sorted(
                LIFECYCLE_COMMANDS
                | RUN_COMMANDS
                | ARTIFACT_COMMANDS
                | PIPELINE_COMMANDS
            ),
        )
        parser.print_help()
        return 0
    if values[0] in LIFECYCLE_COMMANDS:
        return configure.main(values)
    if values[0] in RUN_COMMANDS:
        return runner.main(values)
    if values[0] == "website":
        return website.main(values[1:])
    if values[0] == "import-snapshot":
        return snapshot_import.main(values[1:])
    if values[0] == "checkpoint-snapshot":
        return snapshot_checkpoint.main(values[1:])
    if values[0] in ARTIFACT_COMMANDS:
        return artifact_cli.main(values)
    raise SystemExit(f"Unknown AWS command: {values[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
