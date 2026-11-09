"""Create a durable S3 checkpoint for an in-progress snapshot import."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from . import configure, runner, snapshot_import
from .workers import effective_state


def _checkpoint_prefix(
    state: dict[str, Any],
    job_id: str,
    checkpoint_id: str,
) -> str:
    return (
        f"{snapshot_import._job_prefix(state, job_id)}/"
        f"checkpoints/{checkpoint_id}"
    )


def build_checkpoint_script(
    *,
    state: dict[str, Any],
    job_id: str,
    checkpoint_id: str,
    scratch_dir: str,
    quick_check: bool,
) -> str:
    scratch = scratch_dir.rstrip("/")
    job_dir = f"{scratch}/import-jobs/{job_id}"
    database = f"{job_dir}/articles.db"
    work_dir = f"{job_dir}/work"
    processed = f"{work_dir}/processed"
    snapshot = f"{job_dir}/checkpoint-{checkpoint_id}.db"
    size_file = f"{job_dir}/checkpoint-{checkpoint_id}.size"
    database_sha_file = f"{job_dir}/checkpoint-{checkpoint_id}.db.sha256"
    processed_sha_file = f"{job_dir}/checkpoint-{checkpoint_id}.processed.sha256"
    prefix_uri = runner.s3_uri(
        state["bucket"],
        _checkpoint_prefix(state, job_id, checkpoint_id),
    )
    database_key = (
        f"{_checkpoint_prefix(state, job_id, checkpoint_id)}/articles.db"
    )
    return f"""set -Eeuo pipefail
JOB_DIR={runner.q(job_dir)}
DATABASE={runner.q(database)}
WORK_DIR={runner.q(work_dir)}
PROCESSED={runner.q(processed)}
SNAPSHOT={runner.q(snapshot)}
SIZE_FILE={runner.q(size_file)}
DATABASE_SHA_FILE={runner.q(database_sha_file)}
PROCESSED_SHA_FILE={runner.q(processed_sha_file)}
PREFIX_URI={runner.q(prefix_uri)}
BUCKET={runner.q(state["bucket"])}
DATABASE_KEY={runner.q(database_key)}
QUICK_CHECK={runner.q("1" if quick_check else "0")}
mkdir -p "$WORK_DIR"
exec 9>"$JOB_DIR/import.lock"
if ! flock -n 9; then
  echo "A snapshot batch import is still active." >&2
  exit 75
fi
shopt -s nullglob
for pid_file in {runner.q(scratch)}/runs/*/pid; do
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Another OpenAlex run is active: $pid_file (pid $pid)" >&2
    exit 73
  fi
done
shopt -u nullglob
if [ ! -f "$DATABASE" ]; then
  echo "Missing in-progress database: $DATABASE" >&2
  exit 74
fi
rm -f "$SNAPSHOT"
touch "$PROCESSED"
python3 - "$DATABASE" "$SNAPSHOT" "$QUICK_CHECK" <<'PY'
import sqlite3
import sys

source_path, snapshot_path, quick_check = sys.argv[1:]
source = sqlite3.connect(f"file:{{source_path}}?mode=ro", uri=True, timeout=60)
destination = sqlite3.connect(snapshot_path)
try:
    source.backup(destination, pages=65536, sleep=0.05)
    if quick_check == "1":
        result = destination.execute("PRAGMA quick_check").fetchall()
        if result != [("ok",)]:
            raise SystemExit(f"SQLite backup quick_check failed: {{result[:10]}}")
finally:
    destination.close()
    source.close()
PY
stat -c '%s' "$SNAPSHOT" >"$SIZE_FILE"
sha256sum "$SNAPSHOT" | awk '{{print $1}}' >"$DATABASE_SHA_FILE"
sha256sum "$PROCESSED" | awk '{{print $1}}' >"$PROCESSED_SHA_FILE"
finish() {{
  code="$?"
  rm -f \
    "$SNAPSHOT" \
    "$SIZE_FILE" \
    "$DATABASE_SHA_FILE" \
    "$PROCESSED_SHA_FILE"
  exit "$code"
}}
trap finish EXIT
local_size="$(cat "$SIZE_FILE")"
local_sha256="$(cat "$DATABASE_SHA_FILE")"
remote_size="$(aws s3api head-object \
  --bucket "$BUCKET" \
  --key "$DATABASE_KEY" \
  --query ContentLength \
  --output text 2>/dev/null || true)"
remote_sha256="$(aws s3 cp \
  "$PREFIX_URI/articles.db.sha256" \
  - 2>/dev/null || true)"
if [ "$remote_size" = "$local_size" ] && [ "$remote_sha256" = "$local_sha256" ]; then
  echo "Reusing verified uploaded checkpoint database."
else
  aws s3 cp --no-progress "$SNAPSHOT" "$PREFIX_URI/articles.db"
  aws s3 cp --no-progress "$DATABASE_SHA_FILE" "$PREFIX_URI/articles.db.sha256"
fi
aws s3 cp --no-progress "$PROCESSED" "$PREFIX_URI/processed"
aws s3 cp --no-progress "$SIZE_FILE" "$PREFIX_URI/articles.db.size"
aws s3 cp --no-progress "$PROCESSED_SHA_FILE" "$PREFIX_URI/processed.sha256"
"""


def _checkpoint_keys(
    state: dict[str, Any],
    job_id: str,
    checkpoint_id: str,
) -> dict[str, str]:
    prefix = _checkpoint_prefix(state, job_id, checkpoint_id)
    return {
        "prefix": prefix,
        "database": f"{prefix}/articles.db",
        "processed": f"{prefix}/processed",
        "size": f"{prefix}/articles.db.size",
        "database_sha256": f"{prefix}/articles.db.sha256",
        "processed_sha256": f"{prefix}/processed.sha256",
        "manifest": f"{prefix}/manifest.json",
    }


def _checkpoint_ready(
    s3,
    state: dict[str, Any],
    keys: dict[str, str],
) -> tuple[bool, int | None]:
    if not all(
        snapshot_import._object_exists(s3, state["bucket"], keys[name])
        for name in (
            "database",
            "processed",
            "size",
            "database_sha256",
            "processed_sha256",
        )
    ):
        return False, None
    size_response = s3.get_object(Bucket=state["bucket"], Key=keys["size"])
    expected_size = int(size_response["Body"].read().decode("ascii").strip())
    database_head = s3.head_object(
        Bucket=state["bucket"],
        Key=keys["database"],
    )
    return int(database_head["ContentLength"]) == expected_size, expected_size


def _finalize_checkpoint(
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
    checkpoint: dict[str, Any],
    keep_previous: bool,
) -> None:
    job_id = str(job["job_id"])
    checkpoint_id = str(checkpoint["checkpoint_id"])
    keys = _checkpoint_keys(state, job_id, checkpoint_id)
    ready, size = _checkpoint_ready(s3, state, keys)
    if not ready or size is None:
        raise SystemExit("Checkpoint upload is incomplete or has the wrong size.")
    completed_files = sorted(str(value) for value in checkpoint["completed_files"])
    database_sha256 = (
        s3.get_object(Bucket=state["bucket"], Key=keys["database_sha256"])["Body"]
        .read()
        .decode("ascii")
        .strip()
    )
    processed_sha256 = (
        s3.get_object(Bucket=state["bucket"], Key=keys["processed_sha256"])["Body"]
        .read()
        .decode("ascii")
        .strip()
    )
    for name, digest in (
        ("database", database_sha256),
        ("processed", processed_sha256),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise SystemExit(f"Checkpoint {name} SHA-256 is invalid.")
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "checkpoint_id": checkpoint_id,
        "created_at": checkpoint["created_at"],
        "database_size": size,
        "database_sha256": database_sha256,
        "processed_sha256": processed_sha256,
        "database_s3_uri": runner.s3_uri(state["bucket"], keys["database"]),
        "processed_s3_uri": runner.s3_uri(state["bucket"], keys["processed"]),
        "completed_files": completed_files,
        "inventory_sha256": job["inventory_sha256"],
        "compile_args": job["compile_args"],
        "database_generation": job["database_generation"],
    }
    runner.put_s3_json(
        s3,
        state["bucket"],
        keys["manifest"],
        manifest,
    )
    previous = job.get("latest_checkpoint")
    job["latest_checkpoint"] = {
        **manifest,
        "manifest_s3_uri": runner.s3_uri(state["bucket"], keys["manifest"]),
        "prefix": keys["prefix"],
    }
    job["checkpoint_in_progress"] = None
    job["status"] = checkpoint.get("prior_status") or "running"
    job["message"] = (
        f"Durable checkpoint {checkpoint_id} uploaded "
        f"({size / 1024**3:.1f} GiB)"
    )
    snapshot_import._save_job(s3, state, job)

    if previous and not keep_previous:
        previous_prefix = previous.get("prefix")
        if previous_prefix and previous_prefix != keys["prefix"]:
            snapshot_import._delete_prefix(
                s3,
                state["bucket"],
                str(previous_prefix),
            )


def _wait_or_start_checkpoint(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = job.get("checkpoint_in_progress")
    if checkpoint is None:
        checkpoint = {
            "created_at": runner.utc_now(),
            "completed_files": list(job["completed_files"]),
            "prior_status": job.get("status"),
            "command_id": None,
        }
        job["checkpoint_in_progress"] = checkpoint
        job["status"] = "checkpointing"
        job["message"] = "Preparing durable checkpoint"
        snapshot_import._save_job(s3, state, job)

    cpu = effective_state(state, "cpu")
    command_id = checkpoint.get("command_id")
    checkpoint_id = checkpoint.get("checkpoint_id")
    if command_id and checkpoint_id:
        keys = _checkpoint_keys(
            state,
            str(job["job_id"]),
            str(checkpoint_id),
        )
        ready, _size = _checkpoint_ready(s3, state, keys)
        if ready:
            return checkpoint
        invocation = snapshot_import._wait_invocation(
            session.client("ssm"),
            str(command_id),
            cpu["instance_id"],
            args.command_timeout_seconds,
            args.poll_seconds,
        )
        if invocation.get("Status") in snapshot_import.SSM_SUCCESS_STATUSES:
            return checkpoint
        if not args.retry_failed:
            raise SystemExit(
                f"Checkpoint SSM command ended with {invocation.get('Status')}; "
                "inspect it and pass --retry-failed."
            )

    checkpoint_id = runner.make_run_id()
    checkpoint["checkpoint_id"] = checkpoint_id
    checkpoint["command_id"] = None
    job["message"] = f"Preparing durable checkpoint attempt {checkpoint_id}"
    snapshot_import._save_job(s3, state, job)
    script = build_checkpoint_script(
        state=state,
        job_id=str(job["job_id"]),
        checkpoint_id=checkpoint_id,
        scratch_dir=args.scratch_dir,
        quick_check=not args.skip_quick_check,
    )
    command_id = snapshot_import._send_ssm_script(
        session,
        cpu["instance_id"],
        script,
        f"Checkpoint OpenAlex snapshot import {job['job_id']}",
        args.command_timeout_seconds,
    )
    checkpoint["command_id"] = command_id
    snapshot_import._save_job(s3, state, job)
    invocation = snapshot_import._wait_invocation(
        session.client("ssm"),
        command_id,
        cpu["instance_id"],
        args.command_timeout_seconds,
        args.poll_seconds,
    )
    if invocation.get("Status") not in snapshot_import.SSM_SUCCESS_STATUSES:
        raise SystemExit(
            f"Checkpoint failed with SSM status {invocation.get('Status')}."
        )
    return checkpoint


def run_checkpoint(args: argparse.Namespace) -> int:
    state, session = snapshot_import._load_or_restore_project_state(args)
    s3 = session.client("s3")
    _boto3, client_error = runner.require_boto3()
    job = snapshot_import._load_job(
        s3,
        state,
        args.job_id,
        client_error,
    )
    cpu = effective_state(state, "cpu")
    if job.get("instance_id") != cpu.get("instance_id"):
        raise SystemExit("The job's CPU worker no longer matches configured state.")
    if job.get("db_s3_uri") != state.get("db_s3_uri"):
        raise SystemExit("The job's database URI no longer matches configured state.")
    live_state = runner.instance_state(session.client("ec2"), cpu["instance_id"])
    if live_state != "running":
        raise SystemExit(
            "The CPU worker is not running; its instance-store checkpoint source "
            "is unavailable."
        )
    print(
        f"Job {job['job_id']}: {len(job['completed_files']):,}/"
        f"{job['inventory_files']:,} files complete."
    )
    if not args.yes:
        raise SystemExit(
            "Checkpointing uploads the full SQLite database. Rerun with --yes "
            "after reviewing the job above."
        )

    lease = snapshot_import._acquire_lease(
        s3,
        state,
        str(job["job_id"]),
        args.steal_lease,
    )
    try:
        job = snapshot_import._load_job(
            s3,
            state,
            str(job["job_id"]),
            client_error,
        )
        runner.wait_until_ssm_online(
            session,
            cpu["instance_id"],
            min(args.command_timeout_seconds, 600),
        )
        snapshot_import._reconcile_importing_batch(
            args,
            session,
            s3,
            state,
            job,
        )
        in_progress = job.get("checkpoint_in_progress")
        if in_progress and sorted(in_progress["completed_files"]) != sorted(
            job["completed_files"]
        ):
            if in_progress.get("checkpoint_id"):
                stale_prefix = _checkpoint_prefix(
                    state,
                    str(job["job_id"]),
                    str(in_progress["checkpoint_id"]),
                )
                snapshot_import._delete_prefix(
                    s3,
                    state["bucket"],
                    stale_prefix,
                )
            job["checkpoint_in_progress"] = None
            job["status"] = "running"
            job["message"] = (
                "Discarded stale checkpoint attempt after reconciling "
                "additional completed batches"
            )
            snapshot_import._save_job(s3, state, job)
        while True:
            checkpoint = _wait_or_start_checkpoint(
                args,
                session,
                s3,
                state,
                job,
            )
            snapshot_import._reconcile_completion_markers(
                s3,
                state,
                job,
            )
            if sorted(checkpoint["completed_files"]) != sorted(
                job["completed_files"]
            ):
                snapshot_import._delete_prefix(
                    s3,
                    state["bucket"],
                    _checkpoint_prefix(
                        state,
                        str(job["job_id"]),
                        str(checkpoint["checkpoint_id"]),
                    ),
                )
                job["checkpoint_in_progress"] = None
                job["status"] = "running"
                job["message"] = (
                    "Discarded checkpoint whose batch boundary changed "
                    "during backup"
                )
                snapshot_import._save_job(s3, state, job)
                continue
            _finalize_checkpoint(
                s3,
                state,
                job,
                checkpoint,
                args.keep_previous,
            )
            break
    finally:
        snapshot_import._release_lease(
            s3,
            state,
            str(job["job_id"]),
            lease,
        )
    print(job["message"])
    print(f"Manifest: {job['latest_checkpoint']['manifest_s3_uri']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalex-aws checkpoint-snapshot",
        description=(
            "Back up an in-progress snapshot import's SQLite database and "
            "processed checkpoint to S3."
        ),
    )
    parser.add_argument("job_id")
    parser.add_argument("--state-path", default=runner.DEFAULT_STATE_PATH)
    parser.add_argument("--state-s3-uri")
    parser.add_argument("--state-bucket", default=configure.DEFAULT_BUCKET)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--scratch-dir", default=runner.DEFAULT_SCRATCH_DIR)
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=snapshot_import.DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=snapshot_import.DEFAULT_POLL_SECONDS,
    )
    parser.add_argument("--skip-quick-check", action="store_true")
    parser.add_argument("--keep-previous", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--steal-lease", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command_timeout_seconds < 60:
        raise SystemExit("--command-timeout-seconds must be at least 60.")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero.")
    return run_checkpoint(args)


if __name__ == "__main__":
    raise SystemExit(main())
