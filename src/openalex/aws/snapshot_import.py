"""Stream a cluster-local OpenAlex snapshot through S3 into one EC2 database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from . import configure, runner
from .workers import effective_state

JOB_SCHEMA_VERSION = 1
DEFAULT_BATCH_GIB = 10.0
DEFAULT_PREFETCH_BATCHES = 10
DEFAULT_COMMAND_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_POLL_SECONDS = 10.0
COPY_PART_SIZE = 512 * 1024 * 1024
SINGLE_COPY_LIMIT = 5 * 1024**3
SUPPORTED_SUFFIXES = (".gz", ".parquet")
SSM_PENDING_STATUSES = {"Pending", "InProgress", "Delayed"}
SSM_SUCCESS_STATUSES = {"Success"}
FORBIDDEN_COMPILE_OPTIONS = {
    "--database-url",
    "--delete-sources",
    "--enable-references",
    "--limit",
    "--path-pattern",
}


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    size: int
    mtime_ns: int = 0


@dataclass(frozen=True)
class SnapshotBatch:
    batch_id: str
    files: tuple[SnapshotFile, ...]
    size: int


@dataclass(frozen=True)
class ControllerLease:
    token: str
    etag: str


def _utc_now() -> str:
    return runner.utc_now()


def _job_prefix(state: dict[str, Any], job_id: str) -> str:
    return runner.prefixed_key(state["prefix"], f"import-jobs/{job_id}")


def _job_status_key(state: dict[str, Any], job_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/status.json"


def _batch_prefix(state: dict[str, Any], job_id: str, batch_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/staging/{batch_id}"


def _candidate_key(state: dict[str, Any], job_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/output/articles.db"


def _candidate_size_key(state: dict[str, Any], job_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/output/articles.db.size"


def _lease_key(state: dict[str, Any], job_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/controller-lease.json"


def _batch_log_prefix(state: dict[str, Any], job_id: str, batch_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/logs/{batch_id}"


def _batch_result_key(state: dict[str, Any], job_id: str, batch_id: str) -> str:
    return f"{_job_prefix(state, job_id)}/batch-results/{batch_id}.json"


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe snapshot path: {value!r}")
    return path.as_posix()


def inventory_snapshot(root: Path) -> list[SnapshotFile]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Snapshot root is not a directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.endswith(SUPPORTED_SUFFIXES):
            continue
        relative = _safe_relative_path(path.relative_to(root).as_posix())
        stat = path.stat()
        files.append(
            SnapshotFile(
                relative_path=relative,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    if not files:
        raise SystemExit(f"No supported .gz or .parquet files found under {root}")
    return files


def inventory_sha256(files: Sequence[SnapshotFile]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item.mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_processed_paths(
    processed_path: Path | None,
    snapshot_root: Path,
    inventory: Sequence[SnapshotFile],
) -> set[str]:
    if processed_path is None:
        return set()
    if not processed_path.is_file():
        raise SystemExit(f"Processed checkpoint not found: {processed_path}")

    root = snapshot_root.expanduser().resolve()
    known = {item.relative_path for item in inventory}
    normalized: set[str] = set()
    unresolved: list[str] = []
    ambiguous: list[str] = []
    for raw in processed_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        candidate: str | None = None
        path = Path(value).expanduser()
        if path.is_absolute():
            try:
                candidate = path.resolve().relative_to(root).as_posix()
            except ValueError:
                candidate = None
        else:
            direct = PurePosixPath(value).as_posix().lstrip("./")
            if direct in known:
                candidate = direct

        if candidate not in known:
            slash_value = value.replace("\\", "/").rstrip("/")
            matches = [
                relative
                for relative in known
                if slash_value == relative or slash_value.endswith(f"/{relative}")
            ]
            if len(matches) == 1:
                candidate = matches[0]
            elif len(matches) > 1:
                ambiguous.append(value)
                continue
            else:
                unresolved.append(value)
                continue
        normalized.add(str(candidate))

    if unresolved or ambiguous:
        details = []
        if unresolved:
            details.append(f"not in inventory: {', '.join(unresolved[:5])}")
        if ambiguous:
            details.append(f"ambiguous: {', '.join(ambiguous[:5])}")
        raise SystemExit(
            "The processed checkpoint does not match this snapshot ("
            + "; ".join(details)
            + ")."
        )
    return normalized


def build_batches(
    files: Sequence[SnapshotFile],
    completed: set[str],
    target_bytes: int,
) -> list[SnapshotBatch]:
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    groups: list[list[SnapshotFile]] = []
    current: list[SnapshotFile] = []
    current_size = 0
    for item in files:
        if item.relative_path in completed:
            continue
        if current and current_size + item.size > target_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(item)
        current_size += item.size
    if current:
        groups.append(current)

    batches = []
    for index, group in enumerate(groups, start=1):
        digest = inventory_sha256(group)[:12]
        batches.append(
            SnapshotBatch(
                batch_id=f"{index:05d}-{digest}",
                files=tuple(group),
                size=sum(item.size for item in group),
            )
        )
    return batches


def _batch_to_json(batch: SnapshotBatch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "size": batch.size,
        "status": "pending",
        "command_id": None,
        "files": [
            {
                "path": item.relative_path,
                "size": item.size,
                "mtime_ns": item.mtime_ns,
            }
            for item in batch.files
        ],
    }


def _batch_from_json(value: dict[str, Any]) -> SnapshotBatch:
    files = tuple(
        SnapshotFile(
            relative_path=str(item["path"]),
            size=int(item["size"]),
            mtime_ns=int(item.get("mtime_ns", 0)),
        )
        for item in value["files"]
    )
    return SnapshotBatch(
        batch_id=str(value["batch_id"]),
        files=files,
        size=int(value["size"]),
    )


def _strip_compile_separator(values: Sequence[str]) -> list[str]:
    result = list(values)
    if result and result[0] == "--":
        result.pop(0)
    for value in result:
        option = value.split("=", 1)[0]
        if any(
            managed == option
            or (
                option.startswith("--")
                and len(option) >= 4
                and managed.startswith(option)
            )
            for managed in FORBIDDEN_COMPILE_OPTIONS
        ):
            raise SystemExit(
                f"{option} is managed by import-snapshot and cannot be passed through."
            )
    return result


def _validate_compile_args(values: Sequence[str]) -> None:
    from openalex.db.compile_from_snapshot import parse_args

    try:
        parse_args(
            [
                "/snapshot",
                "--database-url",
                "sqlite:////managed/articles.db",
                *values,
            ]
        )
    except SystemExit as exc:
        if exc.code == 0:
            raise SystemExit(
                "Compile-snapshot help cannot be requested through import-snapshot."
            ) from exc
        raise


def maximum_prefetch_bytes(
    batches: Sequence[SnapshotBatch],
    prefetch_batches: int,
) -> int:
    maximum = 0
    for index in range(len(batches)):
        maximum = max(
            maximum,
            sum(
                batch.size
                for batch in batches[index : index + prefetch_batches]
            ),
        )
    return maximum


def _new_job_state(
    *,
    job_id: str,
    state: dict[str, Any],
    root: Path,
    inventory: Sequence[SnapshotFile],
    initial_completed: set[str],
    batches: Sequence[SnapshotBatch],
    compile_args: Sequence[str],
    batch_bytes: int,
    prefetch_batches: int,
    source_db_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "status": "created",
        "message": "Import job created",
        "created_at": now,
        "updated_at": now,
        "snapshot_root": str(root.expanduser().resolve()),
        "inventory_sha256": inventory_sha256(inventory),
        "inventory_files": len(inventory),
        "inventory_bytes": sum(item.size for item in inventory),
        "initial_completed_files": sorted(initial_completed),
        "completed_files": sorted(initial_completed),
        "compile_args": list(compile_args),
        "batch_bytes": batch_bytes,
        "prefetch_batches": prefetch_batches,
        "worker": "cpu",
        "instance_id": effective_state(state, "cpu")["instance_id"],
        "db_s3_uri": state.get("db_s3_uri"),
        "source_db_identity": source_db_identity,
        "database_generation": uuid.uuid4().hex,
        "batches": [_batch_to_json(batch) for batch in batches],
        "bootstrap_run_id": None,
        "final_command_id": None,
        "candidate_s3_uri": runner.s3_uri(
            state["bucket"], _candidate_key(state, job_id)
        ),
    }


def _save_job(s3, state: dict[str, Any], job: dict[str, Any]) -> None:
    job["updated_at"] = _utc_now()
    runner.put_s3_json(
        s3,
        state["bucket"],
        _job_status_key(state, str(job["job_id"])),
        job,
    )


def _create_job(s3, state: dict[str, Any], job: dict[str, Any]) -> None:
    job["updated_at"] = _utc_now()
    key = _job_status_key(state, str(job["job_id"]))
    try:
        s3.put_object(
            Bucket=state["bucket"],
            Key=key,
            Body=(json.dumps(job, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as exc:
        if runner.client_error_code(exc) in {"412", "PreconditionFailed"}:
            raise SystemExit(
                f"Snapshot import job already exists: {job['job_id']}; "
                "use --resume-job."
            ) from exc
        raise


def _acquire_lease(
    s3,
    state: dict[str, Any],
    job_id: str,
    steal: bool,
) -> ControllerLease:
    key = _lease_key(state, job_id)
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "acquired_at": _utc_now(),
    }
    if steal:
        try:
            current_head = s3.head_object(Bucket=state["bucket"], Key=key)
        except Exception as exc:
            if runner.client_error_code(exc) not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            try:
                current = s3.get_object(Bucket=state["bucket"], Key=key)
                prior = json.loads(current["Body"].read().decode("utf-8"))
            except Exception as exc:
                if runner.client_error_code(exc) in {
                    "404",
                    "NoSuchKey",
                    "NotFound",
                }:
                    prior = {}
                else:
                    raise
            print(
                "Stealing snapshot import controller lease from "
                f"{prior.get('host', 'unknown')} pid {prior.get('pid', 'unknown')}."
            )
            try:
                s3.delete_object(
                    Bucket=state["bucket"],
                    Key=key,
                    IfMatch=str(current_head["ETag"]),
                )
            except Exception as exc:
                if runner.client_error_code(exc) in {"412", "PreconditionFailed"}:
                    raise SystemExit(
                        "The controller lease changed during takeover; retry."
                    ) from exc
                raise
    try:
        response = s3.put_object(
            Bucket=state["bucket"],
            Key=key,
            Body=(json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as exc:
        if runner.client_error_code(exc) in {"412", "PreconditionFailed"}:
            raise SystemExit(
                f"Another controller holds the lease for job {job_id}. "
                "If it is definitely gone, resume with --steal-lease."
            ) from exc
        raise
    etag = response.get("ETag")
    if not etag:
        raise SystemExit("S3 did not return an ETag for the controller lease.")
    return ControllerLease(token=token, etag=str(etag))


def _release_lease(
    s3,
    state: dict[str, Any],
    job_id: str,
    lease: ControllerLease,
) -> None:
    key = _lease_key(state, job_id)
    try:
        s3.delete_object(
            Bucket=state["bucket"],
            Key=key,
            IfMatch=lease.etag,
        )
    except Exception as exc:
        if runner.client_error_code(exc) in {
            "404",
            "NoSuchKey",
            "NotFound",
            "412",
            "PreconditionFailed",
        }:
            return
        raise


def _object_identity(head: dict[str, Any]) -> dict[str, Any]:
    return {
        "size": int(head["ContentLength"]),
        "etag": str(head.get("ETag") or ""),
        "version_id": str(head.get("VersionId") or ""),
    }


def _conditional_s3_copy(
    s3,
    *,
    source_bucket: str,
    source_key: str,
    destination_bucket: str,
    destination_key: str,
    source_head: dict[str, Any],
    destination_etag: str | None = None,
    destination_must_not_exist: bool = False,
    metadata: dict[str, str] | None = None,
) -> None:
    source = {"Bucket": source_bucket, "Key": source_key}
    size = int(source_head["ContentLength"])
    source_etag = str(source_head["ETag"])
    destination_conditions: dict[str, str] = {}
    if destination_etag is not None:
        destination_conditions["IfMatch"] = destination_etag
    if destination_must_not_exist:
        destination_conditions["IfNoneMatch"] = "*"

    if size <= SINGLE_COPY_LIMIT:
        values: dict[str, Any] = {
            "Bucket": destination_bucket,
            "Key": destination_key,
            "CopySource": source,
            "CopySourceIfMatch": source_etag,
            **destination_conditions,
        }
        if metadata is not None:
            values.update(
                Metadata=metadata,
                MetadataDirective="REPLACE",
            )
        s3.copy_object(**values)
        return

    create_values: dict[str, Any] = {
        "Bucket": destination_bucket,
        "Key": destination_key,
    }
    if metadata is not None:
        create_values["Metadata"] = metadata
    upload = s3.create_multipart_upload(**create_values)
    upload_id = str(upload["UploadId"])
    parts = []
    try:
        for part_number, start in enumerate(range(0, size, COPY_PART_SIZE), start=1):
            end = min(start + COPY_PART_SIZE, size) - 1
            response = s3.upload_part_copy(
                Bucket=destination_bucket,
                Key=destination_key,
                UploadId=upload_id,
                PartNumber=part_number,
                CopySource=source,
                CopySourceRange=f"bytes={start}-{end}",
                CopySourceIfMatch=source_etag,
            )
            parts.append(
                {
                    "PartNumber": part_number,
                    "ETag": response["CopyPartResult"]["ETag"],
                }
            )
        s3.complete_multipart_upload(
            Bucket=destination_bucket,
            Key=destination_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
            **destination_conditions,
        )
    except BaseException:
        s3.abort_multipart_upload(
            Bucket=destination_bucket,
            Key=destination_key,
            UploadId=upload_id,
        )
        raise


def _load_job(s3, state: dict[str, Any], job_id: str, client_error) -> dict[str, Any]:
    job = runner.get_s3_json(
        s3,
        state["bucket"],
        _job_status_key(state, job_id),
        client_error,
    )
    if job is None:
        raise SystemExit(f"No snapshot import job found: {job_id}")
    if int(job.get("schema_version", 0)) != JOB_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported snapshot import job schema: {job.get('schema_version')}"
        )
    return job


def _discover_state_key(s3, bucket: str) -> str:
    response = s3.list_objects_v2(Bucket=bucket, Delimiter="/")
    candidates = [configure.STATE_S3_KEY]
    candidates.extend(
        f"{item['Prefix']}{configure.STATE_S3_KEY}"
        for item in response.get("CommonPrefixes", [])
    )
    candidates = [
        key for key in candidates if _object_exists(s3, bucket, key)
    ]
    if not candidates:
        raise SystemExit(
            f"No AWS runner state found in s3://{bucket}; pass --state-s3-uri."
        )
    if len(candidates) != 1:
        locations = ", ".join(f"s3://{bucket}/{key}" for key in candidates)
        raise SystemExit(
            f"Multiple AWS runner states found ({locations}); pass --state-s3-uri."
        )
    return candidates[0]


def _load_or_restore_project_state(args: argparse.Namespace):
    state_path = Path(args.state_path)
    if state_path.is_file() and not args.state_s3_uri:
        state = runner.load_state(str(state_path))
        return state, runner.boto3_session(args, state)

    session = runner.boto3_session(args)
    s3 = session.client("s3")
    if args.state_s3_uri:
        bucket, key = runner.parse_s3_uri(args.state_s3_uri)
    else:
        bucket = args.state_bucket
        key = _discover_state_key(s3, bucket)
    response = s3.get_object(Bucket=bucket, Key=key)
    raw_state = json.loads(response["Body"].read().decode("utf-8"))
    state = runner.normalize_state(raw_state)[0]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(state_path, state)
    print(f"Restored AWS runner state from {runner.s3_uri(bucket, key)}.")
    return state, runner.boto3_session(args, state)


def _wait_for_run(
    s3,
    state: dict[str, Any],
    run_id: str,
    client_error,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    key = f"{runner.run_prefix(state, run_id)}/status.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = runner.get_s3_json(s3, state["bucket"], key, client_error)
        value = None if status is None else status.get("status")
        if value in {"success", "failed", "cancelled"}:
            assert status is not None
            return status
        time.sleep(poll_seconds)
    raise SystemExit(f"Bootstrap run {run_id} timed out.")


def build_bootstrap_command(
    job_id: str,
    bootstrap_run_id: str,
    scratch_dir: str,
    allow_create: bool,
    checkpoint_database_uri: str | None = None,
    checkpoint_processed_uri: str | None = None,
    checkpoint_database_size: int | None = None,
    checkpoint_id: str | None = None,
    checkpoint_database_sha256: str | None = None,
    checkpoint_processed_sha256: str | None = None,
) -> str:
    scratch = scratch_dir.rstrip("/")
    job_dir = f"{scratch}/import-jobs/{job_id}"
    database = f"{job_dir}/articles.db"
    work_dir = f"{job_dir}/work"
    return f"""set -Eeuo pipefail
JOB_DIR={runner.q(job_dir)}
DATABASE={runner.q(database)}
WORK_DIR={runner.q(work_dir)}
BOOTSTRAP_RUN_ID={runner.q(bootstrap_run_id)}
ALLOW_CREATE={runner.q("1" if allow_create else "0")}
CHECKPOINT_DATABASE_URI={runner.q(checkpoint_database_uri or "")}
CHECKPOINT_PROCESSED_URI={runner.q(checkpoint_processed_uri or "")}
CHECKPOINT_DATABASE_SIZE={runner.q(
        "" if checkpoint_database_size is None else checkpoint_database_size
    )}
CHECKPOINT_ID={runner.q(checkpoint_id or "")}
CHECKPOINT_DATABASE_SHA256={runner.q(checkpoint_database_sha256 or "")}
CHECKPOINT_PROCESSED_SHA256={runner.q(checkpoint_processed_sha256 or "")}
RESTORE_MARKER="$JOB_DIR/restored-checkpoint"
shopt -s nullglob
for pid_file in {runner.q(scratch)}/runs/*/pid; do
  case "$pid_file" in
    */"$BOOTSTRAP_RUN_ID"/pid) continue ;;
  esac
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "Another OpenAlex run is active: $pid_file (pid $pid)" >&2
    exit 73
  fi
done
shopt -u nullglob
mkdir -p "$JOB_DIR" "$WORK_DIR"
exec 8>"$JOB_DIR/import.lock"
if ! flock -n 8; then
  echo "A delayed snapshot batch import is still active." >&2
  exit 75
fi
if [ -n "$CHECKPOINT_DATABASE_URI" ]; then
  if [ -f "$DATABASE" ] \
    && [ -f "$WORK_DIR/processed" ] \
    && [ "$(cat "$RESTORE_MARKER" 2>/dev/null || true)" = "$CHECKPOINT_ID" ]; then
    echo "Reusing restored checkpoint $CHECKPOINT_ID"
  elif [ "$ALLOW_CREATE" = "1" ]; then
    rm -f \
      "$DATABASE.download" \
      "$WORK_DIR/processed.download" \
      "$RESTORE_MARKER"
    aws s3 cp --no-progress "$CHECKPOINT_DATABASE_URI" "$DATABASE.download"
    aws s3 cp --no-progress \
      "$CHECKPOINT_PROCESSED_URI" \
      "$WORK_DIR/processed.download"
    if [ -n "$CHECKPOINT_DATABASE_SIZE" ] \
      && [ "$(stat -c '%s' "$DATABASE.download")" != "$CHECKPOINT_DATABASE_SIZE" ]; then
      echo "Downloaded checkpoint database has the wrong size." >&2
      exit 77
    fi
    printf '%s  %s\\n' \
      "$CHECKPOINT_DATABASE_SHA256" \
      "$DATABASE.download" | sha256sum -c -
    printf '%s  %s\\n' \
      "$CHECKPOINT_PROCESSED_SHA256" \
      "$WORK_DIR/processed.download" | sha256sum -c -
    chmod u+w "$DATABASE.download"
    mv "$DATABASE.download" "$DATABASE"
    mv "$WORK_DIR/processed.download" "$WORK_DIR/processed"
    printf '%s\\n' "$CHECKPOINT_ID" >"$RESTORE_MARKER.tmp"
    mv "$RESTORE_MARKER.tmp" "$RESTORE_MARKER"
    echo "Restored checkpoint $CHECKPOINT_ID"
  else
    echo "Checkpoint restore is required but creation was disabled." >&2
    exit 74
  fi
elif [ -f "$DATABASE" ]; then
  echo "Reusing snapshot import database at $DATABASE"
elif [ "$ALLOW_CREATE" = "1" ]; then
    cp --reflink=auto articles.db "$DATABASE"
    rm -f "$WORK_DIR/processed"
  chmod u+w "$DATABASE"
  printf '%s\\n' {runner.q(job_id)} >"$JOB_DIR/job-id"
  echo "Initialized snapshot import database at $DATABASE"
else
  echo "The in-progress EC2 database is missing; instance-store progress was lost." >&2
  exit 74
fi
"""


def _bootstrap_worker(
    args: argparse.Namespace,
    state: dict[str, Any],
    s3,
    client_error,
    job: dict[str, Any],
) -> None:
    initial = set(job["initial_completed_files"])
    completed = set(job["completed_files"])
    restore_checkpoint = job.get("restore_checkpoint")
    if restore_checkpoint is None:
        db_bucket, db_key = runner.parse_s3_uri(str(state["db_s3_uri"]))
        current_identity = _object_identity(
            s3.head_object(Bucket=db_bucket, Key=db_key)
        )
        expected_identity = job.get("source_db_identity")
        if expected_identity is not None and current_identity != expected_identity:
            raise SystemExit(
                "The configured S3 database changed after this import job was "
                "created; refusing to build on an unexpected source version."
            )
    allow_create = completed == initial or restore_checkpoint is not None
    run_id = str(job.get("bootstrap_run_id") or f"snapshot-bootstrap-{job['job_id']}")
    job["bootstrap_run_id"] = run_id
    job["status"] = "bootstrapping"
    job["message"] = "Starting CPU worker and preparing writable database"
    _save_job(s3, state, job)

    command = build_bootstrap_command(
        str(job["job_id"]),
        run_id,
        args.scratch_dir,
        allow_create=allow_create,
        checkpoint_database_uri=(
            None
            if restore_checkpoint is None
            else restore_checkpoint.get("database_s3_uri")
        ),
        checkpoint_processed_uri=(
            None
            if restore_checkpoint is None
            else restore_checkpoint.get("processed_s3_uri")
        ),
        checkpoint_database_size=(
            None
            if restore_checkpoint is None
            else int(restore_checkpoint["database_size"])
        ),
        checkpoint_id=(
            None
            if restore_checkpoint is None
            else str(restore_checkpoint["checkpoint_id"])
        ),
        checkpoint_database_sha256=(
            None
            if restore_checkpoint is None
            else str(restore_checkpoint["database_sha256"])
        ),
        checkpoint_processed_sha256=(
            None
            if restore_checkpoint is None
            else str(restore_checkpoint["processed_sha256"])
        ),
    )
    values = [
        "submit",
        "--worker",
        "cpu",
        "--run-id",
        run_id,
        "--state-path",
        args.state_path,
        "--last-run-path",
        args.last_run_path,
        "--scratch-dir",
        args.scratch_dir,
        "--branch",
        args.branch,
        "--no-auto-inputs",
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    if args.commit:
        values.extend(("--commit", args.commit))
    if restore_checkpoint is not None:
        values.append("--no-database")
    values.extend(("--", "bash", "-lc", command))
    runner.main(values)
    status = _wait_for_run(
        s3,
        state,
        run_id,
        client_error,
        args.poll_seconds,
        args.command_timeout_seconds,
    )
    if status.get("status") != "success":
        job["status"] = "failed"
        job["message"] = f"Bootstrap failed: {status.get('message', 'unknown error')}"
        _save_job(s3, state, job)
        raise SystemExit(job["message"])
    job["status"] = "running"
    if restore_checkpoint is not None:
        job["restore_checkpoint"] = None
        job["message"] = (
            f"Restored durable checkpoint {restore_checkpoint['checkpoint_id']}"
        )
    else:
        job["message"] = "CPU worker and writable database are ready"
    _save_job(s3, state, job)


def _ready_key(state: dict[str, Any], job_id: str, batch_id: str) -> str:
    return f"{_batch_prefix(state, job_id, batch_id)}/_READY.json"


def _object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if runner.client_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True


def _upload_batch(
    s3,
    state: dict[str, Any],
    job_id: str,
    root: Path,
    batch: SnapshotBatch,
) -> None:
    prefix = _batch_prefix(state, job_id, batch.batch_id)
    bucket = state["bucket"]
    for item in batch.files:
        source = root / PurePosixPath(item.relative_path)
        stat = source.stat()
        if stat.st_size != item.size or (
            item.mtime_ns and stat.st_mtime_ns != item.mtime_ns
        ):
            raise RuntimeError(
                f"Snapshot file changed after inventory: {source} "
                f"(expected size={item.size}, mtime_ns={item.mtime_ns}; "
                f"found size={stat.st_size}, mtime_ns={stat.st_mtime_ns})"
            )
        s3.upload_file(
            str(source),
            bucket,
            f"{prefix}/{item.relative_path}",
        )
    runner.put_s3_json(
        s3,
        bucket,
        _ready_key(state, job_id, batch.batch_id),
        {
            "batch_id": batch.batch_id,
            "size": batch.size,
            "files": [
                {
                    "path": item.relative_path,
                    "size": item.size,
                    "mtime_ns": item.mtime_ns,
                }
                for item in batch.files
            ],
        },
    )


def _delete_prefix(s3, bucket: str, prefix: str) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        for start in range(0, len(keys), 1000):
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": keys[start : start + 1000], "Quiet": True},
            )


def build_batch_import_script(
    *,
    state: dict[str, Any],
    job_id: str,
    batch: SnapshotBatch,
    scratch_dir: str,
    compile_args: Sequence[str],
    inventory_sha256: str,
    database_generation: str,
) -> str:
    scratch = scratch_dir.rstrip("/")
    job_dir = f"{scratch}/import-jobs/{job_id}"
    batch_dir = f"{job_dir}/batches/{batch.batch_id}"
    work_dir = f"{job_dir}/work"
    database = f"{job_dir}/articles.db"
    stdout_log = f"{job_dir}/logs/{batch.batch_id}.stdout.log"
    stderr_log = f"{job_dir}/logs/{batch.batch_id}.stderr.log"
    result_file = f"{job_dir}/logs/{batch.batch_id}.result.json"
    staging_uri = runner.s3_uri(
        state["bucket"], _batch_prefix(state, job_id, batch.batch_id)
    )
    log_uri = runner.s3_uri(
        state["bucket"], _batch_log_prefix(state, job_id, batch.batch_id)
    )
    result_uri = runner.s3_uri(
        state["bucket"], _batch_result_key(state, job_id, batch.batch_id)
    )
    command = shlex.join(
        [
            f"{scratch}/venv/bin/openalex",
            "compile-snapshot",
            batch_dir,
            "--database-url",
            f"sqlite:///{database}",
            *compile_args,
        ]
    )
    return f"""set -Eeuo pipefail
JOB_DIR={runner.q(job_dir)}
BATCH_DIR={runner.q(batch_dir)}
WORK_DIR={runner.q(work_dir)}
DATABASE={runner.q(database)}
STDOUT_LOG={runner.q(stdout_log)}
STDERR_LOG={runner.q(stderr_log)}
STAGING_URI={runner.q(staging_uri)}
LOG_URI={runner.q(log_uri)}
RESULT_FILE={runner.q(result_file)}
RESULT_URI={runner.q(result_uri)}
BATCH_ID={runner.q(batch.batch_id)}
INVENTORY_SHA256={runner.q(inventory_sha256)}
DATABASE_GENERATION={runner.q(database_generation)}
mkdir -p "$JOB_DIR/logs" "$WORK_DIR" "$BATCH_DIR"
exec 9>"$JOB_DIR/import.lock"
if ! flock -n 9; then
  echo "Another snapshot batch import holds $JOB_DIR/import.lock" >&2
  exit 75
fi
finish() {{
  code="$?"
  aws s3 cp --no-progress "$STDOUT_LOG" "$LOG_URI/stdout.log" >/dev/null 2>&1 || true
  aws s3 cp --no-progress "$STDERR_LOG" "$LOG_URI/stderr.log" >/dev/null 2>&1 || true
  exit "$code"
}}
trap finish EXIT
if [ ! -f "$DATABASE" ]; then
  echo "Missing in-progress database: $DATABASE" >&2
  exit 74
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
rm -rf "$BATCH_DIR"
mkdir -p "$BATCH_DIR"
aws s3 sync --no-progress "$STAGING_URI/" "$BATCH_DIR/" \
  >>"$STDOUT_LOG" 2>>"$STDERR_LOG"
python3 - "$BATCH_DIR" >>"$STDOUT_LOG" 2>>"$STDERR_LOG" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "_READY.json").read_text(encoding="utf-8"))
expected = set()
for item in manifest.get("files", []):
    relative = Path(str(item["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"Unsafe staged path: {{relative}}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise SystemExit(f"Staged path escapes batch root: {{relative}}")
    size = int(item["size"])
    if not path.is_file() or path.stat().st_size != size:
        raise SystemExit(f"Missing or truncated staged file: {{relative}}")
    expected.add(relative.as_posix())
actual = {{
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name.endswith((".gz", ".parquet"))
}}
if actual != expected:
    raise SystemExit(
        f"Staged snapshot file set differs from manifest: "
        f"missing={{sorted(expected - actual)}}, extra={{sorted(actual - expected)}}"
    )
print(f"Validated {{len(expected)}} staged snapshot files.")
PY
cd "$WORK_DIR"
{command} >>"$STDOUT_LOG" 2>>"$STDERR_LOG"
processed_sha256="$(sha256sum "$WORK_DIR/processed" | awk '{{print $1}}')"
python3 - \
  "$RESULT_FILE" \
  "$BATCH_ID" \
  "$INVENTORY_SHA256" \
  "$DATABASE_GENERATION" \
  "$processed_sha256" <<'PY'
import json
from pathlib import Path
import sys

path, batch_id, inventory_sha256, database_generation, processed_sha256 = sys.argv[1:]
Path(path).write_text(json.dumps({{
    "batch_id": batch_id,
    "inventory_sha256": inventory_sha256,
    "database_generation": database_generation,
    "processed_sha256": processed_sha256,
}}, sort_keys=True) + "\\n", encoding="utf-8")
PY
aws s3 cp --no-progress "$RESULT_FILE" "$RESULT_URI"
rm -rf "$BATCH_DIR"
"""


def _send_ssm_script(
    session,
    instance_id: str,
    script: str,
    comment: str,
    timeout_seconds: int,
) -> str:
    response = session.client("ssm").send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment=comment,
        Parameters={"commands": [script]},
        TimeoutSeconds=timeout_seconds,
    )
    return str(response["Command"]["CommandId"])


def _get_invocation(ssm, command_id: str, instance_id: str):
    return ssm.get_command_invocation(
        CommandId=command_id,
        InstanceId=instance_id,
    )


def _wait_invocation(
    ssm,
    command_id: str,
    instance_id: str,
    timeout_seconds: int,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    missing_since: float | None = None
    while time.monotonic() < deadline:
        try:
            invocation = _get_invocation(ssm, command_id, instance_id)
        except Exception as exc:
            if runner.client_error_code(exc) == "InvocationDoesNotExist":
                if missing_since is None:
                    missing_since = time.monotonic()
                elif time.monotonic() - missing_since >= min(30, timeout_seconds):
                    return {"Status": "InvocationDoesNotExist"}
                time.sleep(poll_seconds)
                continue
            raise
        missing_since = None
        if invocation.get("Status") not in SSM_PENDING_STATUSES:
            return invocation
        time.sleep(poll_seconds)
    if missing_since is not None:
        return {"Status": "InvocationDoesNotExist"}
    raise SystemExit(f"Timed out waiting for SSM command {command_id}.")


def _batch_record(job: dict[str, Any], batch_id: str) -> dict[str, Any]:
    for record in job["batches"]:
        if record["batch_id"] == batch_id:
            return record
    raise KeyError(batch_id)


def _finish_batch(
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
    batch: SnapshotBatch,
) -> None:
    record = _batch_record(job, batch.batch_id)
    completed = set(job["completed_files"])
    completed.update(item.relative_path for item in batch.files)
    job["completed_files"] = sorted(completed)
    record["status"] = "complete"
    record["completed_at"] = _utc_now()
    job["message"] = (
        f"Completed {len(completed):,}/{job['inventory_files']:,} snapshot files"
    )
    _save_job(s3, state, job)
    _delete_prefix(
        s3,
        state["bucket"],
        _batch_prefix(state, str(job["job_id"]), batch.batch_id),
    )
    print(
        f"Completed batch {batch.batch_id}; "
        f"{len(completed):,}/{job['inventory_files']:,} files complete."
    )


def _reconcile_completion_markers(
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> None:
    for record in job["batches"]:
        if record["status"] == "complete":
            continue
        key = _batch_result_key(
            state,
            str(job["job_id"]),
            str(record["batch_id"]),
        )
        try:
            response = s3.get_object(Bucket=state["bucket"], Key=key)
        except Exception as exc:
            if runner.client_error_code(exc) in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                continue
            raise
        marker = json.loads(response["Body"].read().decode("utf-8"))
        if marker.get("database_generation") != job["database_generation"]:
            continue
        if (
            marker.get("batch_id") != record["batch_id"]
            or marker.get("inventory_sha256") != job["inventory_sha256"]
        ):
            raise SystemExit(
                f"Invalid remote completion marker for batch {record['batch_id']}."
            )
        _finish_batch(s3, state, job, _batch_from_json(record))


def _reconcile_importing_batch(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> None:
    _reconcile_completion_markers(s3, state, job)
    instance_id = effective_state(state, "cpu")["instance_id"]
    ssm = session.client("ssm")
    importing = [item for item in job["batches"] if item["status"] == "importing"]
    if len(importing) > 1:
        raise SystemExit("Job state is invalid: multiple batches are marked importing.")
    if not importing:
        return
    record = importing[0]
    command_id = record.get("command_id")
    if not command_id:
        record["status"] = "failed"
        _save_job(s3, state, job)
        raise SystemExit(
            f"Batch {record['batch_id']} has no SSM command id; resume with "
            "--retry-failed after checking the worker."
        )
    try:
        invocation = _get_invocation(ssm, str(command_id), instance_id)
    except Exception as exc:
        if runner.client_error_code(exc) != "InvocationDoesNotExist":
            raise
        invocation = {"Status": "Unknown"}
    if invocation.get("Status") in SSM_PENDING_STATUSES:
        invocation = _wait_invocation(
            ssm,
            str(command_id),
            instance_id,
            args.command_timeout_seconds,
            args.poll_seconds,
        )
    if invocation.get("Status") in SSM_SUCCESS_STATUSES:
        _finish_batch(s3, state, job, _batch_from_json(record))
        return
    record["status"] = "failed"
    record["last_ssm_status"] = invocation.get("Status")
    job["status"] = "failed"
    job["message"] = (
        f"Batch {record['batch_id']} ended with SSM status "
        f"{invocation.get('Status', 'unknown')}"
    )
    _save_job(s3, state, job)
    if not args.retry_failed:
        raise SystemExit(job["message"] + "; inspect logs, then pass --retry-failed.")
    record["status"] = "pending"
    record["command_id"] = None
    job["status"] = "running"
    _save_job(s3, state, job)


def _remote_job_database_exists(
    args: argparse.Namespace,
    session,
    state: dict[str, Any],
    job_id: str,
) -> bool:
    cpu = effective_state(state, "cpu")
    runner.wait_until_ssm_online(
        session,
        cpu["instance_id"],
        min(args.command_timeout_seconds, 600),
    )
    database = (
        f"{args.scratch_dir.rstrip('/')}/import-jobs/{job_id}/articles.db"
    )
    command_id = _send_ssm_script(
        session,
        cpu["instance_id"],
        f"test -f {runner.q(database)} && echo present",
        f"Check OpenAlex snapshot import database {job_id}",
        120,
    )
    invocation = _wait_invocation(
        session.client("ssm"),
        command_id,
        cpu["instance_id"],
        120,
        args.poll_seconds,
    )
    return (
        invocation.get("Status") in SSM_SUCCESS_STATUSES
        and "present" in str(invocation.get("StandardOutputContent") or "")
    )


def _prepare_checkpoint_restore(
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> bool:
    checkpoint = job.get("latest_checkpoint")
    if not checkpoint:
        return False
    if checkpoint.get("inventory_sha256") != job.get("inventory_sha256"):
        raise SystemExit("Latest checkpoint belongs to a different snapshot inventory.")
    if list(checkpoint.get("compile_args", [])) != list(job.get("compile_args", [])):
        raise SystemExit("Latest checkpoint used different compile arguments.")
    for field in ("database_sha256", "processed_sha256"):
        digest = str(checkpoint.get(field) or "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise SystemExit(f"Latest checkpoint has an invalid {field}.")

    database_bucket, database_key = runner.parse_s3_uri(
        str(checkpoint["database_s3_uri"])
    )
    processed_bucket, processed_key = runner.parse_s3_uri(
        str(checkpoint["processed_s3_uri"])
    )
    if database_bucket != state["bucket"] or processed_bucket != state["bucket"]:
        raise SystemExit("Checkpoint objects are outside the configured S3 bucket.")
    database_head = s3.head_object(Bucket=database_bucket, Key=database_key)
    expected_size = int(checkpoint["database_size"])
    if int(database_head["ContentLength"]) != expected_size:
        raise SystemExit("Latest checkpoint database has the wrong size.")
    if not _object_exists(s3, processed_bucket, processed_key):
        raise SystemExit("Latest checkpoint processed file is missing.")

    inventory_files = {
        str(item["path"])
        for record in job["batches"]
        for item in record["files"]
    } | set(job["initial_completed_files"])
    checkpoint_completed = {
        str(value) for value in checkpoint["completed_files"]
    }
    if not checkpoint_completed <= inventory_files:
        raise SystemExit("Latest checkpoint contains files outside the job inventory.")

    for record in job["batches"]:
        batch_files = {str(item["path"]) for item in record["files"]}
        overlap = batch_files & checkpoint_completed
        if overlap and overlap != batch_files:
            raise SystemExit("Latest checkpoint stops in the middle of a job batch.")
        record["status"] = "complete" if overlap else "pending"
        record["command_id"] = None
    job["completed_files"] = sorted(checkpoint_completed)
    job["final_command_id"] = None
    job["checkpoint_in_progress"] = None
    job["restore_checkpoint"] = checkpoint
    job["database_generation"] = uuid.uuid4().hex
    job["status"] = "restoring"
    job["message"] = (
        f"Restoring durable checkpoint {checkpoint['checkpoint_id']} after "
        "instance-store loss"
    )
    _save_job(s3, state, job)
    return True


def _prepare_nvme_resume(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> None:
    cpu = effective_state(state, "cpu")
    live_state = runner.instance_state(session.client("ec2"), cpu["instance_id"])
    database_exists = False
    if live_state == "running":
        database_exists = _remote_job_database_exists(
            args,
            session,
            state,
            str(job["job_id"]),
        )
    if database_exists:
        _reconcile_importing_batch(args, session, s3, state, job)
        return

    if _prepare_checkpoint_restore(s3, state, job):
        return

    initial = set(job["initial_completed_files"])
    completed = set(job["completed_files"])
    if completed != initial:
        raise SystemExit(
            "The in-progress EC2 database is absent and completed batches exist "
            "only on instance-store NVMe. Refusing to continue from an "
            "incomplete S3 database."
        )

    reset = False
    for record in job["batches"]:
        if record["status"] == "failed" and not args.retry_failed:
            raise SystemExit(
                f"Batch {record['batch_id']} previously failed; inspect its logs "
                "and pass --retry-failed to retry."
            )
        if record["status"] == "importing" or (
            record["status"] == "failed" and args.retry_failed
        ):
            record["status"] = "pending"
            record["command_id"] = None
            reset = True
    if job.get("final_command_id"):
        job["final_command_id"] = None
        reset = True
    if reset:
        job["status"] = "created"
        job["message"] = (
            "Reset non-durable first-batch state after the EC2 database was absent"
        )
        _save_job(s3, state, job)


def _import_batch(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
    batch: SnapshotBatch,
) -> None:
    record = _batch_record(job, batch.batch_id)
    script = build_batch_import_script(
        state=state,
        job_id=str(job["job_id"]),
        batch=batch,
        scratch_dir=args.scratch_dir,
        compile_args=job["compile_args"],
        inventory_sha256=str(job["inventory_sha256"]),
        database_generation=str(job["database_generation"]),
    )
    instance_id = effective_state(state, "cpu")["instance_id"]
    command_id = _send_ssm_script(
        session,
        instance_id,
        script,
        f"OpenAlex snapshot {job['job_id']} batch {batch.batch_id}",
        args.command_timeout_seconds,
    )
    record["status"] = "importing"
    record["command_id"] = command_id
    record["started_at"] = _utc_now()
    job["status"] = "running"
    job["message"] = f"Importing batch {batch.batch_id}"
    _save_job(s3, state, job)
    print(
        f"Importing batch {batch.batch_id} on EC2 via SSM ({command_id}).\n"
        f"Logs: {runner.s3_uri(state['bucket'], _batch_log_prefix(state, str(job['job_id']), batch.batch_id))}/"
    )
    invocation = _wait_invocation(
        session.client("ssm"),
        command_id,
        instance_id,
        args.command_timeout_seconds,
        args.poll_seconds,
    )
    if invocation.get("Status") not in SSM_SUCCESS_STATUSES:
        record["status"] = "failed"
        record["last_ssm_status"] = invocation.get("Status")
        job["status"] = "failed"
        job["message"] = (
            f"Batch {batch.batch_id} failed with SSM status "
            f"{invocation.get('Status', 'unknown')}"
        )
        _save_job(s3, state, job)
        stderr = (invocation.get("StandardErrorContent") or "").strip()
        raise SystemExit(f"{job['message']}\n{stderr}".rstrip())
    _finish_batch(s3, state, job, batch)


def _run_batches(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
    snapshot_root: Path,
) -> None:
    _reconcile_importing_batch(args, session, s3, state, job)
    pending_records = [
        item for item in job["batches"] if item["status"] != "complete"
    ]
    if not pending_records:
        return
    for record in pending_records:
        if record["status"] == "failed":
            if not args.retry_failed:
                raise SystemExit(
                    f"Batch {record['batch_id']} previously failed; inspect its logs "
                    "and pass --retry-failed to retry."
                )
            record["status"] = "pending"
            record["command_id"] = None
    _save_job(s3, state, job)

    root = snapshot_root.expanduser().resolve()
    futures: dict[str, Future[None]] = {}
    batches = [_batch_from_json(record) for record in pending_records]
    next_upload = 0
    with ThreadPoolExecutor(max_workers=args.prefetch_batches) as executor:
        while next_upload < min(args.prefetch_batches, len(batches)):
            batch = batches[next_upload]
            if not _object_exists(
                s3,
                state["bucket"],
                _ready_key(state, str(job["job_id"]), batch.batch_id),
            ):
                futures[batch.batch_id] = executor.submit(
                    _upload_batch,
                    s3,
                    state,
                    str(job["job_id"]),
                    root,
                    batch,
                )
            next_upload += 1

        for batch in batches:
            future = futures.pop(batch.batch_id, None)
            if future is not None:
                print(f"Waiting for staged batch {batch.batch_id}.")
                future.result()
            record = _batch_record(job, batch.batch_id)
            record["status"] = "staged"
            record["staged_at"] = _utc_now()
            _save_job(s3, state, job)
            _import_batch(args, session, s3, state, job, batch)

            if next_upload < len(batches):
                next_batch = batches[next_upload]
                if not _object_exists(
                    s3,
                    state["bucket"],
                    _ready_key(
                        state,
                        str(job["job_id"]),
                        next_batch.batch_id,
                    ),
                ):
                    futures[next_batch.batch_id] = executor.submit(
                        _upload_batch,
                        s3,
                        state,
                        str(job["job_id"]),
                        root,
                        next_batch,
                    )
                next_upload += 1


def build_finalize_script(
    *,
    state: dict[str, Any],
    job_id: str,
    scratch_dir: str,
) -> str:
    scratch = scratch_dir.rstrip("/")
    job_dir = f"{scratch}/import-jobs/{job_id}"
    database = f"{job_dir}/articles.db"
    candidate_uri = runner.s3_uri(state["bucket"], _candidate_key(state, job_id))
    size_uri = runner.s3_uri(state["bucket"], _candidate_size_key(state, job_id))
    return f"""set -Eeuo pipefail
JOB_DIR={runner.q(job_dir)}
DATABASE={runner.q(database)}
CANDIDATE_URI={runner.q(candidate_uri)}
SIZE_URI={runner.q(size_uri)}
exec 9>"$JOB_DIR/import.lock"
if ! flock -n 9; then
  echo "Another snapshot import process is active." >&2
  exit 75
fi
if [ ! -f "$DATABASE" ]; then
  echo "Missing completed database: $DATABASE" >&2
  exit 74
fi
check="$(sqlite3 "$DATABASE" 'PRAGMA quick_check;')"
if [ "$check" != "ok" ]; then
  echo "SQLite quick_check failed: $check" >&2
  exit 76
fi
stat -c '%s' "$DATABASE" >"$JOB_DIR/articles.db.size"
aws s3 cp --no-progress "$DATABASE" "$CANDIDATE_URI"
aws s3 cp --no-progress "$JOB_DIR/articles.db.size" "$SIZE_URI"
"""


def _finalize_database(
    args: argparse.Namespace,
    session,
    s3,
    state: dict[str, Any],
    job: dict[str, Any],
) -> None:
    job_id = str(job["job_id"])
    candidate_key = _candidate_key(state, job_id)
    size_key = _candidate_size_key(state, job_id)
    instance_id = effective_state(state, "cpu")["instance_id"]
    candidate_ready = _object_exists(s3, state["bucket"], candidate_key) and _object_exists(
        s3, state["bucket"], size_key
    )
    if not candidate_ready:
        command_id = job.get("final_command_id")
        if command_id:
            invocation = _wait_invocation(
                session.client("ssm"),
                str(command_id),
                instance_id,
                args.command_timeout_seconds,
                args.poll_seconds,
            )
            if invocation.get("Status") not in SSM_SUCCESS_STATUSES:
                if args.retry_failed:
                    job["final_command_id"] = None
                    _save_job(s3, state, job)
                    return _finalize_database(args, session, s3, state, job)
                job["status"] = "failed"
                job["message"] = (
                    "Database validation/upload failed with SSM status "
                    f"{invocation.get('Status', 'unknown')}"
                )
                _save_job(s3, state, job)
                raise SystemExit(job["message"])
        else:
            script = build_finalize_script(
                state=state,
                job_id=job_id,
                scratch_dir=args.scratch_dir,
            )
            command_id = _send_ssm_script(
                session,
                instance_id,
                script,
                f"Finalize OpenAlex snapshot import {job_id}",
                args.command_timeout_seconds,
            )
            job["final_command_id"] = command_id
            job["status"] = "validating"
            job["message"] = "Running SQLite quick_check and uploading candidate database"
            _save_job(s3, state, job)
            invocation = _wait_invocation(
                session.client("ssm"),
                command_id,
                instance_id,
                args.command_timeout_seconds,
                args.poll_seconds,
            )
            if invocation.get("Status") not in SSM_SUCCESS_STATUSES:
                job["status"] = "failed"
                job["message"] = (
                    "Database validation/upload failed with SSM status "
                    f"{invocation.get('Status', 'unknown')}"
                )
                _save_job(s3, state, job)
                raise SystemExit(job["message"])

    size_response = s3.get_object(Bucket=state["bucket"], Key=size_key)
    expected_size = int(size_response["Body"].read().decode("ascii").strip())
    candidate_head = s3.head_object(Bucket=state["bucket"], Key=candidate_key)
    if int(candidate_head["ContentLength"]) != expected_size:
        raise SystemExit(
            "Uploaded candidate database size does not match the EC2 database "
            f"({candidate_head['ContentLength']} != {expected_size})."
        )
    job["candidate_size"] = expected_size

    if args.no_promote:
        _delete_prefix(
            s3,
            state["bucket"],
            f"{_job_prefix(state, job_id)}/staging",
        )
        job["status"] = "candidate_ready"
        job["message"] = "Validated database candidate is ready; promotion was disabled"
        _save_job(s3, state, job)
        return

    db_uri = state.get("db_s3_uri")
    if not db_uri:
        raise SystemExit("Configured state has no db_s3_uri to promote.")
    db_bucket, db_key = runner.parse_s3_uri(str(db_uri))
    if db_bucket != state["bucket"]:
        raise SystemExit(
            "Cross-bucket database promotion is not supported by import-snapshot."
        )
    canonical_head = s3.head_object(Bucket=db_bucket, Key=db_key)
    canonical_metadata = canonical_head.get("Metadata", {})
    if canonical_metadata.get("openalex-import-job") != job_id:
        source_identity = job.get("source_db_identity")
        if source_identity is not None and _object_identity(canonical_head) != source_identity:
            raise SystemExit(
                "The configured S3 database changed during the snapshot import; "
                "the validated candidate was preserved but not promoted."
            )
        backup_key = f"{db_key}.before-{job_id}"
        if not _object_exists(s3, db_bucket, backup_key):
            print(f"Backing up current database to {runner.s3_uri(db_bucket, backup_key)}.")
            _conditional_s3_copy(
                s3,
                source_bucket=db_bucket,
                source_key=db_key,
                destination_bucket=db_bucket,
                destination_key=backup_key,
                source_head=canonical_head,
                destination_must_not_exist=True,
            )
        print(f"Promoting completed database to {db_uri}.")
        _conditional_s3_copy(
            s3,
            source_bucket=state["bucket"],
            source_key=candidate_key,
            destination_bucket=db_bucket,
            destination_key=db_key,
            source_head=candidate_head,
            destination_etag=str(canonical_head["ETag"]),
            metadata={
                "openalex-import-job": job_id,
                "validated-size": str(expected_size),
            },
        )
        job["backup_s3_uri"] = runner.s3_uri(db_bucket, backup_key)

    promoted = s3.head_object(Bucket=db_bucket, Key=db_key)
    if (
        int(promoted["ContentLength"]) != expected_size
        or promoted.get("Metadata", {}).get("openalex-import-job") != job_id
    ):
        raise SystemExit("Promoted database failed size/metadata verification.")
    job["status"] = "success"
    job["message"] = "Snapshot import completed and database promoted"
    job["completed_at"] = _utc_now()
    job["promoted_s3_uri"] = db_uri
    _save_job(s3, state, job)
    s3.delete_object(Bucket=state["bucket"], Key=candidate_key)
    s3.delete_object(Bucket=state["bucket"], Key=size_key)
    _delete_prefix(
        s3,
        state["bucket"],
        f"{_job_prefix(state, job_id)}/staging",
    )


def _pause_worker(args: argparse.Namespace) -> None:
    values = [
        "pause",
        "--worker",
        "cpu",
        "--state-path",
        args.state_path,
        "--wait",
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    configure.main(values)


def _validate_resume(
    job: dict[str, Any],
    state: dict[str, Any],
    inventory: Sequence[SnapshotFile],
    compile_args: Sequence[str],
    batch_bytes: int,
    prefetch_batches: int,
) -> None:
    if job.get("inventory_sha256") != inventory_sha256(inventory):
        raise SystemExit("Snapshot inventory changed since this job was created.")
    if list(job.get("compile_args", [])) != list(compile_args):
        raise SystemExit("Compile-snapshot arguments changed since this job was created.")
    if int(job.get("batch_bytes", 0)) != batch_bytes:
        raise SystemExit("Batch size changed since this job was created.")
    if int(job.get("prefetch_batches", 0)) != prefetch_batches:
        raise SystemExit("Prefetch window changed since this job was created.")
    if job.get("db_s3_uri") != state.get("db_s3_uri"):
        raise SystemExit("Configured database URI changed since this job was created.")
    current_cpu = effective_state(state, "cpu")
    if job.get("instance_id") != current_cpu.get("instance_id"):
        raise SystemExit("Configured CPU worker changed since this job was created.")


def run_import(args: argparse.Namespace) -> int:
    compile_args = _strip_compile_separator(args.compile_args)
    _validate_compile_args(compile_args)
    snapshot_root = Path(args.snapshot_root)
    inventory = inventory_snapshot(snapshot_root)
    batch_bytes = int(args.batch_gib * 1024**3)
    processed = (
        set()
        if args.resume_job
        else normalize_processed_paths(
            Path(args.processed_file) if args.processed_file else None,
            snapshot_root,
            inventory,
        )
    )
    batches = (
        [] if args.resume_job else build_batches(inventory, processed, batch_bytes)
    )
    total_bytes = sum(item.size for item in inventory)
    print(
        f"Snapshot inventory: {len(inventory):,} files, "
        f"{total_bytes / 1024**3:.1f} GiB"
    )
    if not args.resume_job:
        remaining_bytes = sum(batch.size for batch in batches)
        print(
            f"Already processed: {len(processed):,} files; "
            f"remaining: {len(batches):,} batches, "
            f"{remaining_bytes / 1024**3:.1f} GiB"
        )
        print(
            f"Prefetch window: {args.prefetch_batches} batches "
            f"(actual maximum "
            f"{maximum_prefetch_bytes(batches, args.prefetch_batches) / 1024**3:.1f} "
            "GiB staged)"
        )
    if args.dry_run:
        return 0
    if not args.resume_job and not args.yes:
        raise SystemExit(
            "Refusing to start the cost-bearing AWS import without --yes. "
            "Review the inventory above, then rerun with --yes."
        )

    project_state, session = _load_or_restore_project_state(args)
    if not project_state.get("db_s3_uri"):
        raise SystemExit("AWS runner state has no configured input database.")
    s3 = session.client("s3")
    _boto3, client_error = runner.require_boto3()

    if args.resume_job:
        job = _load_job(s3, project_state, args.resume_job, client_error)
        if not args.compile_args_explicit:
            compile_args = list(job["compile_args"])
            _validate_compile_args(compile_args)
        if not args.batch_gib_explicit:
            batch_bytes = int(job["batch_bytes"])
            args.batch_gib = batch_bytes / 1024**3
        if not args.prefetch_batches_explicit:
            args.prefetch_batches = int(job["prefetch_batches"])
        _validate_resume(
            job,
            project_state,
            inventory,
            compile_args,
            batch_bytes,
            args.prefetch_batches,
        )
        remaining_records = [
            record for record in job["batches"] if record["status"] != "complete"
        ]
        remaining_bytes = sum(int(record["size"]) for record in remaining_records)
        pending_batches = [_batch_from_json(record) for record in remaining_records]
        print(
            f"Durable job progress: {len(job['completed_files']):,}/"
            f"{job['inventory_files']:,} files complete; "
            f"{len(remaining_records):,} batches and "
            f"{remaining_bytes / 1024**3:.1f} GiB remain."
        )
        print(
            f"Pinned prefetch window: {args.prefetch_batches} batches "
            f"(actual remaining maximum "
            f"{maximum_prefetch_bytes(pending_batches, args.prefetch_batches) / 1024**3:.1f} "
            "GiB staged)"
        )
        if not args.yes:
            raise SystemExit(
                "Refusing to resume the cost-bearing AWS import without --yes. "
                "Review the durable progress above, then rerun with --yes."
            )
    else:
        job_id = args.job_id or f"snapshot-{runner.make_run_id()}"
        db_bucket, db_key = runner.parse_s3_uri(
            str(project_state["db_s3_uri"])
        )
        job = _new_job_state(
            job_id=job_id,
            state=project_state,
            root=snapshot_root,
            inventory=inventory,
            initial_completed=processed,
            batches=batches,
            compile_args=compile_args,
            batch_bytes=batch_bytes,
            prefetch_batches=args.prefetch_batches,
            source_db_identity=_object_identity(
                s3.head_object(Bucket=db_bucket, Key=db_key)
            ),
        )
        _create_job(s3, project_state, job)
        print(f"Created snapshot import job {job_id}.")

    if job.get("status") == "success":
        _delete_prefix(
            s3,
            project_state["bucket"],
            f"{_job_prefix(project_state, str(job['job_id']))}/staging",
        )
        print(f"Snapshot import job {job['job_id']} is already complete.")
        return 0

    lease = _acquire_lease(
        s3,
        project_state,
        str(job["job_id"]),
        args.steal_lease,
    )
    try:
        job = _load_job(
            s3,
            project_state,
            str(job["job_id"]),
            client_error,
        )
        _validate_resume(
            job,
            project_state,
            inventory,
            compile_args,
            batch_bytes,
            args.prefetch_batches,
        )
        if job.get("checkpoint_in_progress"):
            checkpoint_id = job["checkpoint_in_progress"].get("checkpoint_id")
            raise SystemExit(
                f"Checkpoint {checkpoint_id} is incomplete. Finish it with "
                "checkpoint-snapshot before resuming imports."
            )
        candidate_ready = _object_exists(
            s3,
            project_state["bucket"],
            _candidate_key(project_state, str(job["job_id"])),
        ) and _object_exists(
            s3,
            project_state["bucket"],
            _candidate_size_key(project_state, str(job["job_id"])),
        )
        all_files_complete = len(job["completed_files"]) == int(
            job["inventory_files"]
        )
        if not (all_files_complete and candidate_ready):
            _prepare_nvme_resume(
                args,
                session,
                s3,
                project_state,
                job,
            )
            _bootstrap_worker(args, project_state, s3, client_error, job)
            _run_batches(
                args,
                session,
                s3,
                project_state,
                job,
                snapshot_root,
            )
        _finalize_database(args, session, s3, project_state, job)
    finally:
        _release_lease(
            s3,
            project_state,
            str(job["job_id"]),
            lease,
        )

    if not args.keep_worker_running and job.get("status") in {
        "success",
        "candidate_ready",
    }:
        try:
            _pause_worker(args)
        except (Exception, SystemExit) as exc:
            print(f"Database is safe in S3, but the CPU worker could not be paused: {exc}")
    print(job["message"])
    print(
        "Job status: "
        + runner.s3_uri(
            project_state["bucket"],
            _job_status_key(project_state, str(job["job_id"])),
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalex-aws import-snapshot",
        description=(
            "Relay a cluster-local OpenAlex snapshot through bounded transient "
            "S3 batches into the CPU worker's SQLite database."
        ),
    )
    parser.add_argument("snapshot_root", help="Cluster-local snapshot directory.")
    parser.add_argument("--processed-file", help="Original importer's processed checkpoint.")
    parser.add_argument("--job-id", help="Explicit id for a new import job.")
    parser.add_argument("--resume-job", help="Resume an existing durable import job.")
    parser.add_argument("--batch-gib", type=float, default=DEFAULT_BATCH_GIB)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=DEFAULT_PREFETCH_BATCHES,
        help="Concurrent staged batch limit; SQLite imports remain sequential.",
    )
    parser.add_argument("--state-path", default=runner.DEFAULT_STATE_PATH)
    parser.add_argument("--last-run-path", default=runner.DEFAULT_LAST_RUN_PATH)
    parser.add_argument("--state-s3-uri")
    parser.add_argument("--state-bucket", default=configure.DEFAULT_BUCKET)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--branch", default=runner.DEFAULT_BRANCH)
    parser.add_argument("--commit")
    parser.add_argument("--scratch-dir", default=runner.DEFAULT_SCRATCH_DIR)
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--steal-lease",
        action="store_true",
        help="Take over a job only after confirming its prior controller is gone.",
    )
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--keep-worker-running", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm the cost-bearing AWS import.",
    )
    parser.set_defaults(
        compile_args=[],
        compile_args_explicit=False,
        batch_gib_explicit=False,
        prefetch_batches_explicit=False,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    compile_args: list[str] = []
    if "--" in values:
        separator = values.index("--")
        compile_args = values[separator + 1 :]
        values = values[:separator]
    batch_gib_explicit = any(
        value == "--batch-gib" or value.startswith("--batch-gib=")
        for value in values
    )
    prefetch_batches_explicit = any(
        value == "--prefetch-batches"
        or value.startswith("--prefetch-batches=")
        for value in values
    )
    args = build_parser().parse_args(values)
    args.compile_args = compile_args
    args.compile_args_explicit = bool(compile_args)
    args.batch_gib_explicit = batch_gib_explicit
    args.prefetch_batches_explicit = prefetch_batches_explicit
    if args.job_id and args.resume_job:
        raise SystemExit("--job-id and --resume-job are mutually exclusive.")
    if args.batch_gib <= 0:
        raise SystemExit("--batch-gib must be greater than zero.")
    if args.prefetch_batches < 1:
        raise SystemExit("--prefetch-batches must be at least one.")
    if args.command_timeout_seconds < 60:
        raise SystemExit("--command-timeout-seconds must be at least 60.")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero.")
    return run_import(args)


if __name__ == "__main__":
    raise SystemExit(main())
