#!/usr/bin/env python3
"""Submit and monitor remote pipeline runs on the configured AWS worker."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_STATE_PATH = ".aws_runner_state.json"
DEFAULT_LAST_RUN_PATH = ".aws_runner_last_run.json"
DEFAULT_SCRATCH_DIR = "/mnt/aws-runner"
DEFAULT_BRANCH = "bubbles"
DEFAULT_STATUS_INTERVAL_SECONDS = 4 * 60 * 60
DEFAULT_SSM_READY_TIMEOUT_SECONDS = 600
DEFAULT_ARTIFACT_REFRESH_TIMEOUT_SECONDS = 600
DEFAULT_DOWNLOAD_ROOT = "downloads"
CHECKPOINT_FILENAME = "events_checkpoint.pkl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_boto3():
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except ImportError as exc:
        raise SystemExit("boto3 is required for AWS operations. Install it with: python -m pip install boto3") from exc
    return boto3, ClientError


def boto3_session(args: argparse.Namespace, state: Optional[Dict[str, Any]] = None):
    boto3, _ = require_boto3()
    region = args.region or (state or {}).get("region")
    kwargs: Dict[str, str] = {}
    if args.profile:
        kwargs["profile_name"] = args.profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def prefixed_key(prefix: str, key: str) -> str:
    prefix = normalize_prefix(prefix)
    key = key.lstrip("/")
    return f"{prefix}/{key}" if prefix else key


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    bucket_key = uri[5:]
    bucket, _, key = bucket_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected an s3://bucket/key URI, got {uri!r}")
    return bucket, key


def client_error_code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_repo_url(repo_url: str) -> str:
    """Use HTTPS for public GitHub repos so EC2 does not need SSH keys."""
    if repo_url.startswith("git@github.com:"):
        repo_path = repo_url.removeprefix("git@github.com:")
        return f"https://github.com/{repo_path}"
    return repo_url


def load_state(path: str) -> Dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        raise SystemExit(f"State file not found: {state_path}. Run openalex-aws setup first.")
    return load_json(state_path)


def run_prefix(state: Dict[str, Any], run_id: str) -> str:
    return prefixed_key(state["prefix"], f"runs/{run_id}")


def put_s3_json(s3, bucket: str, key: str, data: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def put_s3_text(s3, bucket: str, key: str, text: str, content_type: str = "text/plain") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def get_s3_json(s3, bucket: str, key: str, client_error) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except client_error as exc:
        if client_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return json.loads(obj["Body"].read().decode("utf-8"))


def get_s3_text(s3, bucket: str, key: str, client_error) -> Optional[str]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except client_error as exc:
        if client_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return obj["Body"].read().decode("utf-8", errors="replace")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def latest_run_id_from_s3(s3, bucket: str, prefix: str) -> Optional[str]:
    paginator = s3.get_paginator("list_objects_v2")
    newest: Optional[Tuple[datetime, str]] = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefixed_key(prefix, "runs/")):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/status.json"):
                continue
            run_id = key.rsplit("/", 2)[-2]
            candidate = (item["LastModified"], run_id)
            if newest is None or candidate[0] > newest[0]:
                newest = candidate
    return newest[1] if newest else None


def resolve_run_id(args: argparse.Namespace, s3=None, state: Optional[Dict[str, Any]] = None) -> str:
    if args.run_id:
        return args.run_id
    last_run_path = Path(args.last_run_path)
    if last_run_path.exists():
        last = load_json(last_run_path)
        if last.get("run_id"):
            return last["run_id"]
    if s3 is not None and state is not None:
        latest = latest_run_id_from_s3(s3, state["bucket"], state["prefix"])
        if latest:
            return latest
    raise SystemExit("No run id supplied and no previous run was found.")


def artifact_output_prefix(state: Dict[str, Any], run_id: str) -> str:
    return f"{run_prefix(state, run_id)}/output/"


def list_run_artifacts(
    s3,
    state: Dict[str, Any],
    run_id: str,
    include_checkpoint: bool = False,
) -> List[Dict[str, Any]]:
    """List downloadable run artifacts, including legacy compiled database locations."""
    bucket = state["bucket"]
    output_prefix = artifact_output_prefix(state, run_id)
    artifacts: Dict[str, Dict[str, Any]] = {}
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative_path = key.removeprefix(output_prefix)
            if not relative_path or relative_path.endswith("/"):
                continue
            if not include_checkpoint and PurePosixPath(relative_path).name == CHECKPOINT_FILENAME:
                continue
            artifacts[relative_path] = {
                "key": key,
                "relative_path": relative_path,
                "size": item.get("Size", 0),
                "last_modified": item.get("LastModified"),
            }

    # compile-from-snapshot runs created before artifact normalization stored
    # the database at the run root rather than beneath output/.
    legacy_key = f"{run_prefix(state, run_id)}/compiled_articles.db"
    for page in paginator.paginate(Bucket=bucket, Prefix=legacy_key):
        for item in page.get("Contents", []):
            if item["Key"] != legacy_key:
                continue
            artifacts.setdefault(
                "compiled_articles.db",
                {
                    "key": legacy_key,
                    "relative_path": "compiled_articles.db",
                    "size": item.get("Size", 0),
                    "last_modified": item.get("LastModified"),
                },
            )

    return [artifacts[name] for name in sorted(artifacts)]


def safe_artifact_destination(root: Path, relative_path: str) -> Path:
    """Map an S3 artifact name beneath root without allowing path traversal."""
    posix_path = PurePosixPath(relative_path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe artifact path: {relative_path!r}")
    return root.joinpath(*posix_path.parts)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def build_artifact_refresh_script(
    state: Dict[str, Any],
    status: Dict[str, Any],
    run_id: str,
    include_checkpoint: bool = False,
) -> str:
    """Build a remote script that publishes current and legacy artifacts."""
    scratch_dir = str(status.get("scratch_dir") or DEFAULT_SCRATCH_DIR).rstrip("/")
    run_dir = f"{scratch_dir}/runs/{run_id}/work"
    destination = s3_uri(state["bucket"], artifact_output_prefix(state, run_id))
    checkpoint_option = "" if include_checkpoint else f" --exclude {q(CHECKPOINT_FILENAME)}"
    compiled_db = f"{scratch_dir}/compiled_articles.db"

    return f"""set -Eeuo pipefail
RUN_DIR={q(run_dir)}
DESTINATION={q(destination)}
COMPILED_DB={q(compiled_db)}
if [ -d "$RUN_DIR/output" ]; then
  aws s3 sync --no-progress "$RUN_DIR/output" "$DESTINATION"{checkpoint_option}
fi
if [ -f "$COMPILED_DB" ]; then
  aws s3 cp --no-progress "$COMPILED_DB" "${{DESTINATION}}compiled_articles.db"
fi
"""


def wait_for_ssm_command(ssm, command_id: str, instance_id: str, timeout: int) -> Dict[str, Any]:
    deadline = time.time() + timeout
    pending_statuses = {"Pending", "InProgress", "Delayed"}
    while time.time() < deadline:
        try:
            invocation = ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=instance_id,
            )
        except Exception as exc:
            if client_error_code(exc) == "InvocationDoesNotExist":
                time.sleep(2)
                continue
            raise

        status = invocation.get("Status")
        if status in pending_statuses:
            time.sleep(2)
            continue
        if status == "Success":
            return invocation
        details = invocation.get("StandardErrorContent") or invocation.get("StatusDetails") or status
        raise SystemExit(f"Artifact refresh failed via SSM ({status}): {details}")

    raise SystemExit(f"Timed out after {timeout}s waiting for artifact refresh command {command_id}")


def refresh_run_artifacts(
    session,
    state: Dict[str, Any],
    status: Dict[str, Any],
    run_id: str,
    include_checkpoint: bool,
    timeout: int,
) -> None:
    instance_id = state.get("instance_id")
    if not instance_id:
        raise SystemExit("No instance_id is configured; cannot refresh remote artifacts.")

    script = build_artifact_refresh_script(
        state,
        status,
        run_id,
        include_checkpoint=include_checkpoint,
    )
    ssm = session.client("ssm")
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment=f"refresh openalex artifacts for {run_id}",
        Parameters={"commands": [script]},
        TimeoutSeconds=timeout,
    )
    command_id = response["Command"]["CommandId"]
    print(f"Refreshing artifacts from {instance_id} via SSM ({command_id})...")
    invocation = wait_for_ssm_command(ssm, command_id, instance_id, timeout)
    output = (invocation.get("StandardOutputContent") or "").strip()
    if output:
        print(output)


def instance_state(ec2, instance_id: str) -> str:
    response = ec2.describe_instances(InstanceIds=[instance_id])
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            return instance.get("State", {}).get("Name", "unknown")
    return "unknown"


def instance_host(session, instance_id: str) -> Optional[str]:
    ec2 = session.client("ec2")
    response = ec2.describe_instances(InstanceIds=[instance_id])
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            return instance.get("PublicDnsName") or instance.get("PublicIpAddress") or instance.get("PrivateIpAddress")
    return None


def ensure_instance_running(session, state: Dict[str, Any], auto_start: bool, wait_for_ssm: bool, timeout: int) -> None:
    ec2 = session.client("ec2")
    instance_id = state["instance_id"]
    current = instance_state(ec2, instance_id)
    if current == "running":
        print(f"Instance {instance_id} is running.")
    elif current == "stopped" and auto_start:
        print(f"Starting instance {instance_id}...")
        ec2.start_instances(InstanceIds=[instance_id])
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        print(f"Instance {instance_id} is running.")
    else:
        raise SystemExit(f"Instance {instance_id} is {current}; start it first or omit --no-start.")

    if wait_for_ssm:
        wait_until_ssm_online(session, instance_id, timeout)


def wait_until_ssm_online(session, instance_id: str, timeout: int) -> None:
    ssm = session.client("ssm")
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
        )
        instances = response.get("InstanceInformationList", [])
        if instances and instances[0].get("PingStatus") == "Online":
            print(f"SSM is online for {instance_id}.")
            return
        time.sleep(10)
    raise SystemExit(
        f"Timed out waiting for SSM on {instance_id}. "
        "Check the AMI, SSM agent, instance profile, subnet egress, and IAM permissions."
    )


def pipeline_command(args: argparse.Namespace) -> str:
    """Return an argument-safe command supplied after `submit --`."""
    values = list(args.remote_command or [])
    if values and values[0] == "--":
        values.pop(0)
    if not values:
        raise SystemExit("A command is required, for example: submit -- openalex events")
    return shlex.join(values)


def q(value: Any) -> str:
    return shlex.quote("" if value is None else str(value))


def build_remote_runner_script(
    *,
    state: Dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    command: str,
) -> str:
    bucket = state["bucket"]
    run_s3_prefix = run_prefix(state, run_id)
    needs_input_db = not args.no_database
    db_s3_uri = args.db_s3_uri or state.get("db_s3_uri")
    repo_url = args.repo_url or state.get("repo_url")
    if needs_input_db and not db_s3_uri:
        raise SystemExit("No database S3 URI configured. Run setup with --db-path or pass --db-s3-uri.")
    if not repo_url:
        raise SystemExit("No repo URL configured. Run setup with --repo-url or pass --repo-url.")
    repo_url = normalize_repo_url(repo_url)
    command_name = next(
        (value for value in args.remote_command if value != "--"),
        "command",
    )

    return f"""#!/usr/bin/env bash
set -Eeuo pipefail

export HOME="${{HOME:-/root}}"
export PATH="$HOME/.local/bin:$PATH"
RUN_ID={q(run_id)}
PIPELINE={q(command_name)}
BUCKET={q(bucket)}
RUN_S3_PREFIX={q(run_s3_prefix)}
REPO_URL={q(repo_url)}
BRANCH={q(args.branch)}
COMMIT={q(args.commit or "")}
DB_S3_URI={q(db_s3_uri)}
NEEDS_INPUT_DB={q("1" if needs_input_db else "0")}
SNAPSHOT_S3_URI={q(args.snapshot_s3_uri or "")}
SCRATCH_DIR={q(args.scratch_dir)}
SNAPSHOT_LOCAL_DIR="${{SCRATCH_DIR}}/openalex-snapshot"
STATUS_INTERVAL_SECONDS={q(args.status_interval_seconds)}
PIPELINE_COMMAND={q(command)}
INSTALL_DEPS={q("1" if not args.skip_dependency_install else "0")}
FORCE_DB_DOWNLOAD={q("1" if args.force_db_download else "0")}
INSTANCE_ID="$(curl -fsS --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)"
export BUCKET RUN_S3_PREFIX PIPELINE_COMMAND

WORK_DIR="${{SCRATCH_DIR}}/runs/${{RUN_ID}}"
RUN_DIR="${{WORK_DIR}}/work"
REPO_DIR="${{SCRATCH_DIR}}/repo"
VENV_DIR="${{SCRATCH_DIR}}/venv"
STATUS_FILE="${{WORK_DIR}}/status.json"
STDOUT_LOG="${{WORK_DIR}}/stdout.log"
STDERR_LOG="${{WORK_DIR}}/stderr.log"
LAUNCHER_LOG="${{WORK_DIR}}/launcher.log"
PID_FILE="${{WORK_DIR}}/pid"
if ! mkdir -p "${{WORK_DIR}}" "${{RUN_DIR}}" "${{SCRATCH_DIR}}" 2>/dev/null; then
  sudo mkdir -p "${{WORK_DIR}}" "${{RUN_DIR}}" "${{SCRATCH_DIR}}"
  sudo chown -R "$(id -u):$(id -g)" "${{SCRATCH_DIR}}"
fi
touch "${{STDOUT_LOG}}" "${{STDERR_LOG}}" "${{LAUNCHER_LOG}}"

write_status() {{
  local status="$1"
  local message="$2"
  local exit_code="${{3:-}}"
  python3 - "$STATUS_FILE" "$RUN_ID" "$PIPELINE" "$status" "$message" "$INSTANCE_ID" "$RUN_S3_PREFIX" "$exit_code" "$SCRATCH_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path, run_id, pipeline, status, message, instance_id, run_s3_prefix, exit_code, scratch_dir = sys.argv[1:10]
payload = {{
    "run_id": run_id,
    "pipeline": pipeline,
    "status": status,
    "message": message,
    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "instance_id": instance_id,
    "pid": None,
    "scratch_dir": scratch_dir,
    "s3_status_uri": f"s3://{{os.environ['BUCKET']}}/{{run_s3_prefix}}/status.json",
    "s3_stdout_uri": f"s3://{{os.environ['BUCKET']}}/{{run_s3_prefix}}/stdout.log",
    "s3_stderr_uri": f"s3://{{os.environ['BUCKET']}}/{{run_s3_prefix}}/stderr.log",
    "s3_output_prefix": f"s3://{{os.environ['BUCKET']}}/{{run_s3_prefix}}/output/",
}}
pid_file = os.path.join(scratch_dir, "runs", run_id, "pid")
if os.path.exists(pid_file):
    try:
        payload["pid"] = int(open(pid_file, encoding="utf-8").read().strip())
    except Exception:
        payload["pid"] = None
if exit_code:
    payload["exit_code"] = int(exit_code)
prior = {{}}
if os.path.exists(path):
    try:
        prior = json.load(open(path, encoding="utf-8"))
    except Exception:
        prior = {{}}
payload["started_at"] = prior.get("started_at", payload["updated_at"])
payload["command"] = prior.get("command", os.environ.get("PIPELINE_COMMAND", ""))
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
}}

sync_artifacts() {{
  aws s3 cp "$STATUS_FILE" "s3://${{BUCKET}}/${{RUN_S3_PREFIX}}/status.json" >/dev/null 2>&1 || true
  aws s3 cp "$STDOUT_LOG" "s3://${{BUCKET}}/${{RUN_S3_PREFIX}}/stdout.log" >/dev/null 2>&1 || true
  aws s3 cp "$STDERR_LOG" "s3://${{BUCKET}}/${{RUN_S3_PREFIX}}/stderr.log" >/dev/null 2>&1 || true
  aws s3 cp "$LAUNCHER_LOG" "s3://${{BUCKET}}/${{RUN_S3_PREFIX}}/launcher.log" >/dev/null 2>&1 || true
}}

sync_output() {{
  if [ -d "${{RUN_DIR}}/output" ]; then
    aws s3 sync "${{RUN_DIR}}/output" "s3://${{BUCKET}}/${{RUN_S3_PREFIX}}/output/" >/dev/null 2>&1 || true
  fi
}}

install_system_tools() {{
  if command -v aws >/dev/null 2>&1 && command -v git >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y awscli git python3 python3-pip sqlite >/tmp/aws-runner-package-install.log 2>&1 || true
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y awscli git python3 python3-pip sqlite >/tmp/aws-runner-package-install.log 2>&1 || true
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update >/tmp/aws-runner-package-install.log 2>&1 || true
    sudo apt-get install -y awscli git python3 python3-pip sqlite3 >>/tmp/aws-runner-package-install.log 2>&1 || true
  fi
  if ! command -v aws >/dev/null 2>&1; then
    python3 -m pip install --user awscli >>/tmp/aws-runner-package-install.log 2>&1
  fi
}}

install_python_deps() {{
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  export PATH="$VENV_DIR/bin:$PATH"
  if [ "$INSTALL_DEPS" != "1" ]; then
    return
  fi
  local fingerprint
  fingerprint="$(cd "$REPO_DIR" && {{ git rev-parse HEAD; shasum -a 256 pyproject.toml; }} | shasum -a 256 | awk '{{print $1}}')"
  if [ -f "$VENV_DIR/.openalex-install" ] && [ "$(cat "$VENV_DIR/.openalex-install")" = "$fingerprint" ]; then
    echo "Reusing installed OpenAlex package at revision $fingerprint" >>"$LAUNCHER_LOG"
    return
  fi
  python -m pip install --upgrade pip >>"$LAUNCHER_LOG" 2>&1
  python -m pip install -e "$REPO_DIR[analysis,website,parquet,aws,download]" >>"$LAUNCHER_LOG" 2>&1
  printf '%s\\n' "$fingerprint" > "$VENV_DIR/.openalex-install"
  python3 - <<'PY' >>"$LAUNCHER_LOG" 2>&1 || true
import nltk
for resource in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:
        print(f"nltk download failed for {{resource}}: {{exc}}")
PY
}}

sync_repo() {{
  if [ ! -d "$REPO_DIR/.git" ]; then
    rm -rf "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR" >>"$LAUNCHER_LOG" 2>&1
  fi
  cd "$REPO_DIR"
  git remote set-url origin "$REPO_URL"
  git fetch --all --prune >>"$LAUNCHER_LOG" 2>&1
  if [ -n "$COMMIT" ]; then
    git checkout "$COMMIT" >>"$LAUNCHER_LOG" 2>&1
  else
    git checkout "$BRANCH" >>"$LAUNCHER_LOG" 2>&1 || git checkout -B "$BRANCH" "origin/$BRANCH" >>"$LAUNCHER_LOG" 2>&1
    git pull --ff-only origin "$BRANCH" >>"$LAUNCHER_LOG" 2>&1
  fi
}}

sync_snapshot() {{
  if [ -z "$SNAPSHOT_S3_URI" ]; then
    return
  fi
  mkdir -p "$SNAPSHOT_LOCAL_DIR"
  aws s3 sync --no-progress "$SNAPSHOT_S3_URI" "$SNAPSHOT_LOCAL_DIR" >>"$LAUNCHER_LOG" 2>&1
}}

download_db() {{
  local cache_dir="${{SCRATCH_DIR}}/cache"
  local target="${{cache_dir}}/articles.db"
  local manifest="${{cache_dir}}/articles.db.json"
  local tmp_target="${{target}}.download"
  local db_without_scheme="${{DB_S3_URI#s3://}}"
  local db_bucket="${{db_without_scheme%%/*}}"
  local db_key="${{db_without_scheme#*/}}"
  local remote_size=""
  local remote_etag=""
  local remote_version=""

  mkdir -p "$cache_dir"
  remote_size="$(aws s3api head-object --bucket "$db_bucket" --key "$db_key" --query ContentLength --output text 2>>"$LAUNCHER_LOG" || true)"
  remote_etag="$(aws s3api head-object --bucket "$db_bucket" --key "$db_key" --query ETag --output text 2>>"$LAUNCHER_LOG" || true)"
  remote_version="$(aws s3api head-object --bucket "$db_bucket" --key "$db_key" --query VersionId --output text 2>>"$LAUNCHER_LOG" || true)"

  if [ "$FORCE_DB_DOWNLOAD" != "1" ] && python3 - "$target" "$manifest" "$DB_S3_URI" "$remote_size" "$remote_etag" "$remote_version" <<'PY'
import json
from pathlib import Path
import sys
target, manifest, uri, size, etag, version = sys.argv[1:]
path = Path(target)
try:
    cached = json.loads(Path(manifest).read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {{"uri": uri, "size": int(size), "etag": etag, "version_id": version}}
raise SystemExit(0 if path.is_file() and path.stat().st_size == int(size) and cached == expected else 1)
PY
  then
    echo "Reusing version-matched SQLite database at $target" >>"$LAUNCHER_LOG"
  else
    echo "Downloading SQLite database from $DB_S3_URI to $target" >>"$LAUNCHER_LOG"
    rm -f "$tmp_target"
    aws s3 cp --no-progress "$DB_S3_URI" "$tmp_target" >>"$LAUNCHER_LOG" 2>&1
    mv "$tmp_target" "$target"
    python3 - "$manifest" "$DB_S3_URI" "$remote_size" "$remote_etag" "$remote_version" <<'PY'
import json
from pathlib import Path
import sys
manifest, uri, size, etag, version = sys.argv[1:]
Path(manifest).write_text(json.dumps({{
    "uri": uri,
    "size": int(size),
    "etag": etag,
    "version_id": version,
}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  fi
  chmod a-w "$target"
  ln -sf "$target" "${{RUN_DIR}}/articles.db"
}}

CHILD_PID=""
on_term() {{
  if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  write_status "cancelled" "Received termination signal" "130"
  sync_artifacts
  sync_output
  exit 130
}}
trap on_term INT TERM

(
  write_status "running" "Installing system tools"
  sync_artifacts
  install_system_tools
  sync_artifacts

  write_status "running" "Syncing repository"
  sync_artifacts
  sync_repo
  sync_artifacts

  write_status "running" "Installing Python dependencies"
  sync_artifacts
  install_python_deps
  sync_artifacts

  if [ -n "$SNAPSHOT_S3_URI" ]; then
    write_status "running" "Syncing snapshot data"
    sync_artifacts
    sync_snapshot
    sync_artifacts
  fi

  if [ "$NEEDS_INPUT_DB" = "1" ]; then
    write_status "running" "Downloading SQLite database"
    sync_artifacts
    download_db
    sync_artifacts
  fi

  export PATH="$VENV_DIR/bin:$PATH"
  cd "$RUN_DIR"
  write_status "running" "Executing: $PIPELINE_COMMAND"
  sync_artifacts
  bash -lc "$PIPELINE_COMMAND" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
  CHILD_PID=$!
  echo "$CHILD_PID" > "$PID_FILE"

  last_sync=0
  while kill -0 "$CHILD_PID" 2>/dev/null; do
    now="$(date +%s)"
    if [ "$((now - last_sync))" -ge "$STATUS_INTERVAL_SECONDS" ]; then
      write_status "running" "Pipeline running with pid $CHILD_PID"
      sync_artifacts
      last_sync="$now"
    fi
    sleep 60
  done

  set +e
  wait "$CHILD_PID"
  exit_code=$?
  set -e
  sync_output
  if [ "$exit_code" -eq 0 ]; then
    write_status "success" "Pipeline completed successfully" "$exit_code"
  else
    write_status "failed" "Pipeline failed with exit code $exit_code" "$exit_code"
  fi
  sync_artifacts
  sync_output
  exit "$exit_code"
) >>"$LAUNCHER_LOG" 2>&1
"""


def build_launcher_script(remote_script: str, args: argparse.Namespace, run_id: str) -> str:
    work_dir = f"{args.scratch_dir.rstrip('/')}/runs/{run_id}"
    run_script = f"{work_dir}/remote_runner.sh"
    scratch_dir = args.scratch_dir.rstrip("/")
    return f"""set -Eeuo pipefail
if ! mkdir -p {q(work_dir)} 2>/dev/null; then
  sudo mkdir -p {q(work_dir)}
  sudo chown -R "$(id -u):$(id -g)" {q(scratch_dir)}
fi
cat > {q(run_script)} <<'AWS_RUNNER_REMOTE_SCRIPT'
{remote_script}
AWS_RUNNER_REMOTE_SCRIPT
chmod +x {q(run_script)}
nohup bash {q(run_script)} > {q(work_dir + "/ssm_launcher.out")} 2>&1 &
echo "Started AWS pipeline runner {run_id}"
"""


def send_ssm_command(session, instance_id: str, launcher_script: str, run_id: str) -> str:
    ssm = session.client("ssm")
    response = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment=f"openalex pipeline run {run_id}",
        Parameters={"commands": [launcher_script]},
        TimeoutSeconds=3600,
    )
    return response["Command"]["CommandId"]


def send_ssh_command(session, state: Dict[str, Any], args: argparse.Namespace, launcher_script: str) -> str:
    host = args.ssh_host or instance_host(session, state["instance_id"])
    if not host:
        raise SystemExit("Could not determine an SSH host for the instance. Pass --ssh-host explicitly.")
    ssh_target = f"{args.ssh_user}@{host}"
    command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    for option in args.ssh_option or []:
        command.extend(["-o", option])
    if args.ssh_key_path:
        command.extend(["-i", args.ssh_key_path])
    command.extend([ssh_target, "bash -s"])
    subprocess.run(command, input=launcher_script, text=True, check=True)
    return f"ssh:{ssh_target}"


def initial_status(
    state: Dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    command: str,
    status: str,
    message: str,
    command_id: Optional[str] = None,
) -> Dict[str, Any]:
    prefix = run_prefix(state, run_id)
    payload = {
        "run_id": run_id,
        "pipeline": next(
            (value for value in args.remote_command if value != "--"),
            "command",
        ),
        "status": status,
        "message": message,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "instance_id": state.get("instance_id"),
        "command_id": command_id,
        "command": command,
        "branch": args.branch,
        "commit": args.commit,
        "scratch_dir": args.scratch_dir,
        "s3_status_uri": s3_uri(state["bucket"], f"{prefix}/status.json"),
        "s3_stdout_uri": s3_uri(state["bucket"], f"{prefix}/stdout.log"),
        "s3_stderr_uri": s3_uri(state["bucket"], f"{prefix}/stderr.log"),
        "s3_output_prefix": s3_uri(state["bucket"], f"{prefix}/output/"),
    }
    return payload


def command_submit(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    session = boto3_session(args, state)
    s3 = session.client("s3")
    run_id = args.run_id or make_run_id()
    command = pipeline_command(args)
    bucket = state["bucket"]
    prefix = run_prefix(state, run_id)

    if not args.no_start:
        ensure_instance_running(
            session,
            state,
            auto_start=True,
            wait_for_ssm=args.executor == "ssm",
            timeout=args.ssm_ready_timeout_seconds,
        )

    status = initial_status(state, args, run_id, command, "submitted", "Submitted remote command")
    put_s3_json(s3, bucket, f"{prefix}/status.json", status)
    remote_script = build_remote_runner_script(state=state, args=args, run_id=run_id, command=command)
    put_s3_text(s3, bucket, f"{prefix}/remote_runner.sh", remote_script, content_type="text/x-shellscript")

    launcher_script = build_launcher_script(remote_script, args, run_id)
    if args.executor == "ssm":
        command_id = send_ssm_command(session, state["instance_id"], launcher_script, run_id)
    else:
        command_id = send_ssh_command(session, state, args, launcher_script)
    status["command_id"] = command_id
    status["message"] = "SSM command accepted; remote runner is starting"
    status["updated_at"] = utc_now()
    put_s3_json(s3, bucket, f"{prefix}/status.json", status)
    write_json(Path(args.last_run_path), {"run_id": run_id, "updated_at": utc_now()})

    print(f"Submitted command as run_id={run_id}")
    print(f"Status: {status['s3_status_uri']}")
    print(f"Logs:   {status['s3_stdout_uri']}")
    print(f"SSM command_id={command_id}")
    return 0


def print_status(status: Dict[str, Any]) -> None:
    print(f"run_id: {status.get('run_id')}")
    print(f"status: {status.get('status')}")
    print(f"message: {status.get('message')}")
    print(f"updated_at: {status.get('updated_at')}")
    print(f"pipeline: {status.get('pipeline')}")
    print(f"instance_id: {status.get('instance_id')}")
    if status.get("pid"):
        print(f"pid: {status.get('pid')}")
    if status.get("exit_code") is not None:
        print(f"exit_code: {status.get('exit_code')}")
    if status.get("s3_stdout_uri"):
        print(f"stdout: {status.get('s3_stdout_uri')}")
    if status.get("s3_stderr_uri"):
        print(f"stderr: {status.get('s3_stderr_uri')}")
    if status.get("s3_output_prefix"):
        print(f"output: {status.get('s3_output_prefix')}")


def command_status(args: argparse.Namespace) -> int:
    boto3, client_error = require_boto3()
    state = load_state(args.state_path)
    session = boto3_session(args, state)
    s3 = session.client("s3")
    run_id = resolve_run_id(args, s3=s3, state=state)
    key = f"{run_prefix(state, run_id)}/status.json"
    status = get_s3_json(s3, state["bucket"], key, client_error)
    if status is None:
        raise SystemExit(f"No status found at {s3_uri(state['bucket'], key)}")
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_status(status)
    return 0


def tail_text(text: str, lines: int) -> str:
    split = text.splitlines()
    if lines > 0:
        split = split[-lines:]
    return "\n".join(split)


def command_logs(args: argparse.Namespace) -> int:
    boto3, client_error = require_boto3()
    state = load_state(args.state_path)
    session = boto3_session(args, state)
    s3 = session.client("s3")
    run_id = resolve_run_id(args, s3=s3, state=state)
    names = ["stdout.log"]
    if args.stderr:
        names = ["stderr.log"]
    if args.launcher:
        names = ["launcher.log"]
    if args.all:
        names = ["launcher.log", "stdout.log", "stderr.log"]
    for name in names:
        key = f"{run_prefix(state, run_id)}/{name}"
        text = get_s3_text(s3, state["bucket"], key, client_error)
        if text is None:
            print(f"Missing {s3_uri(state['bucket'], key)}")
            continue
        if len(names) > 1:
            print(f"==> {name} <==")
        print(tail_text(text, args.lines))
    return 0


def artifact_command_context(args: argparse.Namespace):
    state = load_state(args.state_path)
    session = boto3_session(args, state)
    s3 = session.client("s3")
    run_id = resolve_run_id(args, s3=s3, state=state)
    status_key = f"{run_prefix(state, run_id)}/status.json"
    status = get_s3_json(s3, state["bucket"], status_key, require_boto3()[1]) or {}

    if args.refresh:
        refresh_run_artifacts(
            session,
            state,
            status,
            run_id,
            include_checkpoint=args.include_checkpoint,
            timeout=args.refresh_timeout_seconds,
        )

    artifacts = list_run_artifacts(
        s3,
        state,
        run_id,
        include_checkpoint=args.include_checkpoint,
    )
    return state, s3, run_id, status, artifacts


def command_artifacts(args: argparse.Namespace) -> int:
    state, _s3, run_id, status, artifacts = artifact_command_context(args)
    print(f"run_id: {run_id}")
    if status.get("status"):
        print(f"status: {status['status']}")
    print(f"source: {s3_uri(state['bucket'], artifact_output_prefix(state, run_id))}")

    if not artifacts:
        print("No downloadable artifacts found.")
        if not args.refresh:
            print("Use --refresh to publish current files from the worker.")
        return 0

    total_size = 0
    for artifact in artifacts:
        total_size += artifact["size"]
        print(f"{format_bytes(artifact['size']):>10}  {artifact['relative_path']}")
    print(f"{len(artifacts)} artifact(s), {format_bytes(total_size)} total")
    if not args.include_checkpoint:
        print(f"Checkpoint excluded; pass --include-checkpoint to include {CHECKPOINT_FILENAME}.")
    return 0


def command_download(args: argparse.Namespace) -> int:
    state, s3, run_id, status, artifacts = artifact_command_context(args)
    if not artifacts:
        raise SystemExit(
            "No downloadable artifacts found. If the worker still has the files, retry with --refresh."
        )

    destination = Path(args.output_dir) if args.output_dir else Path(DEFAULT_DOWNLOAD_ROOT) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    total_size = 0
    print(
        f"Downloading {len(artifacts)} artifact(s) for {run_id}"
        f" ({status.get('status', 'unknown')}) to {destination}..."
    )
    for artifact in artifacts:
        local_path = safe_artifact_destination(destination, artifact["relative_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {artifact['relative_path']} ({format_bytes(artifact['size'])})")
        s3.download_file(state["bucket"], artifact["key"], str(local_path))
        total_size += artifact["size"]

    print(f"Downloaded {format_bytes(total_size)} to {destination}")
    if not args.include_checkpoint:
        print(f"Checkpoint excluded; pass --include-checkpoint to download {CHECKPOINT_FILENAME}.")
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    session = boto3_session(args, state)
    s3 = session.client("s3")
    run_id = resolve_run_id(args, s3=s3, state=state)
    status_key = f"{run_prefix(state, run_id)}/status.json"
    status = get_s3_json(s3, state["bucket"], status_key, require_boto3()[1]) or {}
    scratch_dir = args.scratch_dir or status.get("scratch_dir") or DEFAULT_SCRATCH_DIR
    pid_file = f"{scratch_dir.rstrip('/')}/runs/{run_id}/pid"
    cancel_script = f"""set -Eeuo pipefail
if [ -f {q(pid_file)} ]; then
  pid="$(cat {q(pid_file)})"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Sent TERM to $pid"
  else
    echo "Process $pid is not running"
  fi
else
  echo "PID file not found: {pid_file}"
fi
"""
    if args.executor == "ssm":
        command_id = send_ssm_command(session, state["instance_id"], cancel_script, f"cancel-{run_id}")
    else:
        command_id = send_ssh_command(session, state, args, cancel_script)
    status.update(
        {
            "run_id": run_id,
            "status": "cancelling",
            "message": f"Cancellation requested via SSM command {command_id}",
            "updated_at": utc_now(),
            "cancel_command_id": command_id,
        }
    )
    put_s3_json(s3, state["bucket"], status_key, status)
    print(f"Cancellation requested for {run_id}; command_id={command_id}")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="Local AWS runner state file.")
    parser.add_argument("--last-run-path", default=DEFAULT_LAST_RUN_PATH, help="Local file storing the latest run id.")
    parser.add_argument("--profile", default=None, help="AWS profile name.")
    parser.add_argument("--region", default=None, help="AWS region. Defaults to the configured state region.")


def add_artifact_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    parser.add_argument("--run-id", default=None, help="Run id. Defaults to the latest local/S3 run.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Publish current worker artifacts to S3 before listing or downloading.",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help=f"Include the potentially large {CHECKPOINT_FILENAME} file.",
    )
    parser.add_argument(
        "--refresh-timeout-seconds",
        type=int,
        default=DEFAULT_ARTIFACT_REFRESH_TIMEOUT_SECONDS,
        help="Maximum time to wait for an SSM artifact refresh.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and monitor pipelines on the configured AWS worker.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    submit = subparsers.add_parser("submit", help="Submit a remote pipeline run and return immediately.")
    add_common_args(submit)
    submit.add_argument("--run-id", default=None, help="Explicit run id. Defaults to a timestamped id.")
    submit.add_argument("--branch", default=DEFAULT_BRANCH, help="Git branch to checkout before running.")
    submit.add_argument("--commit", default=None, help="Specific commit SHA to checkout instead of the branch head.")
    submit.add_argument("--repo-url", default=None, help="Override repo URL from state.")
    submit.add_argument("--db-s3-uri", default=None, help="Override SQLite database S3 URI from state.")
    submit.add_argument(
        "--no-database",
        action="store_true",
        help="Do not fetch or link the configured corpus for this command.",
    )
    submit.add_argument("--force-db-download", action="store_true", help="Redownload the SQLite DB even if a complete local copy exists.")
    submit.add_argument("--snapshot-s3-uri", default=None, help="S3 URI for an OpenAlex snapshot, used by compile-from-snapshot.")
    submit.add_argument("--scratch-dir", default=DEFAULT_SCRATCH_DIR, help="Remote local scratch directory.")
    submit.add_argument("--executor", choices=["ssm", "ssh"], default="ssm", help="Remote executor.")
    submit.add_argument("--ssh-user", default="ec2-user", help="SSH user for --executor ssh.")
    submit.add_argument("--ssh-key-path", default=None, help="Private key path for --executor ssh.")
    submit.add_argument("--ssh-host", default=None, help="Explicit SSH host. Defaults to EC2 public DNS/IP.")
    submit.add_argument("--ssh-option", action="append", default=[], help="Extra ssh -o option, e.g. ProxyJump=host.")
    submit.add_argument("--no-start", action="store_true", help="Do not start/wait for the instance before submitting.")
    submit.add_argument("--skip-dependency-install", action="store_true", help="Skip remote Python dependency installation.")
    submit.add_argument("--status-interval-seconds", type=int, default=DEFAULT_STATUS_INTERVAL_SECONDS)
    submit.add_argument("--ssm-ready-timeout-seconds", type=int, default=DEFAULT_SSM_READY_TIMEOUT_SECONDS)
    submit.add_argument(
        "remote_command",
        nargs=argparse.REMAINDER,
        help="Command and arguments after --, for example: -- openalex events.",
    )
    submit.set_defaults(func=command_submit)

    status = subparsers.add_parser("status", help="Print the latest S3 status for a run.")
    add_common_args(status)
    status.add_argument("--run-id", default=None, help="Run id. Defaults to the latest local/S3 run.")
    status.add_argument("--json", action="store_true", help="Print raw status JSON.")
    status.set_defaults(func=command_status)

    logs = subparsers.add_parser("logs", help="Print the latest synced run logs from S3.")
    add_common_args(logs)
    logs.add_argument("--run-id", default=None, help="Run id. Defaults to the latest local/S3 run.")
    logs.add_argument("--lines", type=int, default=100, help="Number of trailing lines to print. Use 0 for all.")
    logs.add_argument("--stderr", action="store_true", help="Show stderr.log instead of stdout.log.")
    logs.add_argument("--launcher", action="store_true", help="Show launcher/bootstrap log instead of stdout.log.")
    logs.add_argument("--all", action="store_true", help="Show launcher, stdout, and stderr logs.")
    logs.set_defaults(func=command_logs)

    artifacts = subparsers.add_parser("artifacts", help="List result artifacts stored for a run.")
    add_artifact_args(artifacts)
    artifacts.set_defaults(func=command_artifacts)

    download = subparsers.add_parser("download", help="Download result artifacts for a run.")
    add_artifact_args(download)
    download.add_argument(
        "--output-dir",
        default=None,
        help=f"Local destination. Defaults to {DEFAULT_DOWNLOAD_ROOT}/<run-id>.",
    )
    download.set_defaults(func=command_download)

    cancel = subparsers.add_parser("cancel", help="Request cancellation of a running remote pipeline.")
    add_common_args(cancel)
    cancel.add_argument("--run-id", default=None, help="Run id. Defaults to the latest local/S3 run.")
    cancel.add_argument("--scratch-dir", default=None, help="Remote scratch directory override.")
    cancel.add_argument("--executor", choices=["ssm", "ssh"], default="ssm", help="Remote executor for the cancel request.")
    cancel.add_argument("--ssh-user", default="ec2-user", help="SSH user for --executor ssh.")
    cancel.add_argument("--ssh-key-path", default=None, help="Private key path for --executor ssh.")
    cancel.add_argument("--ssh-host", default=None, help="Explicit SSH host. Defaults to EC2 public DNS/IP.")
    cancel.add_argument("--ssh-option", action="append", default=[], help="Extra ssh -o option, e.g. ProxyJump=host.")
    cancel.set_defaults(func=command_cancel)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
