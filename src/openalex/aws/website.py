"""Build, download, and optionally publish the complete website artifact graph."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from openalex.artifacts.cache import DEFAULT_CACHE_DIR, ArtifactCache
from openalex.artifacts.lockfile import DEFAULT_LOCK_PATH, ArtifactLock
from openalex.artifacts.manifest import ArtifactValidationError, read_json
from openalex.artifacts.pipeline import (
    WEBSITE_STAGES,
    StageSpec,
    adopt_local_artifact,
    forced_stage_names,
    local_payload_is_adoptable,
    manifest_input_digests,
    manifest_is_compatible,
    validate_compatible_artifact,
)
from openalex.artifacts.resolver import ArtifactResolver
from openalex.artifacts.s3_store import S3ArtifactStore

from . import configure, runner


TERMINAL_STATUSES = {"success", "failed", "cancelled"}


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Could not determine the current Git commit") from exc


def _remote_manifest(
    s3,
    store: S3ArtifactStore,
    digest: str,
) -> dict[str, Any] | None:
    key = f"{store.artifact_prefix(digest)}/manifest.json"
    try:
        response = s3.get_object(Bucket=store.bucket, Key=key)
    except Exception as exc:
        if runner.client_error_code(exc) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def _can_reseal_local(stage: StageSpec, root: Path) -> bool:
    if not local_payload_is_adoptable(stage, root):
        return False
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return True
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return True
    version = manifest.get("artifact_version")
    if version is not None and int(version) != stage.artifact_version:
        return False
    if manifest.get("recipe_sha256") not in {None, stage.recipe_sha256()}:
        return False
    return not manifest_input_digests(manifest)


def _select_or_publish(
    stage: StageSpec,
    selected_inputs: Mapping[str, str],
    *,
    resolver: ArtifactResolver,
    lock: ArtifactLock,
    store: S3ArtifactStore,
    s3,
) -> str | None:
    root = Path(stage.output)
    try:
        validation = validate_compatible_artifact(stage, root, selected_inputs)
    except (OSError, ValueError):
        validation = None
    if validation is None and _can_reseal_local(stage, root):
        print(f"Adopting existing local payload for {stage.name}: {root}")
        validation = adopt_local_artifact(stage, root, selected_inputs)
    if validation is not None:
        pin = resolver.publish_and_pin(stage.output)
        print(f"Reusing local {stage.name} artifact {pin.manifest_sha256[:12]}.")
        return pin.manifest_sha256

    pin = lock.get(stage.output)
    if pin is None:
        return None
    manifest = _remote_manifest(s3, store, pin.manifest_sha256)
    if manifest is None or not manifest_is_compatible(
        stage, manifest, selected_inputs
    ):
        return None
    print(f"Reusing pinned {stage.name} artifact {pin.manifest_sha256[:12]}.")
    return pin.manifest_sha256


def _submit_stage(
    stage: StageSpec,
    args: argparse.Namespace,
    run_id: str,
) -> None:
    values = [
        "submit",
        "--worker",
        stage.worker,
        "--run-id",
        run_id,
        "--state-path",
        args.state_path,
        "--last-run-path",
        args.last_run_path,
        "--artifact-lock",
        args.artifact_lock,
        "--artifact-cache",
        args.artifact_cache,
        "--branch",
        args.branch,
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    if args.commit:
        values.extend(("--commit", args.commit))
    for _role, dependency_name in stage.dependencies:
        dependency = next(
            candidate
            for candidate in WEBSITE_STAGES
            if candidate.name == dependency_name
        )
        values.extend(("--input", dependency.output))
    command = list(stage.command)
    if stage.manifest_backed:
        command = [
            "env",
            f"OPENALEX_ARTIFACT_RECIPE_SHA256={stage.recipe_sha256()}",
            *command,
        ]
        values.append("--no-auto-inputs")
    values.extend(("--", *command))
    print(f"Submitting {stage.name} on the {stage.worker} worker ({run_id}).")
    runner.main(values)


def _wait_for_run(
    s3,
    state: dict[str, Any],
    client_error,
    run_id: str,
    poll_seconds: float,
) -> dict[str, Any]:
    key = f"{runner.run_prefix(state, run_id)}/status.json"
    while True:
        status = runner.get_s3_json(
            s3,
            state["bucket"],
            key,
            client_error,
        )
        value = None if status is None else status.get("status")
        if value:
            print(f"{run_id}: {value}")
        if value in TERMINAL_STATUSES:
            assert status is not None
            return status
        time.sleep(poll_seconds)


def _show_failure_logs(args: argparse.Namespace, stage: StageSpec, run_id: str) -> None:
    values = [
        "logs",
        "--worker",
        stage.worker,
        "--run-id",
        run_id,
        "--state-path",
        args.state_path,
        "--last-run-path",
        args.last_run_path,
        "--all",
        "--lines",
        "200",
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    runner.main(values)


def _pause_worker(args: argparse.Namespace, worker: str) -> None:
    values = [
        "pause",
        "--worker",
        worker,
        "--state-path",
        args.state_path,
        "--wait",
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    configure.main(values)


def _download_website(
    args: argparse.Namespace,
    run_id: str,
) -> Path:
    values = [
        "download",
        "--worker",
        "cpu",
        "--run-id",
        run_id,
        "--state-path",
        args.state_path,
        "--last-run-path",
        args.last_run_path,
        "--output-dir",
        args.output_dir,
        "--yes",
    ]
    if args.profile:
        values.extend(("--profile", args.profile))
    if args.region:
        values.extend(("--region", args.region))
    runner.main(values)
    website = Path(args.output_dir) / "website"
    if not (website / "index.html").is_file():
        raise SystemExit(f"Downloaded website is incomplete: {website}")
    return website


def _publish_website(args: argparse.Namespace, website: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "publish-gh-pages.sh"
    subprocess.run(
        ["bash", str(script), str(website), args.remote],
        check=True,
    )


def run_website_pipeline(args: argparse.Namespace) -> int:
    args.commit = args.commit or _current_commit()
    forced = forced_stage_names(args.force)
    project_state = runner.load_state(args.state_path)
    session = runner.boto3_session(args, project_state)
    s3 = session.client("s3")
    _boto3, client_error = runner.require_boto3()
    store = S3ArtifactStore(
        s3,
        project_state["bucket"],
        project_state["prefix"],
    )
    lock = ArtifactLock(args.artifact_lock)
    resolver = ArtifactResolver(
        cache=ArtifactCache(args.artifact_cache),
        lock=lock,
        store=store,
    )
    selected: dict[str, str] = {}
    workers_used: set[str] = set()
    website_run_id: str | None = None

    try:
        for stage in WEBSITE_STAGES:
            selected_inputs = {
                role: selected[dependency]
                for role, dependency in stage.dependencies
            }
            if stage.manifest_backed and stage.name not in forced:
                digest = _select_or_publish(
                    stage,
                    selected_inputs,
                    resolver=resolver,
                    lock=lock,
                    store=store,
                    s3=s3,
                )
                if digest is not None:
                    selected[stage.name] = digest
                    continue

            run_id = f"website-{stage.name}-{runner.make_run_id()}"
            workers_used.add(stage.worker)
            _submit_stage(stage, args, run_id)
            status = _wait_for_run(
                s3,
                project_state,
                client_error,
                run_id,
                args.poll_seconds,
            )
            if status.get("status") != "success":
                _show_failure_logs(args, stage, run_id)
                raise SystemExit(
                    f"{stage.name} failed: {status.get('message', 'unknown error')}"
                )
            if stage.manifest_backed:
                pin = runner.pin_run_artifact(
                    s3,
                    project_state,
                    run_id,
                    stage.output,
                    lock,
                )
                selected[stage.name] = pin.manifest_sha256
            else:
                website_run_id = run_id

            if stage.worker == "gpu" and not args.keep_workers_running:
                _pause_worker(args, "gpu")
                workers_used.discard("gpu")

        if website_run_id is None:
            raise SystemExit("Website build did not produce a run")
        website = _download_website(args, website_run_id)
        if args.deploy:
            _publish_website(args, website)
        print(f"Website ready at {website}.")
        return 0
    finally:
        if not args.keep_workers_running:
            for worker in sorted(workers_used):
                try:
                    _pause_worker(args, worker)
                except Exception as exc:
                    print(f"Could not pause {worker} worker: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openalex-aws website",
        description=(
            "Resolve or build the complete website artifact graph, download it, "
            "and publish it."
        ),
    )
    parser.add_argument("--state-path", default=runner.DEFAULT_STATE_PATH)
    parser.add_argument("--last-run-path", default=runner.DEFAULT_LAST_RUN_PATH)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--branch", default=runner.DEFAULT_BRANCH)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--artifact-lock", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--artifact-cache", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--force",
        action="append",
        default=[],
        metavar="STAGE",
        help="Rebuild this stage even when a reusable artifact exists.",
    )
    parser.add_argument("--output-dir", default=runner.DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--keep-workers-running", action="store_true")
    deployment = parser.add_mutually_exclusive_group()
    deployment.add_argument("--deploy", dest="deploy", action="store_true")
    deployment.add_argument("--no-deploy", dest="deploy", action="store_false")
    parser.set_defaults(deploy=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be greater than zero")
    try:
        return run_website_pipeline(args)
    except ArtifactValidationError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
