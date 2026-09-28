#!/usr/bin/env python3
"""Provision and manage AWS resources for the social-science pipeline runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_STATE_PATH = ".aws_runner_state.json"
DEFAULT_PREFIX = "social-science"
DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "lucas-epistemic-bubbles"
DEFAULT_REPO_URL = "git@github.com:lucasgautheron/social-science.git"
DEFAULT_INSTANCE_TYPE = "i4i.8xlarge"
DEFAULT_AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
DEFAULT_ROOT_VOLUME_GB = 200
STATE_S3_KEY = "state/aws_runner_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_boto3():
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except ImportError as exc:
        raise SystemExit("boto3 is required for AWS operations. Install it with: python -m pip install boto3") from exc
    return boto3, ClientError


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def prefixed_key(prefix: str, key: str) -> str:
    prefix = normalize_prefix(prefix)
    key = key.lstrip("/")
    return f"{prefix}/{key}" if prefix else key


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_repo_url() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        detected = result.stdout.strip()
        if detected:
            return detected
    except (OSError, subprocess.CalledProcessError):
        pass
    return DEFAULT_REPO_URL


def boto3_session(args: argparse.Namespace):
    boto3, _ = require_boto3()
    kwargs: Dict[str, str] = {}
    if getattr(args, "profile", None):
        kwargs["profile_name"] = args.profile
    if getattr(args, "region", None):
        kwargs["region_name"] = args.region
    return boto3.Session(**kwargs)


def client_error_code(exc: Exception) -> str:
    return getattr(exc, "response", {}).get("Error", {}).get("Code", "")


def ensure_bucket(s3, bucket: str, region: str, client_error) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except client_error as exc:
        code = client_error_code(exc)
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

    print(f"Creating S3 bucket {bucket!r} in {region}...")
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def s3_object_exists(s3, bucket: str, key: str, client_error) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except client_error as exc:
        code = client_error_code(exc)
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_file(s3, source: Path, bucket: str, key: str, overwrite: bool, client_error) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if not overwrite and s3_object_exists(s3, bucket, key, client_error):
        print(f"Keeping existing {s3_uri(bucket, key)}. Pass --overwrite-db to replace it.")
        return
    print(f"Uploading {source} to {s3_uri(bucket, key)}...")
    s3.upload_file(str(source), bucket, key)


def upload_state(s3, bucket: str, prefix: str, state: Dict[str, Any]) -> None:
    key = prefixed_key(prefix, STATE_S3_KEY)
    body = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def resolve_ami_id(session, explicit_ami_id: Optional[str]) -> str:
    if explicit_ami_id:
        return explicit_ami_id
    ssm = session.client("ssm")
    response = ssm.get_parameter(Name=DEFAULT_AMI_PARAMETER)
    ami_id = response["Parameter"]["Value"]
    print(f"Using latest Amazon Linux 2023 AMI from SSM parameter: {ami_id}")
    return ami_id


def tag_specifications(project: str, prefix: str) -> Iterable[Dict[str, Any]]:
    tags = [
        {"Key": "Name", "Value": f"{project}-pipeline-runner"},
        {"Key": "Project", "Value": project},
        {"Key": "PipelinePrefix", "Value": prefix},
        {"Key": "ManagedBy", "Value": "code/aws/configure.py"},
    ]
    return [{"ResourceType": resource, "Tags": tags} for resource in ("instance", "volume")]


def describe_instance(ec2, instance_id: str) -> Optional[Dict[str, Any]]:
    response = ec2.describe_instances(InstanceIds=[instance_id])
    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            return instance
    return None


def existing_instance_is_reusable(ec2, state: Dict[str, Any]) -> bool:
    instance_id = state.get("instance_id")
    if not instance_id:
        return False
    try:
        instance = describe_instance(ec2, instance_id)
    except Exception:
        return False
    if not instance:
        return False
    return instance.get("State", {}).get("Name") not in {"shutting-down", "terminated"}


def create_instance(session, args: argparse.Namespace, ami_id: str) -> str:
    ec2 = session.client("ec2")
    run_args: Dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "TagSpecifications": list(tag_specifications(args.project, normalize_prefix(args.prefix))),
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": args.root_volume_gb,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                },
            }
        ],
    }
    if args.key_name:
        run_args["KeyName"] = args.key_name
    if args.subnet_id:
        run_args["SubnetId"] = args.subnet_id
    if args.security_group_id:
        run_args["SecurityGroupIds"] = [args.security_group_id]
    if args.iam_instance_profile:
        profile_key = "Arn" if args.iam_instance_profile.startswith("arn:") else "Name"
        run_args["IamInstanceProfile"] = {profile_key: args.iam_instance_profile}

    print(f"Launching {args.instance_type} instance...")
    response = ec2.run_instances(**run_args)
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"Created instance {instance_id}. Waiting until it is running...")
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    return instance_id


def load_local_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"State file not found: {path}. Run setup first, or pass --state-path.")
    return load_json(path)


def active_running_runs(s3, bucket: str, prefix: str) -> list[str]:
    runs_prefix = prefixed_key(prefix, "runs/")
    paginator = s3.get_paginator("list_objects_v2")
    running: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/status.json"):
                continue
            obj = s3.get_object(Bucket=bucket, Key=key)
            status = json.loads(obj["Body"].read().decode("utf-8"))
            if status.get("status") == "running":
                running.append(status.get("run_id") or key.split("/")[-2])
    return running


def delete_s3_prefix(s3, bucket: str, prefix: str) -> None:
    normalized = normalize_prefix(prefix)
    if not normalized:
        raise SystemExit("Refusing to delete an empty S3 prefix.")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized}/"):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})


def command_setup(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    prefix = normalize_prefix(args.prefix)
    db_key = args.db_s3_key or prefixed_key(prefix, "input/articles.db")
    repo_url = args.repo_url or default_repo_url()

    summary = {
        "region": args.region,
        "bucket": args.bucket,
        "prefix": prefix,
        "instance_type": args.instance_type,
        "ami_id": args.ami_id or f"SSM:{DEFAULT_AMI_PARAMETER}",
        "root_volume_gb": args.root_volume_gb,
        "db_s3_uri": None if args.skip_db_upload else s3_uri(args.bucket, db_key),
        "repo_url": repo_url,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run only: no AWS resources were changed.")
        return 0

    boto3, client_error = require_boto3()
    session = boto3_session(args)
    s3 = session.client("s3")
    ec2 = session.client("ec2")
    session.client("sts").get_caller_identity()

    ensure_bucket(s3, args.bucket, args.region, client_error)

    db_s3_uri = None
    if not args.skip_db_upload:
        upload_file(s3, Path(args.db_path), args.bucket, db_key, args.overwrite_db, client_error)
        db_s3_uri = s3_uri(args.bucket, db_key)

    prior_state: Dict[str, Any] = load_json(state_path) if state_path.exists() else {}
    if prior_state and existing_instance_is_reusable(ec2, prior_state):
        instance_id = prior_state["instance_id"]
        print(f"Reusing existing instance {instance_id}.")
    elif args.no_launch:
        instance_id = prior_state.get("instance_id")
        print("Skipping EC2 launch because --no-launch was set.")
    else:
        ami_id = resolve_ami_id(session, args.ami_id)
        instance_id = create_instance(session, args, ami_id)

    state = {
        **prior_state,
        "region": args.region,
        "bucket": args.bucket,
        "prefix": prefix,
        "project": args.project,
        "instance_id": instance_id,
        "instance_type": args.instance_type,
        "key_name": args.key_name,
        "security_group_id": args.security_group_id,
        "subnet_id": args.subnet_id,
        "iam_instance_profile": args.iam_instance_profile,
        "ami_id": args.ami_id,
        "root_volume_gb": args.root_volume_gb,
        "repo_url": repo_url,
        "db_s3_uri": db_s3_uri or prior_state.get("db_s3_uri"),
        "state_s3_uri": s3_uri(args.bucket, prefixed_key(prefix, STATE_S3_KEY)),
        "updated_at": utc_now(),
    }
    state.setdefault("created_at", utc_now())

    write_json(state_path, state)
    upload_state(s3, args.bucket, prefix, state)
    print(f"Wrote state to {state_path} and {state['state_s3_uri']}.")
    return 0


def command_start(args: argparse.Namespace) -> int:
    state = load_local_state(Path(args.state_path))
    args.region = args.region or state["region"]
    session = boto3_session(args)
    ec2 = session.client("ec2")
    instance_id = state["instance_id"]
    instance = describe_instance(ec2, instance_id)
    state_name = instance.get("State", {}).get("Name") if instance else "unknown"
    if state_name == "running":
        print(f"Instance {instance_id} is already running.")
        return 0
    print(f"Starting instance {instance_id}...")
    ec2.start_instances(InstanceIds=[instance_id])
    if args.wait:
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        print(f"Instance {instance_id} is running.")
    return 0


def command_pause(args: argparse.Namespace) -> int:
    state = load_local_state(Path(args.state_path))
    args.region = args.region or state["region"]
    session = boto3_session(args)
    ec2 = session.client("ec2")
    instance_id = state["instance_id"]
    print("Stopping the instance. Instance-store/NVMe scratch data will be lost; S3 data is retained.")
    ec2.stop_instances(InstanceIds=[instance_id])
    if args.wait:
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id])
        print(f"Instance {instance_id} is stopped.")
    return 0


def command_destroy(args: argparse.Namespace) -> int:
    state = load_local_state(Path(args.state_path))
    args.region = args.region or state["region"]
    session = boto3_session(args)
    s3 = session.client("s3")
    ec2 = session.client("ec2")
    bucket = state["bucket"]
    prefix = state["prefix"]
    running = active_running_runs(s3, bucket, prefix)
    if running and not args.force:
        raise SystemExit(f"Refusing to destroy while runs are marked running: {', '.join(running)}. Pass --force to override.")

    instance_id = state.get("instance_id")
    if instance_id:
        print(f"Terminating instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        if args.wait:
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
            print(f"Instance {instance_id} is terminated.")

    if args.delete_s3:
        print(f"Deleting S3 objects under {s3_uri(bucket, prefix + '/')}...")
        delete_s3_prefix(s3, bucket, prefix)
    else:
        print(f"Leaving S3 data intact under {s3_uri(bucket, prefix + '/')}.")

    state["destroyed_at"] = utc_now()
    write_json(Path(args.state_path), state)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="Local AWS runner state file.")
    parser.add_argument("--profile", default=None, help="AWS profile name.")
    parser.add_argument("--region", default=None, help="AWS region. Defaults to setup state where possible.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure AWS resources for remote pipeline execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create/verify S3 resources, upload DB, and launch/reuse EC2.")
    add_common_args(setup)
    setup.set_defaults(region=DEFAULT_REGION)
    setup.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket for inputs, state, logs, and outputs.")
    setup.add_argument("--prefix", default=DEFAULT_PREFIX, help="S3 key prefix for this project.")
    setup.add_argument("--project", default="social-science", help="Tag/name prefix for AWS resources.")
    setup.add_argument("--db-path", default="articles.db", help="Local SQLite database path to upload.")
    setup.add_argument("--db-s3-key", default=None, help="Explicit S3 key for the SQLite database.")
    setup.add_argument("--skip-db-upload", action="store_true", help="Do not upload the SQLite database during setup.")
    setup.add_argument("--overwrite-db", action="store_true", help="Replace an existing uploaded SQLite database.")
    setup.add_argument("--repo-url", default=None, help="GitHub repository URL that the EC2 worker should clone.")
    setup.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="EC2 instance type.")
    setup.add_argument("--ami-id", default=None, help="AMI ID. Defaults to latest Amazon Linux 2023 via SSM.")
    setup.add_argument("--iam-instance-profile", default=None, help="IAM instance profile name/ARN with SSM and S3 access.")
    setup.add_argument("--key-name", default=None, help="Optional EC2 key pair name.")
    setup.add_argument("--security-group-id", default=None, help="Optional security group ID.")
    setup.add_argument("--subnet-id", default=None, help="Optional subnet ID.")
    setup.add_argument("--root-volume-gb", type=int, default=DEFAULT_ROOT_VOLUME_GB, help="Root EBS volume size in GiB.")
    setup.add_argument("--no-launch", action="store_true", help="Only configure S3/upload state, do not launch EC2.")
    setup.add_argument("--dry-run", action="store_true", help="Print the intended configuration without AWS calls.")
    setup.set_defaults(func=command_setup)

    start = subparsers.add_parser("start", help="Start the configured EC2 instance.")
    add_common_args(start)
    start.add_argument("--wait", action="store_true", help="Wait until the instance reaches running state.")
    start.set_defaults(func=command_start)

    pause = subparsers.add_parser("pause", help="Stop the configured EC2 instance without deleting S3 data.")
    add_common_args(pause)
    pause.add_argument("--wait", action="store_true", help="Wait until the instance reaches stopped state.")
    pause.set_defaults(func=command_pause)

    destroy = subparsers.add_parser("destroy", help="Terminate EC2 resources and optionally delete S3 artifacts.")
    add_common_args(destroy)
    destroy.add_argument("--force", action="store_true", help="Destroy even if a run is marked running.")
    destroy.add_argument("--delete-s3", action="store_true", help="Delete objects under the configured S3 prefix.")
    destroy.add_argument("--wait", action="store_true", help="Wait until the instance reaches terminated state.")
    destroy.set_defaults(func=command_destroy)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
