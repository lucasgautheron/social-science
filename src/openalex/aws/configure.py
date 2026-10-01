#!/usr/bin/env python3
"""Provision and manage AWS resources for the OpenAlex pipeline runner."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .workers import (
    CPU_AMI_PARAMETER,
    CPU_INSTANCE_TYPE,
    DEFAULT_WORKER,
    GPU_AMI_PARAMETER,
    GPU_INSTANCE_TYPE,
    WORKER_CHOICES,
    normalize_state,
    profile,
    require_worker,
    set_worker,
    worker_state,
)

DEFAULT_STATE_PATH = ".aws_runner_state.json"
DEFAULT_PREFIX = "openalex"
DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "lucas-epistemic-bubbles"
DEFAULT_REPO_URL = "https://github.com/lucasgautheron/social-science.git"
DEFAULT_INSTANCE_TYPE = CPU_INSTANCE_TYPE
DEFAULT_IAM_INSTANCE_PROFILE = "openalex-ec2-runner-profile"
DEFAULT_AMI_PARAMETER = CPU_AMI_PARAMETER
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


def normalize_repo_url(repo_url: str) -> str:
    """Use HTTPS for public GitHub repos so EC2 does not need SSH keys."""
    if repo_url.startswith("git@github.com:"):
        repo_path = repo_url.removeprefix("git@github.com:")
        return f"https://github.com/{repo_path}"
    return repo_url


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
            return normalize_repo_url(detected)
    except (OSError, subprocess.CalledProcessError):
        pass
    return normalize_repo_url(DEFAULT_REPO_URL)


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


def resolve_ami_id(
    session,
    explicit_ami_id: Optional[str],
    ami_parameter: str = DEFAULT_AMI_PARAMETER,
) -> str:
    if explicit_ami_id:
        return explicit_ami_id
    ssm = session.client("ssm")
    response = ssm.get_parameter(Name=ami_parameter)
    ami_id = response["Parameter"]["Value"]
    print(f"Using AMI {ami_id} from SSM parameter {ami_parameter}.")
    return ami_id


def tag_specifications(
    project: str,
    prefix: str,
    worker: str = DEFAULT_WORKER,
) -> Iterable[Dict[str, Any]]:
    tags = [
        {"Key": "Name", "Value": f"{project}-pipeline-runner-{worker}"},
        {"Key": "Project", "Value": project},
        {"Key": "PipelinePrefix", "Value": prefix},
        {"Key": "WorkerRole", "Value": worker},
        {"Key": "ManagedBy", "Value": "openalex-aws"},
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
        "TagSpecifications": list(
            tag_specifications(
                args.project,
                normalize_prefix(args.prefix),
                args.worker,
            )
        ),
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
    return normalize_state(load_json(path))[0]


def backup_legacy_state(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    _state, migrated = normalize_state(load_json(path))
    if not migrated:
        return None
    backup = path.with_suffix(path.suffix + ".v1.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def active_running_runs(
    s3,
    bucket: str,
    prefix: str,
    *,
    instance_ids: set[str] | None = None,
) -> list[str]:
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
            if (
                status.get("status") == "running"
                and (
                    instance_ids is None
                    or status.get("instance_id") in instance_ids
                )
            ):
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


def notification_topic_name(state: Dict[str, Any]) -> str:
    """Return an SNS-compatible topic name for this project."""
    project = str(state.get("project") or "openalex")
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", project).strip("-")
    return f"{normalized or 'openalex'}-run-notifications"[:256]


def find_email_subscription(sns, topic_arn: str, email: str) -> Optional[Dict[str, Any]]:
    """Find an existing SNS email subscription, including pending subscriptions."""
    next_token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"TopicArn": topic_arn}
        if next_token:
            kwargs["NextToken"] = next_token
        response = sns.list_subscriptions_by_topic(**kwargs)
        for subscription in response.get("Subscriptions", []):
            if (
                subscription.get("Protocol") == "email"
                and subscription.get("Endpoint", "").casefold() == email.casefold()
            ):
                return subscription
        next_token = response.get("NextToken")
        if not next_token:
            return None


def grant_notification_publish(iam, instance_profile: str, topic_arn: str) -> str:
    """Allow the worker role attached through an instance profile to publish."""
    profile_name = instance_profile.rsplit("/", 1)[-1]
    response = iam.get_instance_profile(InstanceProfileName=profile_name)
    roles = response.get("InstanceProfile", {}).get("Roles", [])
    if not roles:
        raise SystemExit(f"Instance profile {profile_name!r} has no IAM role.")
    role_name = roles[0]["RoleName"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublishOpenAlexRunNotifications",
                "Effect": "Allow",
                "Action": "sns:Publish",
                "Resource": topic_arn,
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="OpenAlexRunNotifications",
        PolicyDocument=json.dumps(policy),
    )
    return role_name


def command_notifications(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    backup = backup_legacy_state(state_path)
    state = load_local_state(state_path)
    args.region = args.region or state["region"]
    session = boto3_session(args)
    sns = session.client("sns")
    iam = session.client("iam")
    s3 = session.client("s3")

    topic_arn = sns.create_topic(Name=notification_topic_name(state))["TopicArn"]
    subscription = find_email_subscription(sns, topic_arn, args.email)
    if subscription is None:
        response = sns.subscribe(
            TopicArn=topic_arn,
            Protocol="email",
            Endpoint=args.email,
            ReturnSubscriptionArn=True,
        )
        subscription_arn = response.get("SubscriptionArn", "PendingConfirmation")
        print(f"Created SNS email subscription: {subscription_arn}")
    else:
        subscription_arn = subscription.get("SubscriptionArn", "PendingConfirmation")
        print(f"Reusing SNS email subscription: {subscription_arn}")

    instance_profiles = {
        str(worker.get("iam_instance_profile"))
        for worker in state.get("workers", {}).values()
        if isinstance(worker, dict) and worker.get("iam_instance_profile")
    }
    if not instance_profiles and state.get("iam_instance_profile"):
        instance_profiles.add(str(state["iam_instance_profile"]))
    if not instance_profiles:
        raise SystemExit("No IAM instance profile is configured for any worker.")
    role_names = sorted(
        grant_notification_publish(iam, instance_profile, topic_arn)
        for instance_profile in instance_profiles
    )

    state.update(
        {
            "notification_email": args.email,
            "notification_topic_arn": topic_arn,
            "updated_at": utc_now(),
        }
    )
    write_json(state_path, state)
    upload_state(s3, state["bucket"], state["prefix"], state)
    print(f"Worker roles {', '.join(repr(name) for name in role_names)} can publish notifications.")
    if backup is not None:
        print(f"Backed up legacy state to {backup}.")
    if subscription_arn == "PendingConfirmation":
        print(f"Confirm the subscription using the email AWS sent to {args.email}.")
    else:
        print(f"Run notifications are enabled for {args.email}.")
    return 0


def command_setup(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    raw_state: Dict[str, Any] = load_json(state_path) if state_path.exists() else {}
    prior_state, migrated = normalize_state(raw_state)
    args.region = args.region or prior_state.get("region") or DEFAULT_REGION
    args.bucket = args.bucket or prior_state.get("bucket") or DEFAULT_BUCKET
    args.prefix = args.prefix or prior_state.get("prefix") or DEFAULT_PREFIX
    args.project = args.project or prior_state.get("project") or "openalex"
    for field in ("key_name", "security_group_id", "subnet_id"):
        if getattr(args, field) is None:
            setattr(args, field, prior_state.get(field))
    if args.iam_instance_profile is None:
        if "iam_instance_profile" in prior_state:
            args.iam_instance_profile = prior_state["iam_instance_profile"]
        else:
            args.iam_instance_profile = DEFAULT_IAM_INSTANCE_PROFILE
    prefix = normalize_prefix(args.prefix)
    db_key = args.db_s3_key or prefixed_key(prefix, "input/articles.db")
    repo_url = normalize_repo_url(
        args.repo_url or prior_state.get("repo_url") or default_repo_url()
    )
    worker_profile = profile(args.worker)
    args.instance_type = args.instance_type or worker_profile["instance_type"]
    ami_parameter = str(worker_profile["ami_parameter"])

    summary = {
        "worker": args.worker,
        "region": args.region,
        "bucket": args.bucket,
        "prefix": prefix,
        "instance_type": args.instance_type,
        "iam_instance_profile": args.iam_instance_profile,
        "ami_id": args.ami_id or f"SSM:{ami_parameter}",
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

    prior_worker = worker_state(prior_state, args.worker)
    if prior_worker and existing_instance_is_reusable(ec2, prior_worker):
        if (
            prior_worker.get("instance_type")
            and prior_worker["instance_type"] != args.instance_type
        ):
            raise SystemExit(
                f"The {args.worker} worker {prior_worker['instance_id']} uses "
                f"{prior_worker['instance_type']}, not {args.instance_type}. "
                f"Destroy it explicitly before changing its instance type."
            )
        instance_id = prior_worker["instance_id"]
        print(f"Reusing existing {args.worker} worker {instance_id}.")
    elif args.no_launch:
        instance_id = prior_worker.get("instance_id")
        print(f"Skipping {args.worker} EC2 launch because --no-launch was set.")
    else:
        ami_id = resolve_ami_id(session, args.ami_id, ami_parameter)
        instance_id = create_instance(session, args, ami_id)

    state = {
        **prior_state,
        "schema_version": 2,
        "default_worker": DEFAULT_WORKER,
        "region": args.region,
        "bucket": args.bucket,
        "prefix": prefix,
        "project": args.project,
        "repo_url": repo_url,
        "db_s3_uri": db_s3_uri or prior_state.get("db_s3_uri"),
        "state_s3_uri": s3_uri(args.bucket, prefixed_key(prefix, STATE_S3_KEY)),
        "updated_at": utc_now(),
    }
    state.setdefault("created_at", utc_now())
    worker = {
        **prior_worker,
        "instance_id": instance_id,
        "instance_type": args.instance_type,
        "key_name": args.key_name,
        "security_group_id": args.security_group_id,
        "subnet_id": args.subnet_id,
        "iam_instance_profile": args.iam_instance_profile,
        "ami_id": args.ami_id,
        "ami_parameter": ami_parameter,
        "root_volume_gb": args.root_volume_gb,
        "updated_at": utc_now(),
    }
    worker.setdefault("created_at", utc_now())
    if instance_id:
        instance = describe_instance(ec2, instance_id)
        worker["last_known_state"] = (
            instance.get("State", {}).get("Name") if instance else "unknown"
        )
        worker["last_known_state_at"] = utc_now()
    set_worker(state, args.worker, worker)

    if migrated:
        backup = backup_legacy_state(state_path)
        if backup is not None:
            print(f"Backed up legacy state to {backup}.")
    write_json(state_path, state)
    upload_state(s3, args.bucket, prefix, state)
    print(f"Wrote state to {state_path} and {state['state_s3_uri']}.")
    return 0


def command_start(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    backup = backup_legacy_state(state_path)
    state = load_local_state(state_path)
    args.region = args.region or state["region"]
    session = boto3_session(args)
    ec2 = session.client("ec2")
    worker = require_worker(state, args.worker)
    instance_id = worker["instance_id"]
    instance = describe_instance(ec2, instance_id)
    state_name = instance.get("State", {}).get("Name") if instance else "unknown"
    if state_name == "running":
        print(f"{args.worker} worker {instance_id} is already running.")
        _persist_worker_state(state_path, state, args.worker, worker, state_name, session)
        return 0
    print(f"Starting {args.worker} worker {instance_id}...")
    ec2.start_instances(InstanceIds=[instance_id])
    if args.wait:
        ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
        state_name = "running"
        print(f"{args.worker} worker {instance_id} is running.")
    else:
        state_name = "pending"
    _persist_worker_state(state_path, state, args.worker, worker, state_name, session)
    if backup is not None:
        print(f"Backed up legacy state to {backup}.")
    return 0


def command_pause(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    backup = backup_legacy_state(state_path)
    state = load_local_state(state_path)
    args.region = args.region or state["region"]
    session = boto3_session(args)
    ec2 = session.client("ec2")
    worker = require_worker(state, args.worker)
    instance_id = worker["instance_id"]
    print(
        f"Stopping the {args.worker} worker. "
        "The fast /mnt instance-store corpus cache will be lost; "
        "S3 data is retained and will repopulate the cache after restart."
    )
    ec2.stop_instances(InstanceIds=[instance_id])
    if args.wait:
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id])
        state_name = "stopped"
        print(f"{args.worker} worker {instance_id} is stopped.")
    else:
        state_name = "stopping"
    _persist_worker_state(state_path, state, args.worker, worker, state_name, session)
    if backup is not None:
        print(f"Backed up legacy state to {backup}.")
    return 0


def _persist_worker_state(
    state_path: Path,
    state: Dict[str, Any],
    worker_name: str,
    worker: Dict[str, Any],
    state_name: str,
    session,
) -> None:
    updated = {
        **worker,
        "last_known_state": state_name,
        "last_known_state_at": utc_now(),
        "updated_at": utc_now(),
    }
    state["updated_at"] = utc_now()
    set_worker(state, worker_name, updated)
    write_json(state_path, state)
    upload_state(session.client("s3"), state["bucket"], state["prefix"], state)


def command_workers(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    backup = backup_legacy_state(state_path)
    state = load_local_state(state_path)
    args.region = args.region or state["region"]
    session = boto3_session(args)
    ec2 = session.client("ec2")

    for worker_name in WORKER_CHOICES:
        worker = worker_state(state, worker_name)
        instance_id = worker.get("instance_id")
        if not instance_id:
            print(f"{worker_name}: not configured")
            continue
        instance = describe_instance(ec2, instance_id)
        state_name = (
            instance.get("State", {}).get("Name") if instance else "not-found"
        )
        updated = {
            **worker,
            "last_known_state": state_name,
            "last_known_state_at": utc_now(),
            "updated_at": utc_now(),
        }
        set_worker(state, worker_name, updated)
        print(
            f"{worker_name}: {instance_id} "
            f"{worker.get('instance_type', 'unknown')} {state_name}"
        )

    state["updated_at"] = utc_now()
    write_json(state_path, state)
    upload_state(
        session.client("s3"),
        state["bucket"],
        state["prefix"],
        state,
    )
    if backup is not None:
        print(f"Backed up legacy state to {backup}.")
    return 0


def command_destroy(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    backup = backup_legacy_state(state_path)
    state = load_local_state(state_path)
    args.region = args.region or state["region"]
    session = boto3_session(args)
    s3 = session.client("s3")
    ec2 = session.client("ec2")
    bucket = state["bucket"]
    prefix = state["prefix"]
    if args.delete_s3 and not args.all_workers:
        raise SystemExit("--delete-s3 requires --all-workers")

    worker_names = (
        list(WORKER_CHOICES)
        if args.all_workers
        else [args.worker]
    )
    selected = {
        worker_name: worker_state(state, worker_name)
        for worker_name in worker_names
    }
    if not args.all_workers:
        require_worker(state, args.worker)
    instance_ids = {
        str(worker["instance_id"])
        for worker in selected.values()
        if worker.get("instance_id")
    }
    running = active_running_runs(
        s3,
        bucket,
        prefix,
        instance_ids=instance_ids,
    )
    if running and not args.force:
        raise SystemExit(f"Refusing to destroy while runs are marked running: {', '.join(running)}. Pass --force to override.")

    for worker_name, worker in selected.items():
        instance_id = worker.get("instance_id")
        if not instance_id:
            continue
        print(f"Terminating {worker_name} worker {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        if args.wait:
            ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
            state_name = "terminated"
            print(f"{worker_name} worker {instance_id} is terminated.")
        else:
            state_name = "shutting-down"
        set_worker(
            state,
            worker_name,
            {
                **worker,
                "destroyed_at": utc_now(),
                "last_known_state": state_name,
                "last_known_state_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

    if args.delete_s3:
        print(f"Deleting S3 objects under {s3_uri(bucket, prefix + '/')}...")
        delete_s3_prefix(s3, bucket, prefix)
    else:
        print(f"Leaving S3 data intact under {s3_uri(bucket, prefix + '/')}.")
        state["updated_at"] = utc_now()
        upload_state(s3, bucket, prefix, state)

    if args.all_workers:
        state["destroyed_at"] = utc_now()
    write_json(state_path, state)
    if backup is not None:
        print(f"Backed up legacy state to {backup}.")
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
    setup.add_argument(
        "--bucket",
        default=None,
        help=f"S3 bucket. Existing state wins; new projects default to {DEFAULT_BUCKET}.",
    )
    setup.add_argument(
        "--prefix",
        default=None,
        help=f"S3 key prefix. Existing state wins; new projects default to {DEFAULT_PREFIX}.",
    )
    setup.add_argument(
        "--project",
        default=None,
        help="Tag/name prefix. Existing state wins; new projects default to openalex.",
    )
    setup.add_argument("--db-path", default="articles.db", help="Local SQLite database path to upload.")
    setup.add_argument("--db-s3-key", default=None, help="Explicit S3 key for the SQLite database.")
    setup.add_argument("--skip-db-upload", action="store_true", help="Do not upload the SQLite database during setup.")
    setup.add_argument("--overwrite-db", action="store_true", help="Replace an existing uploaded SQLite database.")
    setup.add_argument("--repo-url", default=None, help="GitHub repository URL that the EC2 worker should clone.")
    setup.add_argument("--worker", choices=WORKER_CHOICES, default=DEFAULT_WORKER)
    setup.add_argument(
        "--instance-type",
        default=None,
        help=(
            f"EC2 override. Defaults: cpu={DEFAULT_INSTANCE_TYPE}, "
            f"gpu={GPU_INSTANCE_TYPE}."
        ),
    )
    setup.add_argument(
        "--ami-id",
        default=None,
        help=(
            "AMI override. CPU defaults to Amazon Linux 2023; GPU defaults "
            f"to the latest DLAMI from {GPU_AMI_PARAMETER}."
        ),
    )
    setup.add_argument(
        "--iam-instance-profile",
        default=None,
        help=(
            "IAM instance profile name/ARN with SSM and S3 access. "
            f"Defaults to existing state or {DEFAULT_IAM_INSTANCE_PROFILE}."
        ),
    )
    setup.add_argument("--key-name", default=None, help="Optional EC2 key pair name.")
    setup.add_argument("--security-group-id", default=None, help="Optional security group ID.")
    setup.add_argument("--subnet-id", default=None, help="Optional subnet ID.")
    setup.add_argument("--root-volume-gb", type=int, default=DEFAULT_ROOT_VOLUME_GB, help="Root EBS volume size in GiB.")
    setup.add_argument("--no-launch", action="store_true", help="Only configure S3/upload state, do not launch EC2.")
    setup.add_argument("--dry-run", action="store_true", help="Print the intended configuration without AWS calls.")
    setup.set_defaults(func=command_setup)

    start = subparsers.add_parser("start", help="Start the configured EC2 instance.")
    add_common_args(start)
    start.add_argument("--worker", choices=WORKER_CHOICES, default=DEFAULT_WORKER)
    start.add_argument("--wait", action="store_true", help="Wait until the instance reaches running state.")
    start.set_defaults(func=command_start)

    pause = subparsers.add_parser("pause", help="Stop the configured EC2 instance without deleting S3 data.")
    add_common_args(pause)
    pause.add_argument("--worker", choices=WORKER_CHOICES, default=DEFAULT_WORKER)
    pause.add_argument("--wait", action="store_true", help="Wait until the instance reaches stopped state.")
    pause.set_defaults(func=command_pause)

    workers = subparsers.add_parser(
        "workers",
        help="List configured CPU/GPU workers and persist legacy state migration.",
    )
    add_common_args(workers)
    workers.set_defaults(func=command_workers)

    notifications = subparsers.add_parser(
        "notifications",
        help="Email run completion and failure notifications through AWS SNS.",
    )
    add_common_args(notifications)
    notifications.add_argument("--email", required=True, help="Email address to notify.")
    notifications.set_defaults(func=command_notifications)

    destroy = subparsers.add_parser("destroy", help="Terminate EC2 resources and optionally delete S3 artifacts.")
    add_common_args(destroy)
    destroy.add_argument("--worker", choices=WORKER_CHOICES, default=DEFAULT_WORKER)
    destroy.add_argument(
        "--all-workers",
        action="store_true",
        help="Terminate every configured worker.",
    )
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
