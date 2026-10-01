import json
from types import SimpleNamespace

from openalex.aws import configure
from openalex.aws.workers import (
    CPU_AMI_PARAMETER,
    GPU_AMI_PARAMETER,
    GPU_INSTANCE_TYPE,
    effective_state,
    normalize_state,
)

LEGACY_STATE = {
    "ami_id": None,
    "bucket": "lucas-epistemic-bubbles",
    "created_at": "2026-09-28T01:47:33+00:00",
    "db_s3_uri": "s3://lucas-epistemic-bubbles/social-science/input/articles.db",
    "iam_instance_profile": None,
    "instance_id": "i-01814639b39ec1d72",
    "instance_type": "i4i.8xlarge",
    "key_name": None,
    "prefix": "social-science",
    "project": "social-science",
    "region": "us-east-1",
    "repo_url": "git@github.com:lucasgautheron/social-science.git",
    "root_volume_gb": 200,
    "security_group_id": None,
    "state_s3_uri": (
        "s3://lucas-epistemic-bubbles/social-science/state/aws_runner_state.json"
    ),
    "subnet_id": None,
    "updated_at": "2026-09-28T01:47:33+00:00",
}


def test_legacy_state_migrates_without_losing_paused_cpu_instance():
    state, migrated = normalize_state(LEGACY_STATE)

    assert migrated is True
    assert state["schema_version"] == 2
    assert state["default_worker"] == "cpu"
    assert state["workers"]["cpu"]["instance_id"] == "i-01814639b39ec1d72"
    assert state["workers"]["cpu"]["instance_type"] == "i4i.8xlarge"
    assert state["workers"]["cpu"]["ami_parameter"] == CPU_AMI_PARAMETER
    assert state["instance_id"] == "i-01814639b39ec1d72"
    assert state["key_name"] is None
    assert state["iam_instance_profile"] is None


def test_gpu_profile_defaults_and_effective_state():
    state, _migrated = normalize_state(LEGACY_STATE)
    state["workers"]["gpu"] = {
        "instance_id": "i-gpu",
        "instance_type": GPU_INSTANCE_TYPE,
        "ami_parameter": GPU_AMI_PARAMETER,
    }

    selected = effective_state(state, "gpu")
    assert selected["instance_id"] == "i-gpu"
    assert selected["instance_type"] == "g6.8xlarge"
    assert selected["worker"] == "gpu"
    assert state["workers"]["cpu"]["instance_id"] == "i-01814639b39ec1d72"


def test_worker_tags_are_distinct():
    cpu = list(configure.tag_specifications("project", "prefix", "cpu"))[0]["Tags"]
    gpu = list(configure.tag_specifications("project", "prefix", "gpu"))[0]["Tags"]

    assert {"Key": "Name", "Value": "project-pipeline-runner-cpu"} in cpu
    assert {"Key": "WorkerRole", "Value": "cpu"} in cpu
    assert {"Key": "Name", "Value": "project-pipeline-runner-gpu"} in gpu
    assert {"Key": "WorkerRole", "Value": "gpu"} in gpu


def test_gpu_setup_preserves_legacy_cpu_worker(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(LEGACY_STATE), encoding="utf-8")
    uploaded = []

    class FakeSTS:
        def get_caller_identity(self):
            return {}

    class FakeSession:
        def client(self, name):
            return FakeSTS() if name == "sts" else SimpleNamespace()

    monkeypatch.setattr(configure, "require_boto3", lambda: (object(), Exception))
    monkeypatch.setattr(configure, "boto3_session", lambda _args: FakeSession())
    monkeypatch.setattr(configure, "ensure_bucket", lambda *_args: None)
    monkeypatch.setattr(configure, "resolve_ami_id", lambda *_args: "ami-gpu")
    monkeypatch.setattr(configure, "create_instance", lambda *_args: "i-gpu")
    monkeypatch.setattr(
        configure,
        "describe_instance",
        lambda *_args: {"State": {"Name": "running"}},
    )
    monkeypatch.setattr(
        configure,
        "upload_state",
        lambda _s3, _bucket, _prefix, state: uploaded.append(state),
    )
    args = configure.build_parser().parse_args(
        [
            "setup",
            "--worker",
            "gpu",
            "--state-path",
            str(state_path),
            "--skip-db-upload",
            "--repo-url",
            "https://example.test/repo.git",
        ]
    )

    assert args.func(args) == 0
    state = json.loads(state_path.read_text())
    assert state["workers"]["cpu"]["instance_id"] == "i-01814639b39ec1d72"
    assert state["workers"]["gpu"]["instance_id"] == "i-gpu"
    assert state["workers"]["gpu"]["instance_type"] == "g6.8xlarge"
    assert state["instance_id"] == "i-01814639b39ec1d72"
    assert state["bucket"] == "lucas-epistemic-bubbles"
    assert state["prefix"] == "social-science"
    assert state["project"] == "social-science"
    assert state_path.with_suffix(".json.v1.bak").is_file()
    assert uploaded[-1]["workers"]["gpu"]["instance_id"] == "i-gpu"
