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


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


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


def test_managed_instance_profile_has_ssm_and_scoped_s3_access():
    class FakeIAM:
        def __init__(self):
            self.attached_policy = None
            self.inline_policy = None
            self.added_role = None

        def get_instance_profile(self, **_kwargs):
            raise FakeClientError("NoSuchEntity")

        def get_role(self, **_kwargs):
            raise FakeClientError("NoSuchEntity")

        def create_role(self, **_kwargs):
            return {}

        def attach_role_policy(self, **kwargs):
            self.attached_policy = kwargs

        def put_role_policy(self, **kwargs):
            self.inline_policy = kwargs

        def create_instance_profile(self, **_kwargs):
            return {"InstanceProfile": {"Roles": []}}

        def add_role_to_instance_profile(self, **kwargs):
            self.added_role = kwargs

    iam = FakeIAM()

    profile_name = configure.ensure_runner_instance_profile(
        iam,
        configure.DEFAULT_IAM_INSTANCE_PROFILE,
        "bucket",
        "project",
        FakeClientError,
    )

    assert profile_name == configure.DEFAULT_IAM_INSTANCE_PROFILE
    assert iam.attached_policy["PolicyArn"].endswith(
        "/AmazonSSMManagedInstanceCore"
    )
    policy = json.loads(iam.inline_policy["PolicyDocument"])
    object_access = next(
        statement
        for statement in policy["Statement"]
        if statement["Sid"] == "ReadWriteProjectObjects"
    )
    assert object_access["Resource"] == "arn:aws:s3:::bucket/project/*"
    assert iam.added_role["RoleName"] == configure.DEFAULT_IAM_ROLE


def test_instance_profile_is_attached_to_reused_worker():
    class FakeEC2:
        def __init__(self):
            self.attachment = None

        def describe_iam_instance_profile_associations(self, **_kwargs):
            return {"IamInstanceProfileAssociations": []}

        def associate_iam_instance_profile(self, **kwargs):
            self.attachment = kwargs

    ec2 = FakeEC2()
    configure.ensure_instance_profile_attachment(
        ec2,
        "i-gpu",
        configure.DEFAULT_IAM_INSTANCE_PROFILE,
    )

    assert ec2.attachment == {
        "InstanceId": "i-gpu",
        "IamInstanceProfile": {
            "Name": configure.DEFAULT_IAM_INSTANCE_PROFILE
        },
    }


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
    monkeypatch.setattr(
        configure,
        "ensure_runner_instance_profile",
        lambda *_args: configure.DEFAULT_IAM_INSTANCE_PROFILE,
    )
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
    assert state["workers"]["gpu"]["iam_instance_profile"] == (
        configure.DEFAULT_IAM_INSTANCE_PROFILE
    )
    assert state["instance_id"] == "i-01814639b39ec1d72"
    assert state["bucket"] == "lucas-epistemic-bubbles"
    assert state["prefix"] == "social-science"
    assert state["project"] == "social-science"
    assert state_path.with_suffix(".json.v1.bak").is_file()
    assert uploaded[-1]["workers"]["gpu"]["instance_id"] == "i-gpu"


def test_gpu_setup_repairs_profile_on_existing_worker(tmp_path, monkeypatch):
    state, _migrated = normalize_state(LEGACY_STATE)
    state["workers"]["gpu"] = {
        "instance_id": "i-existing-gpu",
        "instance_type": GPU_INSTANCE_TYPE,
        "iam_instance_profile": None,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    attached = []

    class FakeSTS:
        def get_caller_identity(self):
            return {}

    class FakeSession:
        def client(self, name):
            return FakeSTS() if name == "sts" else SimpleNamespace()

    monkeypatch.setattr(configure, "require_boto3", lambda: (object(), Exception))
    monkeypatch.setattr(configure, "boto3_session", lambda _args: FakeSession())
    monkeypatch.setattr(configure, "ensure_bucket", lambda *_args: None)
    monkeypatch.setattr(
        configure,
        "ensure_runner_instance_profile",
        lambda *_args: configure.DEFAULT_IAM_INSTANCE_PROFILE,
    )
    monkeypatch.setattr(
        configure, "existing_instance_is_reusable", lambda *_args: True
    )
    monkeypatch.setattr(
        configure,
        "ensure_instance_profile_attachment",
        lambda _ec2, instance_id, profile: attached.append(
            (instance_id, profile)
        ),
    )
    monkeypatch.setattr(
        configure,
        "describe_instance",
        lambda *_args: {"State": {"Name": "running"}},
    )
    monkeypatch.setattr(configure, "upload_state", lambda *_args: None)
    args = configure.build_parser().parse_args(
        [
            "setup",
            "--worker",
            "gpu",
            "--state-path",
            str(state_path),
            "--skip-db-upload",
        ]
    )

    assert args.func(args) == 0
    assert attached == [
        ("i-existing-gpu", configure.DEFAULT_IAM_INSTANCE_PROFILE)
    ]
    repaired = json.loads(state_path.read_text())
    assert repaired["workers"]["gpu"]["iam_instance_profile"] == (
        configure.DEFAULT_IAM_INSTANCE_PROFILE
    )
