"""Shared CPU/GPU worker profiles and state migration helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

STATE_SCHEMA_VERSION = 2
DEFAULT_WORKER = "cpu"
WORKER_CHOICES = ("cpu", "gpu")

CPU_INSTANCE_TYPE = "i4i.8xlarge"
GPU_INSTANCE_TYPE = "g6.8xlarge"
CPU_AMI_PARAMETER = (
    "/aws/service/ami-amazon-linux-latest/"
    "al2023-ami-kernel-default-x86_64"
)
GPU_AMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
)

WORKER_PROFILES = {
    "cpu": {
        "instance_type": CPU_INSTANCE_TYPE,
        "ami_parameter": CPU_AMI_PARAMETER,
        "gpu": False,
    },
    "gpu": {
        "instance_type": GPU_INSTANCE_TYPE,
        "ami_parameter": GPU_AMI_PARAMETER,
        "gpu": True,
    },
}

WORKER_STATE_FIELDS = (
    "instance_id",
    "instance_type",
    "key_name",
    "security_group_id",
    "subnet_id",
    "iam_instance_profile",
    "ami_id",
    "ami_parameter",
    "root_volume_gb",
    "created_at",
    "updated_at",
    "last_known_state",
    "last_known_state_at",
)

MIRRORED_FIELDS = (
    "instance_id",
    "instance_type",
    "key_name",
    "security_group_id",
    "subnet_id",
    "iam_instance_profile",
    "ami_id",
    "root_volume_gb",
)


def normalize_state(value: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Return schema-v2 state without discarding a legacy CPU instance."""
    state = deepcopy(value)
    migrated = int(state.get("schema_version", 1)) < STATE_SCHEMA_VERSION
    workers = state.get("workers")
    if not isinstance(workers, dict):
        workers = {}
        state["workers"] = workers
        migrated = True
    if state.get("instance_id") and "cpu" not in workers:
        workers["cpu"] = {
            key: state.get(key)
            for key in WORKER_STATE_FIELDS
            if key in state
        }
        workers["cpu"].setdefault("ami_parameter", CPU_AMI_PARAMETER)
        migrated = True
    state["schema_version"] = STATE_SCHEMA_VERSION
    state.setdefault("default_worker", DEFAULT_WORKER)
    _mirror_default_worker(state)
    return state, migrated


def worker_state(state: Dict[str, Any], worker: str) -> Dict[str, Any]:
    if worker not in WORKER_CHOICES:
        raise ValueError(f"Unknown worker {worker!r}")
    workers = state.get("workers")
    if not isinstance(workers, dict):
        return {}
    value = workers.get(worker)
    return value if isinstance(value, dict) else {}


def require_worker(state: Dict[str, Any], worker: str) -> Dict[str, Any]:
    value = worker_state(state, worker)
    if not value.get("instance_id"):
        raise SystemExit(
            f"No {worker} worker is configured. Run "
            f"`openalex-aws setup --worker {worker}` first."
        )
    return value


def set_worker(
    state: Dict[str, Any],
    worker: str,
    value: Dict[str, Any],
) -> Dict[str, Any]:
    if worker not in WORKER_CHOICES:
        raise ValueError(f"Unknown worker {worker!r}")
    state.setdefault("workers", {})[worker] = dict(value)
    state["schema_version"] = STATE_SCHEMA_VERSION
    state.setdefault("default_worker", DEFAULT_WORKER)
    _mirror_default_worker(state)
    return state


def effective_state(state: Dict[str, Any], worker: str) -> Dict[str, Any]:
    """Overlay one worker onto shared project state for existing runner helpers."""
    selected = require_worker(state, worker)
    return {**state, **selected, "worker": worker}


def profile(worker: str) -> Dict[str, Any]:
    if worker not in WORKER_PROFILES:
        raise ValueError(f"Unknown worker {worker!r}")
    return WORKER_PROFILES[worker]


def _mirror_default_worker(state: Dict[str, Any]) -> None:
    selected = worker_state(state, str(state.get("default_worker", DEFAULT_WORKER)))
    for key in MIRRORED_FIELDS:
        if key in selected:
            state[key] = selected[key]
        elif key in state and state.get("workers"):
            state.pop(key, None)
