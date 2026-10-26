import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openalex.artifacts.cache import ArtifactCache
from openalex.artifacts.lockfile import ArtifactLock, ArtifactPin
from openalex.artifacts.manifest import (
    ArtifactValidationError,
    complete_manifest,
    validate_artifact,
)
from openalex.artifacts.pipeline import (
    STAGES,
    WEBSITE_STAGES,
    adopt_local_artifact,
    artifact_is_compatible,
)
from openalex.artifacts.resolver import ArtifactResolver
from openalex.artifacts.s3_store import S3ArtifactStore
from openalex.artifacts.specs import dependency_paths, validate_dependency


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.uploads = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key))
        self.objects[(bucket, key)] = Path(filename).read_bytes()
        self.metadata[(bucket, key)] = (ExtraArgs or {}).get("Metadata", {})

    def download_file(self, bucket, key, filename):
        if (bucket, key) not in self.objects:
            error = RuntimeError("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def head_object(self, Bucket, Key):
        try:
            payload = self.objects[(Bucket, Key)]
        except KeyError:
            error = RuntimeError("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error from None
        return {
            "ContentLength": len(payload),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }


def write_artifact(path, *, complete=True):
    path.mkdir(parents=True)
    (path / "payload.bin").write_bytes(b"payload")
    return complete_manifest(
        path,
        {"artifact_version": 1, "complete": complete},
        stage="test",
    )


def test_complete_manifest_detects_corrupt_payload(tmp_path):
    artifact = tmp_path / "artifact"
    write_artifact(artifact)
    result = validate_artifact(artifact)
    assert result.file_count == 1
    assert result.legacy is False

    (artifact / "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        validate_artifact(artifact)


def test_explicitly_incomplete_legacy_manifest_is_rejected(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps({"complete": False}), encoding="utf-8"
    )
    with pytest.raises(ArtifactValidationError, match="explicitly incomplete"):
        validate_artifact(artifact)


def test_store_repairs_partial_remote_prefix(tmp_path):
    artifact = tmp_path / "artifact"
    write_artifact(artifact)
    digest = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
    s3 = FakeS3()
    store = S3ArtifactStore(s3, "bucket", "project")
    manifest_key = f"project/artifacts/{digest}/manifest.json"
    s3.objects[("bucket", manifest_key)] = (artifact / "manifest.json").read_bytes()

    store.publish(artifact)

    keys = [key for _, key in s3.uploads]
    assert f"project/artifacts/{digest}/payload.bin" in keys
    assert keys[-1] == manifest_key


def test_resolver_hydrates_pinned_artifact_and_reuses_cache(tmp_path):
    source = tmp_path / "source"
    write_artifact(source)
    s3 = FakeS3()
    store = S3ArtifactStore(s3, "bucket", "project")
    published = store.publish(source)

    lock = ArtifactLock(tmp_path / "artifacts.lock.json")
    lock.pin(
        ArtifactPin(
            "output/test",
            published["manifest_sha256"],
            "test",
            published["s3_uri"],
        )
    )
    lock.write()
    cache = ArtifactCache(tmp_path / "cache")
    work = tmp_path / "work"
    work.mkdir()
    resolver = ArtifactResolver(cache=cache, lock=lock, store=store, base_dir=work)

    first = resolver.resolve("output/test")
    assert first.source == "cache"
    assert (work / "output/test/payload.bin").read_bytes() == b"payload"

    s3.objects.clear()
    (work / "output/test/payload.bin").unlink()
    second = resolver.resolve("output/test")
    assert second.source == "cache"
    assert (work / "output/test/payload.bin").read_bytes() == b"payload"


def test_command_dependencies_share_local_and_aws_paths():
    assert dependency_paths(
        "new-links",
        [
            "--events-dir",
            "events",
            "--clusters-dir",
            "clusters",
            "--embeddings-dir",
            "embeddings",
            "--author-embeddings-dir",
            "author_embeddings",
        ],
    ) == [
        "events",
        "clusters",
        "embeddings",
        "author_embeddings",
    ]
    assert dependency_paths(
        "visualize-cluster-topics",
        ["--topics-dir", "custom/topics"],
    ) == [
        "output/events",
        "output/event_clusters",
        "custom/topics",
        "output/author_topics",
    ]


def test_cluster_topic_dependency_requires_vectors_or_legacy_model(tmp_path):
    topics = tmp_path / "topics"
    write_artifact(topics)
    with pytest.raises(ArtifactValidationError, match="neither exported"):
        validate_dependency(
            "visualize-cluster-topics",
            ["--topics-dir", str(topics)],
            topics,
        )

    (topics / "bertopic_model").mkdir()
    validate_dependency(
        "visualize-cluster-topics",
        ["--topics-dir", str(topics)],
        topics,
    )


def test_website_pipeline_graph_and_workers_are_deterministic():
    names = [stage.name for stage in WEBSITE_STAGES]
    assert names.index("embeddings") < names.index("topics")
    assert names.index("author-embeddings") < names.index("new-links")
    assert names.index("new-links") < names.index("visualize-new-links")
    assert names[-1] == "build-website"
    assert STAGES["embeddings"].worker == "gpu"
    assert all(
        stage.worker == "cpu"
        for stage in WEBSITE_STAGES
        if stage.name != "embeddings"
    )


def test_artifact_compatibility_checks_selected_input_digests(tmp_path):
    source = tmp_path / "events"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"events")
    source_manifest = complete_manifest(
        source,
        {"artifact_version": 1},
        stage="events",
    )
    source_digest = validate_artifact(source).digest

    clusters = tmp_path / "clusters"
    clusters.mkdir()
    (clusters / "payload.bin").write_bytes(b"clusters")
    complete_manifest(
        clusters,
        {"artifact_version": 1},
        stage="cluster-events",
        inputs=[{"role": "events", "manifest_sha256": source_digest}],
    )
    validation = validate_artifact(clusters)
    assert artifact_is_compatible(
        STAGES["cluster-events"],
        validation,
        {"events": source_digest},
    )
    assert not artifact_is_compatible(
        STAGES["cluster-events"],
        validation,
        {"events": "different"},
    )
    assert source_manifest["stage"] == "events"


def test_missing_manifest_payload_is_adopted_without_recomputation(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    np.save(events / "event_vocabulary.npy", np.asarray(["topic"]))
    np.save(events / "event_document_frequency.npy", np.asarray([4]))
    from scipy import sparse

    sparse.save_npz(events / "event_cooccurrence.npz", sparse.csr_matrix([[0]]))
    incidence = events / "incidence"
    incidence.mkdir()
    sparse.save_npz(incidence / "part_00000.npz", sparse.csr_matrix([[1]]))
    np.save(incidence / "part_00000_years.npy", np.asarray([2020]))
    np.save(incidence / "part_00000_article_ids.npy", np.asarray([1]))

    validation = adopt_local_artifact(STAGES["events"], events, {})

    assert validation.manifest["adopted_local_payload"] is True
    assert validation.manifest["stage"] == "events"
    assert validation.manifest["artifact_version"] == 1
    assert validation.manifest["papers_by_year"] == {"2020": 1}
    assert validation.file_count == 6
