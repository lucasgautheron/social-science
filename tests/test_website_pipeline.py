from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from openalex.artifacts.pipeline import STAGES
from openalex.aws import cli as aws_cli
from openalex.aws import website


class FakeSession:
    def __init__(self):
        self.s3 = object()

    def client(self, name):
        assert name == "s3"
        return self.s3


def _args(*values):
    return website.build_parser().parse_args(
        ["--commit", "abc123", "--poll-seconds", "0.01", *values]
    )


def _pipeline_mocks(tmp_path):
    session = FakeSession()
    return (
        mock.patch.object(
            website.runner,
            "load_state",
            return_value={"bucket": "bucket", "prefix": "project"},
        ),
        mock.patch.object(website.runner, "boto3_session", return_value=session),
        mock.patch.object(
            website.runner,
            "require_boto3",
            return_value=(object(), RuntimeError),
        ),
        mock.patch.object(
            website,
            "_select_or_publish",
            side_effect=lambda stage, *_args, **_kwargs: f"digest-{stage.name}",
        ),
        mock.patch.object(
            website,
            "_download_website",
            return_value=tmp_path / "output" / "website",
        ),
    )


def test_parser_defaults_to_deploy_and_accepts_no_deploy():
    assert _args().deploy is True
    assert _args("--no-deploy").deploy is False


def test_default_commit_rejects_dirty_or_unpushed_code():
    with mock.patch.object(
        website.subprocess,
        "check_output",
        return_value=" M src/openalex/example.py\n",
    ):
        with pytest.raises(SystemExit, match="uncommitted changes"):
            website._current_commit()

    with mock.patch.object(
        website.subprocess,
        "check_output",
        side_effect=["", "abc123\n", ""],
    ):
        with pytest.raises(SystemExit, match="not present on a tracked remote"):
            website._current_commit()


def test_website_pipeline_is_registered_in_aws_cli():
    with mock.patch.object(website, "main", return_value=0) as main:
        assert aws_cli.main(["website", "--no-deploy"]) == 0
    main.assert_called_once_with(["--no-deploy"])


def test_pipeline_reuses_artifacts_and_only_runs_final_website(tmp_path):
    patches = _pipeline_mocks(tmp_path)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4] as download,
        mock.patch.object(website, "_submit_stage") as submit,
        mock.patch.object(
            website,
            "_wait_for_run",
            return_value={"status": "success"},
        ),
        mock.patch.object(website, "_pause_worker") as pause,
        mock.patch.object(website, "_publish_website") as publish,
    ):
        result = website.run_website_pipeline(_args("--no-deploy"))

    assert result == 0
    assert submit.call_count == 1
    assert submit.call_args.args[0].name == "build-website"
    download.assert_called_once()
    publish.assert_not_called()
    pause.assert_called_once_with(mock.ANY, "cpu")


def test_forced_gpu_stage_is_routed_pinned_and_paused(tmp_path):
    patches = _pipeline_mocks(tmp_path)
    pin = SimpleNamespace(manifest_sha256="rebuilt-embeddings")
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        mock.patch.object(website, "_submit_stage") as submit,
        mock.patch.object(
            website,
            "_wait_for_run",
            return_value={"status": "success"},
        ),
        mock.patch.object(
            website.runner,
            "pin_run_artifact",
            return_value=pin,
        ) as pin_run,
        mock.patch.object(website, "_pause_worker") as pause,
        mock.patch.object(website, "_publish_website"),
    ):
        website.run_website_pipeline(_args("--force", "embeddings"))

    assert [call.args[0].name for call in submit.call_args_list] == [
        "embeddings",
        "build-website",
    ]
    assert submit.call_args_list[0].args[0].worker == "gpu"
    assert pin_run.call_args.args[3] == STAGES["embeddings"].output
    assert [call.args[1] for call in pause.call_args_list] == ["gpu", "cpu"]


def test_failed_stage_prints_logs_and_still_pauses_worker(tmp_path):
    patches = _pipeline_mocks(tmp_path)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        mock.patch.object(website, "_submit_stage"),
        mock.patch.object(
            website,
            "_wait_for_run",
            return_value={"status": "failed", "message": "boom"},
        ),
        mock.patch.object(website, "_show_failure_logs") as logs,
        mock.patch.object(website, "_pause_worker") as pause,
    ):
        with pytest.raises(SystemExit, match="embeddings failed: boom"):
            website.run_website_pipeline(_args("--force", "embeddings"))

    logs.assert_called_once()
    pause.assert_called_once_with(mock.ANY, "gpu")


def test_submit_stage_routes_explicit_inputs_and_recipe():
    args = _args("--no-deploy")
    with mock.patch.object(website.runner, "main") as run:
        website._submit_stage(STAGES["new-links"], args, "run-1")

    values = run.call_args.args[0]
    assert values[values.index("--worker") + 1] == "cpu"
    assert values.count("--input") == 4
    assert "--no-auto-inputs" in values
    assert any(
        value.startswith("OPENALEX_ARTIFACT_RECIPE_SHA256=")
        for value in values
    )


def test_wait_for_run_has_a_deadline():
    with (
        mock.patch.object(
            website.runner,
            "get_s3_json",
            return_value={"status": "running"},
        ),
        mock.patch.object(
            website.time,
            "monotonic",
            side_effect=[0.0, 2.0],
        ),
    ):
        with pytest.raises(SystemExit, match="exceeded the stage timeout"):
            website._wait_for_run(
                object(),
                {"bucket": "bucket", "prefix": "project"},
                RuntimeError,
                "run-1",
                1.0,
                1.0,
            )


def test_publish_website_uses_existing_site_directory(tmp_path):
    site = tmp_path / "website"
    args = _args()
    with mock.patch.object(website.subprocess, "run") as run:
        website._publish_website(args, site)

    command = run.call_args.args[0]
    assert command[0] == "bash"
    assert command[-2:] == [str(site), "origin"]
    assert run.call_args.kwargs["check"] is True


def test_download_replaces_existing_site_atomically(tmp_path):
    output = tmp_path / "output"
    existing = output / "website"
    existing.mkdir(parents=True)
    (existing / "stale.html").write_text("stale", encoding="utf-8")
    args = _args("--output-dir", str(output))

    def download(values):
        destination = Path(values[values.index("--output-dir") + 1]) / "website"
        destination.mkdir(parents=True)
        for relative in website.REQUIRED_WEBSITE_FILES:
            (destination / relative).write_text("fresh", encoding="utf-8")

    with mock.patch.object(website.runner, "main", side_effect=download):
        result = website._download_website(args, "run-1")

    assert result == existing
    assert (result / "index.html").read_text(encoding="utf-8") == "fresh"
    assert not (result / "stale.html").exists()


def test_download_keeps_previous_site_when_remote_site_is_incomplete(tmp_path):
    output = tmp_path / "output"
    existing = output / "website"
    existing.mkdir(parents=True)
    (existing / "index.html").write_text("previous", encoding="utf-8")
    args = _args("--output-dir", str(output))

    def download(values):
        destination = Path(values[values.index("--output-dir") + 1]) / "website"
        destination.mkdir(parents=True)
        (destination / "index.html").write_text("incomplete", encoding="utf-8")

    with mock.patch.object(website.runner, "main", side_effect=download):
        with pytest.raises(SystemExit, match="missing clusters.html"):
            website._download_website(args, "run-1")

    assert (existing / "index.html").read_text(encoding="utf-8") == "previous"
