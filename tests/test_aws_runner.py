import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openalex.aws import runner as aws_run


class FakePaginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        del Bucket
        contents = [
            {"Key": key, "Size": size}
            for key, size in self.objects.items()
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.downloads = []

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self.objects)

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        Path(filename).write_text(key, encoding="utf-8")


class FakeSSM:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.commands = []

    def send_command(self, **kwargs):
        self.commands.append(kwargs)
        return {"Command": {"CommandId": "command-1"}}

    def get_command_invocation(self, **_kwargs):
        return next(self.statuses)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.state = {"bucket": "bucket", "prefix": "project", "instance_id": "i-123"}
        self.run_id = "run-1"

    def test_list_artifacts_filters_checkpoint_and_supports_legacy_database(self):
        objects = {
            "project/runs/run-1/output/results.csv": 10,
            "project/runs/run-1/output/nested/matrix.npz": 20,
            "project/runs/run-1/output/events_checkpoint.pkl": 30,
            "project/runs/run-1/compiled_articles.db": 40,
        }

        artifacts = aws_run.list_run_artifacts(
            FakeS3(objects),
            self.state,
            self.run_id,
        )

        self.assertEqual(
            [artifact["relative_path"] for artifact in artifacts],
            ["compiled_articles.db", "nested/matrix.npz", "results.csv"],
        )

    def test_list_artifacts_can_include_checkpoint(self):
        objects = {
            "project/runs/run-1/output/events_checkpoint.pkl": 30,
        }

        artifacts = aws_run.list_run_artifacts(
            FakeS3(objects),
            self.state,
            self.run_id,
            include_checkpoint=True,
        )

        self.assertEqual(artifacts[0]["relative_path"], "events_checkpoint.pkl")

    def test_safe_artifact_destination_rejects_traversal(self):
        root = Path("/tmp/results")
        self.assertEqual(
            aws_run.safe_artifact_destination(root, "nested/results.csv"),
            root / "nested/results.csv",
        )
        with self.assertRaises(ValueError):
            aws_run.safe_artifact_destination(root, "../secrets")

    def test_refresh_script_collects_current_and_legacy_outputs(self):
        script = aws_run.build_artifact_refresh_script(
            self.state,
            {"scratch_dir": "/scratch"},
            self.run_id,
        )

        self.assertIn("RUN_DIR=/scratch/runs/run-1/work", script)
        self.assertIn('"$RUN_DIR/output"', script)
        self.assertIn("/scratch/compiled_articles.db", script)
        self.assertIn("--exclude events_checkpoint.pkl", script)
        self.assertIn("s3://bucket/project/runs/run-1/output/", script)

        script_with_checkpoint = aws_run.build_artifact_refresh_script(
            self.state,
            {"scratch_dir": "/scratch"},
            self.run_id,
            include_checkpoint=True,
        )
        self.assertNotIn("--exclude events_checkpoint.pkl", script_with_checkpoint)

    def test_submit_command_is_argument_safe(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--", "openalex", "events", "--min-ngram", "2"]
        )
        self.assertEqual(args.scratch_dir, "/mnt/aws-runner")
        self.assertEqual(
            aws_run.pipeline_command(args),
            "openalex events --min-ngram 2",
        )

    def test_remote_runner_uses_versioned_persistent_cache(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--", "openalex", "events"]
        )
        state = {
            **self.state,
            "repo_url": "https://example.test/repo.git",
            "db_s3_uri": "s3://bucket/input/articles.db",
        }
        script = aws_run.build_remote_runner_script(
            state=state,
            args=args,
            run_id=self.run_id,
            command=aws_run.pipeline_command(args),
        )
        self.assertIn('target="${cache_dir}/articles.db"', script)
        self.assertIn("remote_etag", script)
        self.assertIn("remote_version", script)
        self.assertIn('chmod a-w "$target"', script)
        self.assertIn('cd "$RUN_DIR"', script)

    def test_wait_for_ssm_command_returns_success(self):
        ssm = FakeSSM(
            [
                {"Status": "InProgress"},
                {"Status": "Success", "StandardOutputContent": "done"},
            ]
        )

        with mock.patch.object(aws_run.time, "sleep"):
            result = aws_run.wait_for_ssm_command(ssm, "command", "i-123", timeout=10)

        self.assertEqual(result["Status"], "Success")

    def test_wait_for_ssm_command_surfaces_remote_failure(self):
        ssm = FakeSSM(
            [
                {"Status": "Failed", "StandardErrorContent": "remote sync failed"},
            ]
        )

        with self.assertRaisesRegex(SystemExit, "remote sync failed"):
            aws_run.wait_for_ssm_command(ssm, "command", "i-123", timeout=10)

    def test_live_status_script_reads_process_storage_and_logs(self):
        script = aws_run.build_live_status_script(
            {"scratch_dir": "/scratch"},
            self.run_id,
            lines=12,
        )

        self.assertIn("WORK_DIR=/scratch/runs/run-1", script)
        self.assertIn("DATABASE_PATH=/scratch/cache/articles.db", script)
        self.assertIn("LINES=12", script)
        self.assertIn('ps -p "$pid"', script)
        self.assertIn("show_log launcher.log", script)
        self.assertIn("show_log stdout.log", script)
        self.assertIn("show_log stderr.log", script)

    def test_fetch_live_status_uses_ssm_and_returns_console_output(self):
        ssm = FakeSSM(
            [
                {
                    "Status": "Success",
                    "StandardOutputContent": "process: running\nlatest event batch",
                },
            ]
        )
        session = mock.Mock()
        session.client.return_value = ssm

        output = aws_run.fetch_live_status(
            session,
            self.state,
            {"instance_id": "i-runner", "scratch_dir": "/scratch"},
            self.run_id,
            lines=8,
            timeout=30,
        )

        self.assertIn("process: running", output)
        self.assertEqual(ssm.commands[0]["InstanceIds"], ["i-runner"])
        self.assertIn("LINES=8", ssm.commands[0]["Parameters"]["commands"][0])

    def test_download_preserves_structure(self):
        artifact = {
            "key": "project/runs/run-1/output/nested/results.csv",
            "relative_path": "nested/results.csv",
            "size": 5,
        }
        fake_s3 = FakeS3({})
        parser = aws_run.build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parser.parse_args(["download", "--output-dir", temp_dir])
            context = (self.state, fake_s3, self.run_id, {"status": "success"}, [artifact])
            with mock.patch.object(aws_run, "artifact_command_context", return_value=context):
                result = aws_run.command_download(args)

            self.assertEqual(result, 0)
            self.assertTrue((Path(temp_dir) / "nested/results.csv").is_file())

    def test_artifact_cli_defaults(self):
        args = aws_run.build_parser().parse_args(["download"])

        self.assertFalse(args.refresh)
        self.assertFalse(args.include_checkpoint)
        self.assertIsNone(args.output_dir)
        self.assertEqual(
            args.refresh_timeout_seconds,
            aws_run.DEFAULT_ARTIFACT_REFRESH_TIMEOUT_SECONDS,
        )

    def test_live_status_cli_defaults(self):
        args = aws_run.build_parser().parse_args(["status", "--live"])

        self.assertTrue(args.live)
        self.assertEqual(args.lines, 20)
        self.assertEqual(
            args.live_timeout_seconds,
            aws_run.DEFAULT_LIVE_STATUS_TIMEOUT_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
