import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openalex.aws import configure as aws_configure
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
        self.uploads = []

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self.objects)

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        Path(filename).write_text(key, encoding="utf-8")

    def head_object(self, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            error = RuntimeError("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {"ContentLength": self.objects[Key]}

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))
        self.objects[key] = Path(filename).stat().st_size


class FakeSSM:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.commands = []

    def send_command(self, **kwargs):
        self.commands.append(kwargs)
        return {"Command": {"CommandId": "command-1"}}

    def get_command_invocation(self, **_kwargs):
        return next(self.statuses)


class FakeIAM:
    def __init__(self):
        self.policy = None

    def get_instance_profile(self, InstanceProfileName):
        return {
            "InstanceProfile": {
                "InstanceProfileName": InstanceProfileName,
                "Roles": [{"RoleName": "worker-role"}],
            }
        }

    def put_role_policy(self, **kwargs):
        self.policy = kwargs


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

    def test_publish_input_artifact_is_content_addressed_and_manifest_last(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "output" / "events"
            artifact.mkdir(parents=True)
            (artifact / "data.npy").write_bytes(b"data")
            (artifact / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")
            s3 = FakeS3({})

            published = aws_run.publish_input_artifacts(
                s3,
                self.state,
                ["output/events"],
                base_dir=root,
            )

            digest = published[0]["manifest_sha256"]
            prefix = f"project/artifacts/{digest}"
            self.assertEqual(published[0]["path"], "output/events")
            self.assertEqual(published[0]["s3_uri"], f"s3://bucket/{prefix}/")
            self.assertEqual(
                [key for _, _, key in s3.uploads],
                [f"{prefix}/data.npy", f"{prefix}/manifest.json"],
            )

            upload_count = len(s3.uploads)
            repeated = aws_run.publish_input_artifacts(
                s3,
                self.state,
                ["output/events"],
                base_dir=root,
            )
            self.assertEqual(repeated, published)
            self.assertEqual(len(s3.uploads), upload_count)

    def test_publish_input_artifact_validates_paths_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "missing-manifest").mkdir()
            s3 = FakeS3({})

            for value in ("/absolute", "../traversal", ".", "output"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    aws_run.publish_input_artifacts(
                        s3, self.state, [value], base_dir=root
                    )
            with self.assertRaisesRegex(SystemExit, "manifest.json"):
                aws_run.publish_input_artifacts(
                    s3, self.state, ["missing-manifest"], base_dir=root
                )

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

        script_with_input = aws_run.build_artifact_refresh_script(
            self.state,
            {
                "scratch_dir": "/scratch",
                "inputs": [{"path": "output/events"}],
            },
            self.run_id,
        )
        self.assertIn("--exclude events", script_with_input)
        self.assertIn("--exclude 'events/*'", script_with_input)

    def test_submit_command_is_argument_safe(self):
        args = aws_run.build_parser().parse_args(
            [
                "submit",
                "--input",
                "output/events",
                "--input",
                "output/event_clusters",
                "--",
                "openalex",
                "events",
                "--min-ngram",
                "2",
            ]
        )
        self.assertEqual(args.scratch_dir, "/mnt/aws-runner")
        self.assertEqual(args.input, ["output/events", "output/event_clusters"])
        self.assertEqual(
            aws_run.pipeline_command(args),
            "openalex events --min-ngram 2",
        )
        self.assertEqual(args.worker, "cpu")

    def test_submit_can_select_gpu_worker(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--worker", "gpu", "--", "openalex", "embeddings"]
        )
        self.assertEqual(args.worker, "gpu")

    def test_last_run_history_is_retained_per_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last-run.json"
            aws_run.record_last_run(path, "cpu-run", "cpu")
            aws_run.record_last_run(path, "gpu-run", "gpu")
            recorded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(recorded["last_run_id"], "gpu-run")
            self.assertEqual(recorded["by_worker"]["cpu"]["run_id"], "cpu-run")
            self.assertEqual(recorded["by_worker"]["gpu"]["run_id"], "gpu-run")

            cpu_args = aws_run.build_parser().parse_args(
                [
                    "status",
                    "--worker",
                    "cpu",
                    "--last-run-path",
                    str(path),
                ]
            )
            self.assertEqual(aws_run.resolve_run_id(cpu_args), "cpu-run")

    def test_remote_runner_uses_versioned_persistent_cache(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--", "openalex", "events"]
        )
        args.prepared_inputs = [
            {
                "path": "output/events",
                "manifest_sha256": "abc123",
                "s3_uri": "s3://bucket/project/artifacts/abc123/",
            }
        ]
        state = {
            **self.state,
            "repo_url": "https://example.test/repo.git",
            "db_s3_uri": "s3://bucket/input/articles.db",
            "region": "us-east-1",
            "notification_topic_arn": "arn:aws:sns:us-east-1:123:openalex-runs",
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
        self.assertIn("python3.11", script)
        self.assertIn("sys.version_info < (3, 11)", script)
        self.assertIn('rm -rf "$VENV_DIR"', script)
        self.assertIn('"$PYTHON_BIN" -m venv "$VENV_DIR"', script)
        self.assertIn(
            "[analysis,website,parquet,aws,download,embeddings,topics]",
            script,
        )
        self.assertIn("hashlib.sha256()", script)
        self.assertNotIn("shasum", script)
        self.assertIn(
            "NOTIFICATION_TOPIC_ARN=arn:aws:sns:us-east-1:123:openalex-runs",
            script,
        )
        self.assertIn("aws sns publish", script)
        self.assertIn('notify_terminal_status "$terminal_status"', script)
        self.assertIn('trap on_error ERR', script)
        self.assertIn('cache_root="${SCRATCH_DIR}/cache/artifacts/${expected_digest}"', script)
        self.assertIn(
            "stage_input output/events abc123 s3://bucket/project/artifacts/abc123/",
            script,
        )
        self.assertIn('cp -a --reflink=auto "${cache_root}/." "$destination/"', script)
        self.assertIn("--exclude events --exclude 'events/*'", script)
        self.assertLess(
            script.index('CURRENT_STEP="staging input artifacts"'),
            script.index('bash -lc "$PIPELINE_COMMAND"'),
        )

        status = aws_run.initial_status(
            state,
            args,
            self.run_id,
            aws_run.pipeline_command(args),
            "submitted",
            "Submitted remote command",
        )
        self.assertEqual(status["inputs"], args.prepared_inputs)
        self.assertEqual(status["worker"], "cpu")

    def test_gpu_remote_runner_validates_cuda(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--worker", "gpu", "--", "openalex", "embeddings"]
        )
        args.prepared_inputs = []
        state = {
            **self.state,
            "worker": "gpu",
            "instance_type": "g6.8xlarge",
            "repo_url": "https://example.test/repo.git",
            "db_s3_uri": "s3://bucket/input/articles.db",
            "region": "us-east-1",
        }
        script = aws_run.build_remote_runner_script(
            state=state,
            args=args,
            run_id=self.run_id,
            command=aws_run.pipeline_command(args),
        )
        self.assertIn("GPU_WORKER=1", script)
        self.assertIn("nvidia-smi", script)
        self.assertIn("torch.cuda.is_available()", script)
        self.assertIn("WORKER=gpu", script)

    def test_cancel_targets_the_instance_recorded_by_the_run(self):
        args = aws_run.build_parser().parse_args(
            ["cancel", "--run-id", "gpu-run"]
        )
        state = {
            "bucket": "bucket",
            "prefix": "project",
            "region": "us-east-1",
            "default_worker": "cpu",
            "workers": {
                "cpu": {"instance_id": "i-cpu", "instance_type": "i4i.8xlarge"},
                "gpu": {"instance_id": "i-gpu", "instance_type": "g6.8xlarge"},
            },
        }
        sent = []

        class Session:
            def client(self, name):
                self.name = name
                return object()

        with (
            mock.patch.object(aws_run, "load_state", return_value=state),
            mock.patch.object(aws_run, "boto3_session", return_value=Session()),
            mock.patch.object(
                aws_run,
                "get_s3_json",
                return_value={
                    "run_id": "gpu-run",
                    "worker": "gpu",
                    "instance_id": "i-gpu",
                    "scratch_dir": "/scratch",
                },
            ),
            mock.patch.object(
                aws_run,
                "send_ssm_command",
                side_effect=lambda _session, instance_id, _script, _run_id: (
                    sent.append(instance_id) or "command-1"
                ),
            ),
            mock.patch.object(aws_run, "put_s3_json"),
            mock.patch.object(aws_run, "require_boto3", return_value=(None, Exception)),
        ):
            self.assertEqual(aws_run.command_cancel(args), 0)

        self.assertEqual(sent, ["i-gpu"])

    def test_submit_routes_to_selected_gpu_and_records_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            last_run = Path(temp_dir) / "last-run.json"
            args = aws_run.build_parser().parse_args(
                [
                    "submit",
                    "--worker",
                    "gpu",
                    "--run-id",
                    "gpu-run",
                    "--last-run-path",
                    str(last_run),
                    "--",
                    "openalex",
                    "embeddings",
                ]
            )
            state = {
                "bucket": "bucket",
                "prefix": "project",
                "region": "us-east-1",
                "repo_url": "https://example.test/repo.git",
                "db_s3_uri": "s3://bucket/input/articles.db",
                "default_worker": "cpu",
                "workers": {
                    "cpu": {
                        "instance_id": "i-cpu",
                        "instance_type": "i4i.8xlarge",
                    },
                    "gpu": {
                        "instance_id": "i-gpu",
                        "instance_type": "g6.8xlarge",
                    },
                },
            }
            started = []
            sent = []

            class Session:
                def client(self, _name):
                    return object()

            with (
                mock.patch.object(aws_run, "load_state", return_value=state),
                mock.patch.object(aws_run, "boto3_session", return_value=Session()),
                mock.patch.object(
                    aws_run,
                    "publish_input_artifacts",
                    return_value=[],
                ),
                mock.patch.object(
                    aws_run,
                    "ensure_instance_running",
                    side_effect=lambda _session, selected, **_kwargs: started.append(
                        selected["instance_id"]
                    ),
                ),
                mock.patch.object(aws_run, "put_s3_json"),
                mock.patch.object(aws_run, "put_s3_text"),
                mock.patch.object(
                    aws_run,
                    "send_ssm_command",
                    side_effect=lambda _session, instance_id, _script, _run_id: (
                        sent.append(instance_id) or "command-1"
                    ),
                ),
            ):
                self.assertEqual(aws_run.command_submit(args), 0)

            self.assertEqual(started, ["i-gpu"])
            self.assertEqual(sent, ["i-gpu"])
            history = json.loads(last_run.read_text(encoding="utf-8"))
            self.assertEqual(history["last_worker"], "gpu")
            self.assertEqual(history["by_worker"]["gpu"]["run_id"], "gpu-run")

    def test_launcher_mounts_instance_store_before_creating_run_directory(self):
        args = aws_run.build_parser().parse_args(
            ["submit", "--", "openalex", "events"]
        )
        script = aws_run.build_launcher_script("# remote runner", args, self.run_id)

        self.assertIn('grep -qi "Instance Storage"', script)
        self.assertIn("No EC2 NVMe instance-store devices were found", script)
        self.assertIn('mdadm --create "$storage_device"', script)
        self.assertIn('mkfs.xfs -f "$storage_device"', script)
        self.assertIn('mount -o noatime,nodiratime "$storage_device"', script)
        self.assertIn("Refusing to mount instance storage while pipeline pid", script)
        self.assertLess(
            script.index('mount -o noatime,nodiratime "$storage_device"'),
            script.index("cat > /mnt/aws-runner/runs/run-1/remote_runner.sh"),
        )

    def test_notification_topic_name_is_sns_compatible(self):
        name = aws_configure.notification_topic_name({"project": "OpenAlex / social science"})

        self.assertEqual(name, "OpenAlex-social-science-run-notifications")

    def test_grant_notification_publish_targets_worker_role_and_topic(self):
        iam = FakeIAM()
        topic_arn = "arn:aws:sns:us-east-1:123:openalex-runs"

        role_name = aws_configure.grant_notification_publish(
            iam,
            "arn:aws:iam::123:instance-profile/openalex-worker",
            topic_arn,
        )

        self.assertEqual(role_name, "worker-role")
        self.assertEqual(iam.policy["RoleName"], "worker-role")
        policy = json.loads(iam.policy["PolicyDocument"])
        self.assertEqual(policy["Statement"][0]["Resource"], topic_arn)

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
        self.assertIn(f"MAX_LOG_BYTES={aws_run.DEFAULT_LIVE_LOG_BYTES}", script)
        self.assertIn('tail -c "$MAX_LOG_BYTES"', script)
        self.assertIn("tr '\\r' '\\n'", script)
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
            args = parser.parse_args(["download", "--output-dir", temp_dir, "--yes"])
            context = (self.state, fake_s3, self.run_id, {"status": "success"}, [artifact])
            with mock.patch.object(aws_run, "artifact_command_context", return_value=context):
                result = aws_run.command_download(args)

            self.assertEqual(result, 0)
            self.assertTrue((Path(temp_dir) / "nested/results.csv").is_file())

    def test_download_defaults_to_output_and_confirms_existing_directory(self):
        artifact = {
            "key": "project/runs/run-1/output/events/manifest.json",
            "relative_path": "events/manifest.json",
            "size": 5,
        }
        fake_s3 = FakeS3({})
        context = (self.state, fake_s3, self.run_id, {"status": "success"}, [artifact])
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "output"
            destination.mkdir()
            args = aws_run.build_parser().parse_args(["download"])
            with (
                mock.patch.object(aws_run, "DEFAULT_DOWNLOAD_ROOT", str(destination)),
                mock.patch.object(aws_run, "artifact_command_context", return_value=context),
                mock.patch("builtins.input", return_value=""),
            ):
                result = aws_run.command_download(args)

            self.assertEqual(result, 0)
            self.assertTrue((destination / "events" / "manifest.json").is_file())

    def test_download_decline_leaves_existing_directory_unchanged(self):
        artifact = {
            "key": "project/runs/run-1/output/results.csv",
            "relative_path": "results.csv",
            "size": 5,
        }
        fake_s3 = FakeS3({})
        context = (self.state, fake_s3, self.run_id, {"status": "success"}, [artifact])
        with tempfile.TemporaryDirectory() as temp_dir:
            args = aws_run.build_parser().parse_args(
                ["download", "--output-dir", temp_dir]
            )
            with (
                mock.patch.object(aws_run, "artifact_command_context", return_value=context),
                mock.patch("builtins.input", return_value="n"),
            ):
                result = aws_run.command_download(args)

        self.assertEqual(result, 0)
        self.assertEqual(fake_s3.downloads, [])

    def test_artifact_cli_defaults(self):
        args = aws_run.build_parser().parse_args(["download"])

        self.assertFalse(args.refresh)
        self.assertFalse(args.include_checkpoint)
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.yes)
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
