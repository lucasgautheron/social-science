import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openalex.aws import cli as aws_cli
from openalex.aws import snapshot_checkpoint, snapshot_import


class MissingObject(RuntimeError):
    def __init__(self, code="404"):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakePaginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix=""):
        del Bucket
        contents = [
            {"Key": key, "Size": len(value)}
            for key, value in self.objects.items()
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.metadata = {}
        self.events = []
        self.multipart = {}

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return FakePaginator(self.objects)

    def list_objects_v2(self, Bucket, Delimiter=None):
        del Bucket, Delimiter
        prefixes = sorted(
            {key.split("/", 1)[0] + "/" for key in self.objects if "/" in key}
        )
        return {
            "CommonPrefixes": [{"Prefix": prefix} for prefix in prefixes],
        }

    def put_object(self, Bucket, Key, Body, **kwargs):
        del Bucket
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise MissingObject("PreconditionFailed")
        self.events.append(("put", Key))
        self.objects[Key] = bytes(Body)
        return self.head_object(Bucket="", Key=Key)

    def get_object(self, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject()
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise MissingObject()
        return {
            "ContentLength": len(self.objects[Key]),
            "ETag": f'"{hashlib.md5(self.objects[Key]).hexdigest()}"',
            "Metadata": self.metadata.get(Key, {}),
        }

    def upload_file(self, filename, bucket, key):
        del bucket
        self.events.append(("upload", key))
        self.objects[key] = Path(filename).read_bytes()

    def delete_objects(self, Bucket, Delete):
        del Bucket
        for item in Delete["Objects"]:
            self.events.append(("delete", item["Key"]))
            self.objects.pop(item["Key"], None)

    def delete_object(self, Bucket, Key, IfMatch=None):
        if IfMatch is not None:
            current = self.head_object(Bucket=Bucket, Key=Key)["ETag"]
            if current != IfMatch:
                raise MissingObject("PreconditionFailed")
        del Bucket
        self.events.append(("delete", Key))
        self.objects.pop(Key, None)

    def copy(self, source, bucket, key, ExtraArgs=None):
        del bucket
        payload = self.objects[source["Key"]]
        self.events.append(("copy", source["Key"], key))
        self.objects[key] = payload
        if ExtraArgs and ExtraArgs.get("MetadataDirective") == "REPLACE":
            self.metadata[key] = ExtraArgs.get("Metadata", {})

    def copy_object(
        self,
        Bucket,
        Key,
        CopySource,
        CopySourceIfMatch=None,
        IfMatch=None,
        IfNoneMatch=None,
        Metadata=None,
        MetadataDirective=None,
    ):
        source_head = self.head_object(
            Bucket=CopySource["Bucket"],
            Key=CopySource["Key"],
        )
        if (
            CopySourceIfMatch is not None
            and source_head["ETag"] != CopySourceIfMatch
        ):
            raise MissingObject("PreconditionFailed")
        if IfMatch is not None:
            destination = self.head_object(Bucket=Bucket, Key=Key)
            if destination["ETag"] != IfMatch:
                raise MissingObject("PreconditionFailed")
        if IfNoneMatch == "*" and Key in self.objects:
            raise MissingObject("PreconditionFailed")
        self.events.append(("copy", CopySource["Key"], Key))
        self.objects[Key] = self.objects[CopySource["Key"]]
        if MetadataDirective == "REPLACE":
            self.metadata[Key] = dict(Metadata or {})

    def create_multipart_upload(self, Bucket, Key, Metadata=None):
        upload_id = f"upload-{len(self.multipart) + 1}"
        self.multipart[upload_id] = {
            "bucket": Bucket,
            "key": Key,
            "metadata": dict(Metadata or {}),
            "parts": {},
        }
        return {"UploadId": upload_id}

    def upload_part_copy(
        self,
        Bucket,
        Key,
        UploadId,
        PartNumber,
        CopySource,
        CopySourceRange,
        CopySourceIfMatch,
    ):
        del Bucket, Key
        source_head = self.head_object(
            Bucket=CopySource["Bucket"],
            Key=CopySource["Key"],
        )
        if source_head["ETag"] != CopySourceIfMatch:
            raise MissingObject("PreconditionFailed")
        bounds = CopySourceRange.removeprefix("bytes=").split("-")
        start, end = (int(value) for value in bounds)
        payload = self.objects[CopySource["Key"]][start : end + 1]
        self.multipart[UploadId]["parts"][PartNumber] = payload
        etag = f'"{hashlib.md5(payload).hexdigest()}"'
        return {"CopyPartResult": {"ETag": etag}}

    def complete_multipart_upload(
        self,
        Bucket,
        Key,
        UploadId,
        MultipartUpload,
        IfMatch=None,
        IfNoneMatch=None,
    ):
        if IfMatch is not None:
            destination = self.head_object(Bucket=Bucket, Key=Key)
            if destination["ETag"] != IfMatch:
                raise MissingObject("PreconditionFailed")
        if IfNoneMatch == "*" and Key in self.objects:
            raise MissingObject("PreconditionFailed")
        upload = self.multipart.pop(UploadId)
        self.objects[Key] = b"".join(
            upload["parts"][part["PartNumber"]]
            for part in MultipartUpload["Parts"]
        )
        self.metadata[Key] = upload["metadata"]
        self.events.append(("complete-multipart", Key))

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        del Bucket, Key
        self.multipart.pop(UploadId, None)


class SnapshotImportTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "bucket": "bucket",
            "prefix": "project",
            "db_s3_uri": "s3://bucket/project/input/articles.db",
            "default_worker": "cpu",
            "workers": {
                "cpu": {
                    "instance_id": "i-cpu",
                    "instance_type": "i4i.8xlarge",
                }
            },
        }

    def test_inventory_checkpoint_and_batches_skip_completed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "snapshot"
            (root / "first").mkdir(parents=True)
            (root / "second").mkdir()
            first = root / "first" / "part.gz"
            second = root / "second" / "part.parquet"
            ignored = root / "README"
            first.write_bytes(b"a" * 4)
            second.write_bytes(b"b" * 6)
            ignored.write_text("ignored", encoding="utf-8")
            checkpoint = Path(temp_dir) / "processed"
            checkpoint.write_text(
                f"/old/archive/{first.relative_to(root).as_posix()}\n",
                encoding="utf-8",
            )

            inventory = snapshot_import.inventory_snapshot(root)
            completed = snapshot_import.normalize_processed_paths(
                checkpoint,
                root,
                inventory,
            )
            batches = snapshot_import.build_batches(inventory, completed, 5)

        self.assertEqual(
            [item.relative_path for item in inventory],
            ["first/part.gz", "second/part.parquet"],
        )
        self.assertEqual(completed, {"first/part.gz"})
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].files[0].relative_path, "second/part.parquet")

    def test_checkpoint_rejects_unknown_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "snapshot"
            root.mkdir()
            (root / "part.gz").write_bytes(b"x")
            checkpoint = Path(temp_dir) / "processed"
            checkpoint.write_text("/old/archive/missing.gz\n", encoding="utf-8")
            inventory = snapshot_import.inventory_snapshot(root)

            with self.assertRaisesRegex(SystemExit, "does not match"):
                snapshot_import.normalize_processed_paths(
                    checkpoint,
                    root,
                    inventory,
                )

    def test_compile_arguments_reject_pipeline_managed_options(self):
        for option in (
            "--database-url=sqlite:///other.db",
            "--database-u=sqlite:///other.db",
            "--delete-sources",
            "--enable-references",
            "--limit",
            "--lim",
            "--path-pattern=.*",
        ):
            with self.subTest(option=option):
                with self.assertRaisesRegex(SystemExit, "managed"):
                    snapshot_import._strip_compile_separator(["--", option])

    def test_command_is_registered_and_preserves_compile_arguments(self):
        with mock.patch.object(snapshot_import, "main", return_value=0) as main:
            result = aws_cli.main(
                [
                    "import-snapshot",
                    "/snapshot",
                    "--dry-run",
                    "--",
                    "--from-year",
                    "2010",
                ]
            )

        self.assertEqual(result, 0)
        main.assert_called_once_with(
            ["/snapshot", "--dry-run", "--", "--from-year", "2010"]
        )

        captured = []
        with mock.patch.object(
            snapshot_import,
            "run_import",
            side_effect=lambda args: captured.append(args) or 0,
        ):
            snapshot_import.main(
                ["/snapshot", "--dry-run", "--", "--from-year", "2010"]
            )
        self.assertTrue(captured[0].dry_run)
        self.assertEqual(captured[0].compile_args, ["--from-year", "2010"])

    def test_checkpoint_command_is_registered(self):
        with mock.patch.object(snapshot_checkpoint, "main", return_value=0) as main:
            result = aws_cli.main(["checkpoint-snapshot", "job-1", "--yes"])

        self.assertEqual(result, 0)
        main.assert_called_once_with(["job-1", "--yes"])

    def test_batch_script_uses_one_locked_database_and_stable_workdir(self):
        batch = snapshot_import.SnapshotBatch(
            batch_id="00001-deadbeef",
            files=(snapshot_import.SnapshotFile("a/part.gz", 10),),
            size=10,
        )
        script = snapshot_import.build_batch_import_script(
            state=self.state,
            job_id="job-1",
            batch=batch,
            scratch_dir="/scratch",
            compile_args=["--from-year", "2010"],
            inventory_sha256="inventory",
            database_generation="generation-1",
        )

        self.assertIn('exec 9>"$JOB_DIR/import.lock"', script)
        self.assertIn("flock -n 9", script)
        self.assertIn("cd \"$WORK_DIR\"", script)
        self.assertIn(
            "sqlite:////scratch/import-jobs/job-1/articles.db",
            script,
        )
        self.assertIn("--from-year 2010", script)
        self.assertIn("Another OpenAlex run is active", script)
        self.assertIn("Missing or truncated staged file", script)
        self.assertIn("batch-results/00001-deadbeef.json", script)

    def test_checkpoint_script_locks_and_validates_sqlite_backup_before_upload(self):
        script = snapshot_checkpoint.build_checkpoint_script(
            state=self.state,
            job_id="job-1",
            checkpoint_id="checkpoint-1",
            scratch_dir="/scratch",
            quick_check=True,
        )

        self.assertIn('exec 9>"$JOB_DIR/import.lock"', script)
        self.assertIn("source.backup(destination", script)
        self.assertIn('destination.execute("PRAGMA quick_check")', script)
        self.assertIn('sha256sum "$SNAPSHOT"', script)
        self.assertIn("$PREFIX_URI/articles.db", script)
        self.assertIn("$PREFIX_URI/processed", script)
        self.assertLess(
            script.index("source.backup(destination"),
            script.index('aws s3 cp --no-progress "$SNAPSHOT"'),
        )

    def test_finish_records_completion_before_deleting_staging(self):
        batch = snapshot_import.SnapshotBatch(
            batch_id="00001-deadbeef",
            files=(snapshot_import.SnapshotFile("part.gz", 10),),
            size=10,
        )
        job = {
            "job_id": "job-1",
            "inventory_files": 1,
            "completed_files": [],
            "batches": [snapshot_import._batch_to_json(batch)],
        }
        events = []
        with (
            mock.patch.object(
                snapshot_import,
                "_save_job",
                side_effect=lambda *_args: events.append("save"),
            ),
            mock.patch.object(
                snapshot_import,
                "_delete_prefix",
                side_effect=lambda *_args: events.append("delete"),
            ),
        ):
            snapshot_import._finish_batch(
                object(),
                self.state,
                job,
                batch,
            )

        self.assertEqual(events, ["save", "delete"])
        self.assertEqual(job["completed_files"], ["part.gz"])
        self.assertEqual(job["batches"][0]["status"], "complete")

    def test_remote_completion_marker_recovers_send_before_state_save(self):
        batch = snapshot_import.SnapshotBatch(
            batch_id="00001-deadbeef",
            files=(snapshot_import.SnapshotFile("part.gz", 10),),
            size=10,
        )
        record = snapshot_import._batch_to_json(batch)
        record["status"] = "staged"
        job = {
            "job_id": "job-1",
            "inventory_sha256": "inventory",
            "database_generation": "generation-1",
            "inventory_files": 1,
            "completed_files": [],
            "batches": [record],
        }
        marker_key = snapshot_import._batch_result_key(
            self.state,
            "job-1",
            batch.batch_id,
        )
        s3 = FakeS3(
            {
                marker_key: json.dumps(
                    {
                        "batch_id": batch.batch_id,
                        "inventory_sha256": "inventory",
                        "database_generation": "generation-1",
                        "processed_sha256": "a" * 64,
                    }
                ).encode("utf-8")
            }
        )

        snapshot_import._reconcile_completion_markers(
            s3,
            self.state,
            job,
        )

        self.assertEqual(record["status"], "complete")
        self.assertEqual(job["completed_files"], ["part.gz"])

    def test_restore_generation_ignores_post_checkpoint_batch_marker(self):
        batch = snapshot_import.SnapshotBatch(
            batch_id="00001-deadbeef",
            files=(snapshot_import.SnapshotFile("part.gz", 10),),
            size=10,
        )
        record = snapshot_import._batch_to_json(batch)
        job = {
            "job_id": "job-1",
            "inventory_sha256": "inventory",
            "database_generation": "restored-generation",
            "inventory_files": 1,
            "completed_files": [],
            "batches": [record],
        }
        marker_key = snapshot_import._batch_result_key(
            self.state,
            "job-1",
            batch.batch_id,
        )
        s3 = FakeS3(
            {
                marker_key: json.dumps(
                    {
                        "batch_id": batch.batch_id,
                        "inventory_sha256": "inventory",
                        "database_generation": "pre-restore-generation",
                    }
                ).encode("utf-8")
            }
        )

        snapshot_import._reconcile_completion_markers(
            s3,
            self.state,
            job,
        )

        self.assertEqual(record["status"], "pending")
        self.assertEqual(job["completed_files"], [])

    def test_upload_batch_writes_ready_marker_last(self):
        batch = snapshot_import.SnapshotBatch(
            batch_id="00001-deadbeef",
            files=(
                snapshot_import.SnapshotFile("a.gz", 3),
                snapshot_import.SnapshotFile("nested/b.parquet", 4),
            ),
            size=7,
        )
        s3 = FakeS3()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.gz").write_bytes(b"aaa")
            (root / "nested").mkdir()
            (root / "nested" / "b.parquet").write_bytes(b"bbbb")
            snapshot_import._upload_batch(
                s3,
                self.state,
                "job-1",
                root,
                batch,
            )

        self.assertEqual(s3.events[-1][0], "put")
        self.assertTrue(s3.events[-1][1].endswith("/_READY.json"))

    def test_prefetch_window_submits_ten_before_first_import(self):
        batches = [
            snapshot_import.SnapshotBatch(
                f"{index:05d}-batch",
                (snapshot_import.SnapshotFile(f"{index}.gz", 1),),
                1,
            )
            for index in range(1, 13)
        ]
        job = {
            "job_id": "job-1",
            "inventory_files": 12,
            "initial_completed_files": [],
            "completed_files": [],
            "compile_args": [],
            "batches": [snapshot_import._batch_to_json(batch) for batch in batches],
        }
        events = []

        class Future:
            def __init__(self, function, arguments):
                self.function = function
                self.arguments = arguments

            def result(self):
                self.function(*self.arguments)

        class Executor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def submit(self, function, *arguments):
                batch = arguments[-1]
                events.append(("submit", batch.batch_id))
                return Future(function, arguments)

        args = mock.Mock(
            prefetch_batches=10,
            retry_failed=False,
            scratch_dir="/scratch",
            command_timeout_seconds=100,
            poll_seconds=0.01,
        )
        with (
            mock.patch.object(
                snapshot_import,
                "ThreadPoolExecutor",
                Executor,
            ),
            mock.patch.object(
                snapshot_import,
                "_object_exists",
                return_value=False,
            ),
            mock.patch.object(
                snapshot_import,
                "_upload_batch",
                side_effect=lambda *_args: events.append(
                    ("upload", _args[-1].batch_id)
                ),
            ),
            mock.patch.object(
                snapshot_import,
                "_import_batch",
                side_effect=lambda *_args: events.append(
                    ("import", _args[-1].batch_id)
                ),
            ),
            mock.patch.object(snapshot_import, "_save_job"),
            mock.patch.object(snapshot_import, "_reconcile_importing_batch"),
        ):
            snapshot_import._run_batches(
                args,
                mock.Mock(),
                mock.Mock(),
                self.state,
                job,
                Path("/snapshot"),
            )

        first_import = next(
            index for index, event in enumerate(events) if event[0] == "import"
        )
        self.assertEqual(
            len([event for event in events[:first_import] if event[0] == "submit"]),
            10,
        )
        eleventh_submit = events.index(("submit", batches[10].batch_id))
        self.assertGreater(eleventh_submit, first_import)

    def test_finalize_promotion_backs_up_before_replacing_database(self):
        job_id = "job-1"
        candidate_key = snapshot_import._candidate_key(self.state, job_id)
        size_key = snapshot_import._candidate_size_key(self.state, job_id)
        canonical_key = "project/input/articles.db"
        s3 = FakeS3(
            {
                candidate_key: b"new database",
                size_key: str(len(b"new database")).encode("ascii"),
                canonical_key: b"old database",
            }
        )
        job = {
            "job_id": job_id,
            "final_command_id": None,
            "completed_files": ["part.gz"],
        }
        args = mock.Mock(
            no_promote=False,
            command_timeout_seconds=100,
            poll_seconds=0.01,
        )
        session = mock.Mock()

        snapshot_import._finalize_database(
            args,
            session,
            s3,
            self.state,
            job,
        )

        backup_key = f"{canonical_key}.before-{job_id}"
        self.assertEqual(s3.objects[backup_key], b"old database")
        self.assertEqual(s3.objects[canonical_key], b"new database")
        self.assertEqual(
            s3.metadata[canonical_key]["openalex-import-job"],
            job_id,
        )
        self.assertEqual(job["status"], "success")
        self.assertLess(
            s3.events.index(("copy", canonical_key, backup_key)),
            s3.events.index(("copy", candidate_key, canonical_key)),
        )
        status_key = snapshot_import._job_status_key(self.state, job_id)
        self.assertLess(
            s3.events.index(("put", status_key)),
            s3.events.index(("delete", candidate_key)),
        )

    def test_promotion_refuses_changed_canonical_database(self):
        job_id = "job-1"
        candidate_key = snapshot_import._candidate_key(self.state, job_id)
        size_key = snapshot_import._candidate_size_key(self.state, job_id)
        canonical_key = "project/input/articles.db"
        s3 = FakeS3(
            {
                candidate_key: b"new database",
                size_key: str(len(b"new database")).encode("ascii"),
                canonical_key: b"unexpected replacement",
            }
        )
        job = {
            "job_id": job_id,
            "final_command_id": None,
            "completed_files": ["part.gz"],
            "source_db_identity": {
                "size": len(b"old database"),
                "etag": "",
                "version_id": "",
            },
        }
        args = mock.Mock(
            no_promote=False,
            command_timeout_seconds=100,
            poll_seconds=0.01,
        )

        with self.assertRaisesRegex(SystemExit, "changed during"):
            snapshot_import._finalize_database(
                args,
                mock.Mock(),
                s3,
                self.state,
                job,
            )

        self.assertEqual(s3.objects[canonical_key], b"unexpected replacement")

    def test_controller_lease_requires_explicit_takeover(self):
        s3 = FakeS3()
        first = snapshot_import._acquire_lease(
            s3,
            self.state,
            "job-1",
            steal=False,
        )
        with self.assertRaisesRegex(SystemExit, "Another controller"):
            snapshot_import._acquire_lease(
                s3,
                self.state,
                "job-1",
                steal=False,
            )

        second = snapshot_import._acquire_lease(
            s3,
            self.state,
            "job-1",
            steal=True,
        )
        self.assertNotEqual(first, second)
        snapshot_import._release_lease(
            s3,
            self.state,
            "job-1",
            first,
        )
        self.assertIn(
            snapshot_import._lease_key(self.state, "job-1"),
            s3.objects,
        )
        snapshot_import._release_lease(
            s3,
            self.state,
            "job-1",
            second,
        )
        self.assertNotIn(
            snapshot_import._lease_key(self.state, "job-1"),
            s3.objects,
        )

    def test_conditional_promotion_cannot_replace_changed_destination(self):
        s3 = FakeS3({"candidate": b"new", "canonical": b"unexpected"})
        candidate_head = s3.head_object(Bucket="bucket", Key="candidate")

        with self.assertRaises(MissingObject):
            snapshot_import._conditional_s3_copy(
                s3,
                source_bucket="bucket",
                source_key="candidate",
                destination_bucket="bucket",
                destination_key="canonical",
                source_head=candidate_head,
                destination_etag='"old-etag"',
            )

        self.assertEqual(s3.objects["canonical"], b"unexpected")

    def test_large_conditional_copy_uses_atomic_multipart_completion(self):
        s3 = FakeS3({"candidate": b"new-database", "canonical": b"old"})
        source_head = s3.head_object(Bucket="bucket", Key="candidate")
        destination_head = s3.head_object(Bucket="bucket", Key="canonical")

        with (
            mock.patch.object(snapshot_import, "SINGLE_COPY_LIMIT", 1),
            mock.patch.object(snapshot_import, "COPY_PART_SIZE", 3),
        ):
            snapshot_import._conditional_s3_copy(
                s3,
                source_bucket="bucket",
                source_key="candidate",
                destination_bucket="bucket",
                destination_key="canonical",
                source_head=source_head,
                destination_etag=destination_head["ETag"],
                metadata={"openalex-import-job": "job-1"},
            )

        self.assertEqual(s3.objects["canonical"], b"new-database")
        self.assertEqual(
            s3.metadata["canonical"]["openalex-import-job"],
            "job-1",
        )
        self.assertEqual(s3.multipart, {})

    def test_missing_first_batch_database_discards_stale_ssm_success(self):
        batch = snapshot_import.SnapshotBatch(
            "00001-batch",
            (snapshot_import.SnapshotFile("part.gz", 1),),
            1,
        )
        record = snapshot_import._batch_to_json(batch)
        record.update({"status": "importing", "command_id": "old-command"})
        job = {
            "job_id": "job-1",
            "initial_completed_files": [],
            "completed_files": [],
            "batches": [record],
            "final_command_id": None,
        }
        session = mock.Mock()
        with (
            mock.patch.object(
                snapshot_import.runner,
                "instance_state",
                return_value="stopped",
            ),
            mock.patch.object(snapshot_import, "_save_job") as save,
        ):
            snapshot_import._prepare_nvme_resume(
                mock.Mock(),
                session,
                mock.Mock(),
                self.state,
                job,
            )

        self.assertEqual(record["status"], "pending")
        self.assertIsNone(record["command_id"])
        save.assert_called_once()

    def test_missing_database_refuses_to_discard_completed_batch(self):
        batch = snapshot_import.SnapshotBatch(
            "00001-batch",
            (snapshot_import.SnapshotFile("part.gz", 1),),
            1,
        )
        record = snapshot_import._batch_to_json(batch)
        record["status"] = "complete"
        job = {
            "job_id": "job-1",
            "initial_completed_files": [],
            "completed_files": ["part.gz"],
            "batches": [record],
            "final_command_id": None,
        }
        with mock.patch.object(
            snapshot_import.runner,
            "instance_state",
            return_value="stopped",
        ):
            with self.assertRaisesRegex(SystemExit, "completed batches"):
                snapshot_import._prepare_nvme_resume(
                    mock.Mock(),
                    mock.Mock(),
                    mock.Mock(),
                    self.state,
                    job,
                )

    def test_checkpoint_restore_resets_only_post_checkpoint_batches(self):
        first = snapshot_import.SnapshotBatch(
            "00001-first",
            (snapshot_import.SnapshotFile("first.gz", 1),),
            1,
        )
        second = snapshot_import.SnapshotBatch(
            "00002-second",
            (snapshot_import.SnapshotFile("second.gz", 1),),
            1,
        )
        first_record = snapshot_import._batch_to_json(first)
        first_record["status"] = "complete"
        second_record = snapshot_import._batch_to_json(second)
        second_record["status"] = "complete"
        checkpoint_prefix = "project/import-jobs/job-1/checkpoints/checkpoint-1"
        checkpoint = {
            "checkpoint_id": "checkpoint-1",
            "inventory_sha256": "inventory",
            "compile_args": [],
            "database_size": 10,
            "database_sha256": "a" * 64,
            "processed_sha256": "b" * 64,
            "database_s3_uri": f"s3://bucket/{checkpoint_prefix}/articles.db",
            "processed_s3_uri": f"s3://bucket/{checkpoint_prefix}/processed",
            "completed_files": ["first.gz"],
        }
        job = {
            "job_id": "job-1",
            "inventory_sha256": "inventory",
            "compile_args": [],
            "initial_completed_files": [],
            "completed_files": ["first.gz", "second.gz"],
            "batches": [first_record, second_record],
            "latest_checkpoint": checkpoint,
            "final_command_id": "stale-final",
        }
        s3 = FakeS3(
            {
                f"{checkpoint_prefix}/articles.db": b"x" * 10,
                f"{checkpoint_prefix}/processed": b"first\n",
            }
        )

        restored = snapshot_import._prepare_checkpoint_restore(
            s3,
            self.state,
            job,
        )

        self.assertTrue(restored)
        self.assertEqual(job["completed_files"], ["first.gz"])
        self.assertEqual(first_record["status"], "complete")
        self.assertEqual(second_record["status"], "pending")
        self.assertEqual(job["status"], "restoring")
        self.assertIsNone(job["final_command_id"])

    def test_restore_command_installs_database_and_processed_before_marker(self):
        command = snapshot_import.build_bootstrap_command(
            "job-1",
            "bootstrap-1",
            "/scratch",
            allow_create=True,
            checkpoint_database_uri="s3://bucket/checkpoint/articles.db",
            checkpoint_processed_uri="s3://bucket/checkpoint/processed",
            checkpoint_database_size=10,
            checkpoint_id="checkpoint-1",
            checkpoint_database_sha256="a" * 64,
            checkpoint_processed_sha256="b" * 64,
        )

        marker_write = command.index('printf \'%s\\n\' "$CHECKPOINT_ID"')
        self.assertLess(
            command.index('mv "$DATABASE.download" "$DATABASE"'),
            marker_write,
        )
        self.assertLess(
            command.index(
                'mv "$WORK_DIR/processed.download" "$WORK_DIR/processed"'
            ),
            marker_write,
        )
        self.assertIn('"$DATABASE.download" | sha256sum -c -', command)
        self.assertIn('"$WORK_DIR/processed.download" | sha256sum -c -', command)

    def test_finalize_checkpoint_records_manifest_before_pruning_previous(self):
        prefix = "project/import-jobs/job-1/checkpoints/checkpoint-2"
        previous_prefix = "project/import-jobs/job-1/checkpoints/checkpoint-1"
        s3 = FakeS3(
            {
                f"{prefix}/articles.db": b"database",
                f"{prefix}/processed": b"done\n",
                f"{prefix}/articles.db.size": b"8",
                f"{prefix}/articles.db.sha256": b"a" * 64,
                f"{prefix}/processed.sha256": b"b" * 64,
                f"{previous_prefix}/articles.db": b"old",
            }
        )
        job = {
            "job_id": "job-1",
            "inventory_sha256": "inventory",
            "compile_args": [],
            "database_generation": "generation-1",
            "status": "checkpointing",
            "latest_checkpoint": {"prefix": previous_prefix},
        }
        checkpoint = {
            "checkpoint_id": "checkpoint-2",
            "created_at": "2026-11-05T00:00:00+00:00",
            "completed_files": ["first.gz"],
            "prior_status": "running",
        }

        snapshot_checkpoint._finalize_checkpoint(
            s3,
            self.state,
            job,
            checkpoint,
            keep_previous=False,
        )

        manifest_key = f"{prefix}/manifest.json"
        status_key = snapshot_import._job_status_key(self.state, "job-1")
        self.assertIn(manifest_key, s3.objects)
        self.assertEqual(job["latest_checkpoint"]["database_size"], 8)
        self.assertNotIn(f"{previous_prefix}/articles.db", s3.objects)
        self.assertLess(
            s3.events.index(("put", status_key)),
            s3.events.index(("delete", f"{previous_prefix}/articles.db")),
        )

    def test_dry_run_never_loads_aws_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "part.gz").write_bytes(b"x")
            args = snapshot_import.build_parser().parse_args(
                [str(root), "--dry-run"]
            )
            with mock.patch.object(
                snapshot_import,
                "_load_or_restore_project_state",
            ) as load_state:
                result = snapshot_import.run_import(args)

        self.assertEqual(result, 0)
        load_state.assert_not_called()

    def test_cost_confirmation_precedes_aws_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "part.gz").write_bytes(b"x")
            args = snapshot_import.build_parser().parse_args([str(root)])
            with mock.patch.object(
                snapshot_import,
                "_load_or_restore_project_state",
            ) as load_state:
                with self.assertRaisesRegex(SystemExit, "without --yes"):
                    snapshot_import.run_import(args)

        load_state.assert_not_called()

    def test_new_job_state_preserves_initial_completed_files(self):
        files = [
            snapshot_import.SnapshotFile("a.gz", 1),
            snapshot_import.SnapshotFile("b.gz", 2),
        ]
        batches = snapshot_import.build_batches(files, {"a.gz"}, 10)
        job = snapshot_import._new_job_state(
            job_id="job-1",
            state=self.state,
            root=Path("/snapshot"),
            inventory=files,
            initial_completed={"a.gz"},
            batches=batches,
            compile_args=["--from-year", "2010"],
            batch_bytes=10,
            prefetch_batches=10,
        )

        self.assertEqual(job["initial_completed_files"], ["a.gz"])
        self.assertEqual(job["completed_files"], ["a.gz"])
        self.assertEqual(job["batches"][0]["files"][0]["path"], "b.gz")

    def test_job_payload_is_json_serializable(self):
        batch = snapshot_import.SnapshotBatch(
            "00001-deadbeef",
            (snapshot_import.SnapshotFile("part.gz", 1),),
            1,
        )
        job = snapshot_import._new_job_state(
            job_id="job-1",
            state=self.state,
            root=Path("/snapshot"),
            inventory=list(batch.files),
            initial_completed=set(),
            batches=[batch],
            compile_args=[],
            batch_bytes=1,
            prefetch_batches=10,
        )

        json.dumps(job)


if __name__ == "__main__":
    unittest.main()
