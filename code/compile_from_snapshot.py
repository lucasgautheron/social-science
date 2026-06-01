#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime
import gzip
import importlib
import json
import os
import re
import sys
from typing import Optional, Set


DEFAULT_DOMAIN_IDS = ["1", "2", "3", "4"]
DEFAULT_FROM_YEAR = 2015
DEFAULT_TO_PUBLICATION_DATE = "2025-12-31"
DEFAULT_LANGUAGE = "en"


def split_values(values):
    if values is None:
        return None

    split = []
    for value in values:
        split.extend(part.strip() for part in value.split(","))

    return {value for value in split if value}


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be an ISO date like 2025-10-31"
        ) from exc


def normalize_openalex_id(value, prefix=None):
    if value is None:
        return None

    value = str(value).strip().rstrip("/")
    if not value:
        return None

    if value.startswith("https://openalex.org/"):
        value = value.replace("https://openalex.org/", "", 1)

    if "/" in value:
        value = value.rsplit("/", 1)[-1]

    if prefix and value.upper().startswith(prefix.upper()):
        value = value[1:]

    return value


def normalize_id_set(values, prefix=None):
    if values is None:
        return None
    return {normalize_openalex_id(value, prefix=prefix) for value in values}


def nested_get(data, path):
    value = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def iter_snapshot_works(path):
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def record_deleted_file(path, deleted_log="deleted"):
    with open(deleted_log, "a", encoding="utf-8") as fp:
        fp.write(f"{path}\n")


def import_compile_module():
    candidate_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.getcwd(), "code"),
        os.getcwd(),
    ]
    for path in candidate_paths:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    try:
        return importlib.import_module("compile")
    except ModuleNotFoundError as exc:
        if exc.name != "compile":
            raise
        raise ModuleNotFoundError(
            "Could not import compile.py. Run this script from the repository root, "
            "or place compile.py in the same directory as compile_from_snapshot.py."
        ) from exc


@dataclass
class SnapshotFilters:
    from_year: Optional[int]
    to_year: Optional[int]
    from_publication_date: Optional[object]
    to_publication_date: Optional[object]
    languages: Optional[Set[str]]
    domain_ids: Optional[Set[str]]
    field_ids: Optional[Set[str]]
    subfield_ids: Optional[Set[str]]
    topic_ids: Optional[Set[str]]
    require_abstract: bool
    work_types: Optional[Set[str]]
    source_ids: Optional[Set[str]]

    def matches(self, work):
        if not has_required_compile_fields(work):
            return False

        publication_year = work.get("publication_year")
        if self.from_year is not None and publication_year < self.from_year:
            return False
        if self.to_year is not None and publication_year > self.to_year:
            return False

        publication_date = datetime.strptime(work["publication_date"], "%Y-%m-%d").date()
        if (
            self.from_publication_date is not None
            and publication_date < self.from_publication_date
        ):
            return False
        if (
            self.to_publication_date is not None
            and publication_date > self.to_publication_date
        ):
            return False

        if self.languages is not None and work.get("language") not in self.languages:
            return False

        if self.require_abstract and work.get("abstract_inverted_index") is None:
            return False

        if self.work_types is not None and work.get("type") not in self.work_types:
            return False

        if self.source_ids is not None:
            source_id = normalize_openalex_id(
                nested_get(work, ("primary_location", "source", "id")),
                prefix="S",
            )
            if source_id not in self.source_ids:
                return False

        primary_topic = work.get("primary_topic") or {}
        if self.domain_ids is not None:
            domain_id = normalize_openalex_id(
                nested_get(primary_topic, ("domain", "id"))
            )
            if domain_id not in self.domain_ids:
                return False

        if self.field_ids is not None:
            field_id = normalize_openalex_id(
                nested_get(primary_topic, ("field", "id"))
            )
            if field_id not in self.field_ids:
                return False

        if self.subfield_ids is not None:
            subfield_id = normalize_openalex_id(
                nested_get(primary_topic, ("subfield", "id"))
            )
            if subfield_id not in self.subfield_ids:
                return False

        if self.topic_ids is not None:
            topic_ids = {
                normalize_openalex_id(topic.get("id"), prefix="T")
                for topic in work.get("topics", [])
                if isinstance(topic, dict)
            }
            primary_topic_id = normalize_openalex_id(primary_topic.get("id"), prefix="T")
            if primary_topic_id is not None:
                topic_ids.add(primary_topic_id)
            if not topic_ids.intersection(self.topic_ids):
                return False

        return True


def has_required_compile_fields(work):
    primary_topic = work.get("primary_topic") or {}

    required_values = [
        work.get("id"),
        work.get("title"),
        work.get("publication_year"),
        work.get("publication_date"),
        nested_get(primary_topic, ("domain", "id")),
        nested_get(primary_topic, ("field", "id")),
        nested_get(primary_topic, ("subfield", "id")),
    ]
    if any(value is None for value in required_values):
        return False

    for list_field in ("locations", "topics", "referenced_works", "authorships"):
        if not isinstance(work.get(list_field), list):
            return False

    return "abstract_inverted_index" in work


def make_snapshot_compiler_class(compile_module):
    class SnapshotSQLCompiler(compile_module.OptimizedSQLCompiler):
        def configure_filters(self, filters):
            self.minimum_publication_year = filters.from_year
            self.allowed_languages = filters.languages

        def compile_snapshot_works(
            self,
            snapshot_root,
            filters,
            path_pattern=None,
            limit=None,
        ):
            self.load_processed()
            processed_records = 0

            for root, _, filenames in os.walk(snapshot_root):
                for filename in sorted(filenames):
                    if not filename.endswith(".gz"):
                        continue

                    path = os.path.join(root, filename)

                    if path_pattern and not re.search(path_pattern, path):
                        continue

                    if path in self.processed:
                        print(f"Skipping: {path}")
                        continue

                    print(f"Processing snapshot file: {path}")

                    for work in iter_snapshot_works(path):
                        if not filters.matches(work):
                            continue

                        self.add_article(work)
                        processed_records += 1

                        if limit is not None and processed_records >= limit:
                            if any(self.temp_data.values()):
                                self.flush_batch_data()
                            print(f"Reached --limit={limit}; stopping.")
                            return

                    self.temp_data["files"].append(path)
                    if any(self.temp_data.values()):
                        self.flush_batch_data()
                    os.remove(path)
                    record_deleted_file(path)
                    print(f"Deleted processed snapshot file: {path}")

            if any(self.temp_data.values()):
                self.flush_batch_data()

    return SnapshotSQLCompiler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile OpenAlex works from a local works snapshot."
    )
    parser.add_argument(
        "snapshot_root",
        help="Path to the local OpenAlex works snapshot directory.",
    )
    parser.add_argument(
        "--database-url",
        default="sqlite:///articles.db",
        help="SQLAlchemy database URL. Default: sqlite:///articles.db",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per insert batch. Default: 10000",
    )
    parser.add_argument(
        "--path-pattern",
        help="Optional regex matched against snapshot file paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after this many records match the filters.",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=DEFAULT_FROM_YEAR,
        help=f"Minimum publication year. Default: {DEFAULT_FROM_YEAR}",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        help="Maximum publication year.",
    )
    parser.add_argument(
        "--from-publication-date",
        type=parse_date,
        help="Minimum publication date, inclusive, as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-publication-date",
        type=parse_date,
        default=parse_date(DEFAULT_TO_PUBLICATION_DATE),
        help="Maximum publication date, inclusive, as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--language",
        action="append",
        help="Language code. Can be repeated or comma-separated. Default: en",
    )
    parser.add_argument(
        "--domain-id",
        action="append",
        help="OpenAlex domain ID or URL. Can be repeated or comma-separated. Default: 3",
    )
    parser.add_argument(
        "--field-id",
        action="append",
        help="OpenAlex field ID or URL. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--subfield-id",
        action="append",
        help="OpenAlex subfield ID or URL. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--topic-id",
        action="append",
        help="OpenAlex topic ID or URL. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--require-abstract",
        dest="require_abstract",
        action="store_true",
        default=True,
        help="Only import works with abstracts. Enabled by default.",
    )
    parser.add_argument(
        "--no-require-abstract",
        dest="require_abstract",
        action="store_false",
        help="Allow works without abstracts.",
    )
    parser.add_argument(
        "--work-type",
        action="append",
        help="OpenAlex work type, such as article. Can be repeated or comma-separated.",
        default=["article"],
    )
    parser.add_argument(
        "--source-id",
        action="append",
        help="OpenAlex primary source ID or URL. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--enable-references",
        dest="enable_references",
        action="store_true",
        default=True,
        help="Store citation edges from referenced_works. Enabled by default.",
    )
    parser.add_argument(
        "--disable-references",
        dest="enable_references",
        action="store_false",
        help="Skip citation edge insertion.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.to_year is not None and args.to_year < args.from_year:
        parser.error("--to-year must be greater than or equal to --from-year")
    if (
        args.from_publication_date is not None
        and args.to_publication_date is not None
        and args.to_publication_date < args.from_publication_date
    ):
        parser.error("--to-publication-date must be on or after --from-publication-date")

    return args


def build_filters(args):
    language_values = split_values(args.language) or {DEFAULT_LANGUAGE}
    domain_values = split_values(args.domain_id) or set(DEFAULT_DOMAIN_IDS)

    return SnapshotFilters(
        from_year=args.from_year,
        to_year=args.to_year,
        from_publication_date=args.from_publication_date,
        to_publication_date=args.to_publication_date,
        languages=language_values,
        domain_ids=normalize_id_set(domain_values),
        field_ids=normalize_id_set(split_values(args.field_id)),
        subfield_ids=normalize_id_set(split_values(args.subfield_id)),
        topic_ids=normalize_id_set(split_values(args.topic_id), prefix="T"),
        require_abstract=args.require_abstract,
        work_types=split_values(args.work_type),
        source_ids=normalize_id_set(split_values(args.source_id), prefix="S"),
    )


def print_stats(compiler):
    stats = compiler.get_stats()
    print("Final database statistics:")
    for table, count in stats.items():
        print(f"{table}: {count}")


def main():
    args = parse_args()
    filters = build_filters(args)

    compile_module = import_compile_module()
    compile_module.ENABLE_REFERENCES = args.enable_references
    SnapshotSQLCompiler = make_snapshot_compiler_class(compile_module)

    compiler = SnapshotSQLCompiler(args.database_url, batch_size=args.batch_size)
    compiler.configure_filters(filters)

    # compile.py refers to a module-level compiler while printing batch stats.
    compile_module.compiler = compiler

    compiler.compile_snapshot_works(
        args.snapshot_root,
        filters=filters,
        path_pattern=args.path_pattern,
        limit=args.limit,
    )
    print_stats(compiler)


if __name__ == "__main__":
    main()
