#!/usr/bin/env python3
import argparse
import sqlite3
from urllib.parse import urlparse


INDEX_NAME = "idx_references_cites_cited"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate citation edges and create a unique index on "
            '"references" (cites, cited).'
        )
    )
    parser.add_argument(
        "--database-url",
        default="sqlite:///articles.db",
        help="Database URL. SQLite is supported directly. Default: sqlite:///articles.db",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report duplicate counts without deleting rows or creating the index.",
    )
    return parser.parse_args()


def sqlite_path_from_url(database_url):
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "", 1)
    if database_url.startswith("sqlite://"):
        parsed = urlparse(database_url)
        return parsed.path
    return None


def get_duplicate_group_count(conn):
    return conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT 1
            FROM "references"
            GROUP BY cites, cited
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]


def get_duplicate_row_count(conn):
    return conn.execute(
        """
        SELECT COALESCE(SUM(n - 1), 0)
        FROM (
            SELECT COUNT(*) AS n
            FROM "references"
            GROUP BY cites, cited
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]


def build_sqlite_reference_index(database_url, dry_run=False):
    database_path = sqlite_path_from_url(database_url)
    if database_path is None:
        raise ValueError(
            "Only SQLite URLs are supported by this script, such as sqlite:///articles.db"
        )

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        duplicate_groups = get_duplicate_group_count(conn)
        duplicate_rows = get_duplicate_row_count(conn)
        print(f"Duplicate citation pairs: {duplicate_groups}")
        print(f"Duplicate rows to remove: {duplicate_rows}")

        if dry_run:
            print("Dry run: no rows deleted and no index created.")
            return

        print("Removing duplicate reference rows, keeping the lowest id per pair.")
        conn.execute("BEGIN")
        try:
            conn.execute(
                """
                DELETE FROM "references"
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM "references"
                    GROUP BY cites, cited
                )
                """
            )
            print(f"Rows deleted: {conn.total_changes}")

            print(f"Creating unique index {INDEX_NAME}.")
            conn.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {INDEX_NAME}
                ON "references" (cites, cited)
                """
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        remaining_duplicates = get_duplicate_row_count(conn)
        print(f"Remaining duplicate rows: {remaining_duplicates}")
        print("Reference unique index is ready.")


def main():
    args = parse_args()
    build_sqlite_reference_index(args.database_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
