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
    return parser.parse_args()


def sqlite_path_from_url(database_url):
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "", 1)
    if database_url.startswith("sqlite://"):
        parsed = urlparse(database_url)
        return parsed.path
    return None


def build_sqlite_reference_index(database_url):
    database_path = sqlite_path_from_url(database_url)
    if database_path is None:
        raise ValueError(
            "Only SQLite URLs are supported by this script, such as sqlite:///articles.db"
        )

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

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

        print("Reference unique index is ready.")


def main():
    args = parse_args()
    build_sqlite_reference_index(args.database_url)


if __name__ == "__main__":
    main()
