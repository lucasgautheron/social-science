import argparse
import os
import sqlite3
from collections.abc import Sequence


def vacuum_sqlite_database(db_path):
    """
    VACUUM all tables in a SQLite database.

    Args:
        db_path (str): Path to the SQLite database file
    """
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return False

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            conn.close()
            return True

        print(f"Found {len(tables)} table(s) in the database.")

        # VACUUM each table individually (though VACUUM works on the entire database)
        # Note: SQLite's VACUUM command works on the entire database, not individual tables
        print("Running VACUUM on the database...")
        cursor.execute("VACUUM;")

        # Get database size before and after for comparison
        cursor.execute("PRAGMA page_count;")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size;")
        page_size = cursor.fetchone()[0]
        db_size = (page_count * page_size) / (1024 * 1024)  # Size in MB

        print("VACUUM completed successfully!")
        print(f"Database size: {db_size:.2f} MB")
        print(f"Tables processed: {', '.join([table[0] for table in tables])}")

        conn.close()
        return True

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VACUUM an explicit writable SQLite database.")
    parser.add_argument("db_path")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive rewrite of the target database.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.yes:
        raise SystemExit("Refusing to VACUUM without --yes")
    return 0 if vacuum_sqlite_database(args.db_path) else 2


if __name__ == "__main__":
    raise SystemExit(main())