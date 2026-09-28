import sqlite3
import sys
import os


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

        print(f"VACUUM completed successfully!")
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


def main():
    # You can specify your database path here
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default database path - change this to your database file
        db_path = "your_database.db"

        # Prompt user for database path if default doesn't exist
        if not os.path.exists(db_path):
            db_path = input("Enter the path to your SQLite database file: ").strip()

    print(f"Attempting to VACUUM database: {db_path}")

    # Create a backup reminder
    print("\nIMPORTANT: Consider backing up your database before running VACUUM!")
    response = input("Do you want to continue? (y/N): ").strip().lower()

    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    # Perform the VACUUM operation
    success = vacuum_sqlite_database(db_path)

    if success:
        print("\nVACUUM operation completed successfully!")
    else:
        print("\nVACUUM operation failed!")


if __name__ == "__main__":
    main()