#!/usr/bin/env python3
import sqlite3
import random


def create_articles_random_order(db_path: str, seed: int = 42,
                                 fetch_batch_size: int = 100000,
                                 insert_batch_size: int = 100000):
    """
    Create and populate the articles_order table with randomized article IDs

    Args:
        db_path: Path to your SQLite database
        seed: Random seed for consistent ordering (default: 42)
        fetch_batch_size: Number of article IDs to fetch from SQLite at a time
        insert_batch_size: Number of randomized rows to insert per executemany call
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Drop existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS articles_order")

        # Get only article IDs that can be processed by variations.py.
        cursor.execute("""
                       SELECT a.article_id
                       FROM articles a
                                JOIN abstracts ab ON a.article_id = ab.article_id
                       """)
        article_ids = []
        while True:
            rows = cursor.fetchmany(fetch_batch_size)
            if not rows:
                break
            article_ids.extend(row[0] for row in rows)

        if not article_ids:
            print("No articles found in the articles table!")
            return

        # Shuffle with fixed seed
        random.seed(seed)
        random.shuffle(article_ids)

        # Create the order table
        cursor.execute("""
                       CREATE TABLE articles_order
                       (
                           article_id  INTEGER,
                           random_rank INTEGER PRIMARY KEY
                       )
                       """)

        # Insert shuffled order
        insert_sql = "INSERT INTO articles_order (article_id, random_rank) VALUES (?, ?)"
        for start in range(0, len(article_ids), insert_batch_size):
            batch = [
                (article_id, rank)
                for rank, article_id in enumerate(
                    article_ids[start:start + insert_batch_size],
                    start + 1
                )
            ]
            cursor.executemany(insert_sql, batch)

        # Create index for faster joins
        cursor.execute("CREATE INDEX idx_articles_order_id ON articles_order(article_id)")

        conn.commit()
        print(f"Created articles_order table with {len(article_ids)} articles in random order")


# Usage example
if __name__ == "__main__":
    DB_PATH = "articles.db"  # Replace with your actual database path

    # Step 1: Create the random order table (run this once)
    create_articles_random_order(DB_PATH)
