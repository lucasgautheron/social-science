#!/usr/bin/env python3
import sqlite3
import random


def create_articles_random_order(db_path: str, seed: int = 42):
    """
    Create and populate the articles_order table with randomized article IDs

    Args:
        db_path: Path to your SQLite database
        seed: Random seed for consistent ordering (default: 42)
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Drop existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS articles_order")

        # Get all article IDs
        cursor.execute("SELECT article_id FROM articles")
        article_ids = [row[0] for row in cursor.fetchall()]

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
        for rank, article_id in enumerate(article_ids, 1):
            cursor.execute("INSERT INTO articles_order (article_id, random_rank) VALUES (?, ?)",
                           (article_id, rank))

        # Create index for faster joins
        cursor.execute("CREATE INDEX idx_articles_order_id ON articles_order(article_id)")

        conn.commit()
        print(f"Created articles_order table with {len(article_ids)} articles in random order")


# Usage example
if __name__ == "__main__":
    DB_PATH = "articles.db"  # Replace with your actual database path

    # Step 1: Create the random order table (run this once)
    create_articles_random_order(DB_PATH)
