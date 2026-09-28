import argparse
import logging
import multiprocessing as mp
import pickle
import sqlite3
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextEmbeddingProcessor:
    def __init__(self, embeddings_db_path="embeddings.db", source_db_url=None,
                 batch_size=1000, n_workers=None, use_multiprocessing=True):
        self.embeddings_db_path = embeddings_db_path
        self.source_db_url = source_db_url
        self.batch_size = batch_size
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.use_multiprocessing = use_multiprocessing
        self.embedding_dim = 384  # Dimension for paraphrase-multilingual-MiniLM-L12-v2

        # Store database URL (don't create engine in __init__)
        self.source_db_url = source_db_url
        self.source_db_url = source_db_url

        # Configure PyTorch threads for better CPU utilization
        if not use_multiprocessing:
            torch.set_num_threads(mp.cpu_count())

        # Initialize model (will be recreated in each process if using multiprocessing)
        if not use_multiprocessing:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Initialize embeddings database
        self._init_database()

        # Cursor-based pagination state
        self.last_random_rank = None

    def _init_database(self):
        """Initialize SQLite database with embeddings table"""
        Path(self.embeddings_db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS embeddings
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY,
                           article_id
                           TEXT
                           UNIQUE,
                           embedding
                           BLOB,
                           processed_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Create index for faster lookups
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS idx_article_id ON embeddings(article_id)
                       ''')

        conn.commit()
        conn.close()

    def _get_processed_ids(self):
        """Get set of already processed article IDs"""
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT article_id FROM embeddings")
        processed_ids = {str(row[0]) for row in cursor.fetchall()}

        conn.close()
        return processed_ids

    def _process_batch_worker(self, batch_data):
        """Worker function for multiprocessing - processes a single batch"""
        texts, article_ids = batch_data

        # Each worker creates its own model instance
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        try:
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
            return list(zip(article_ids, embeddings))
        except Exception as e:
            logger.error(f"Error in worker process: {e}")
            return []

    def _save_embeddings_batch(self, batch_data):
        """Save a batch of embeddings to database"""
        if not batch_data:  # Skip empty batches
            return

        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        # Prepare data for insertion
        insert_data = []
        for article_id, embedding in batch_data:
            # Serialize embedding as bytes
            embedding_blob = pickle.dumps(embedding.astype(np.float32))
            # Ensure article_id is a string
            article_id_str = str(article_id)
            insert_data.append((article_id_str, embedding_blob))

        # Insert batch
        cursor.executemany('''
            INSERT OR REPLACE INTO embeddings (article_id, embedding)
            VALUES (:article_id, :embedding)
        ''', [{'article_id': aid, 'embedding': emb} for aid, emb in insert_data])

        conn.commit()
        conn.close()

    def _encode_texts_parallel(self, texts, article_ids):
        """Parallelize only the model encoding part"""
        # Calculate chunk size for parallel processing
        chunk_size = max(1, len(texts) // self.n_workers)

        # Split texts into chunks (keep article_ids aligned)
        text_chunks = []
        id_chunks = []
        for i in range(0, len(texts), chunk_size):
            text_chunks.append(texts[i:i + chunk_size])
            id_chunks.append(article_ids[i:i + chunk_size])

        # Process chunks in parallel - only the encoding part
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._encode_chunk_worker, chunk_texts)
                       for chunk_texts in text_chunks]

            # Collect embeddings and pair with article_ids
            all_results = []
            for i, future in enumerate(futures):
                chunk_embeddings = future.result()
                chunk_ids = id_chunks[i]
                # Zip embeddings with their corresponding article_ids
                chunk_results = list(zip(chunk_ids, chunk_embeddings))
                all_results.extend(chunk_results)

            return all_results

    def _encode_chunk_worker(self, texts):
        """Worker function that only does the embedding encoding"""
        # Each worker creates its own model instance
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        try:
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error in encoding worker: {e}")
            return []

    def _encode_texts_sequential(self, texts, article_ids):
        """Sequential processing using single model instance"""
        try:
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
            return list(zip(article_ids, embeddings))
        except Exception as e:
            logger.error(f"Error processing texts: {e}")
            return []

    def get_total_records(self) -> int:
        """Get total number of records to process"""
        query = """
                SELECT COUNT(*) as total
                FROM articles a
                         JOIN abstracts ab ON a.article_id = ab.article_id \
                """

        with create_engine(self.source_db_url).connect() as conn:
            result = pd.read_sql_query(query, conn)
            return result['total'].iloc[0]

    def process_batch_from_db(self, batch_size: int) -> bool:
        """Process a single batch from database using cursor-based pagination"""
        if self.last_random_rank is None:
            query = """
                    SELECT a.article_id,
                           a.title,
                           ab.abstract,
                           ao.random_rank
                    FROM articles a
                             JOIN abstracts ab ON a.article_id = ab.article_id
                             JOIN articles_order ao ON a.article_id = ao.article_id
                    ORDER BY ao.random_rank LIMIT :batch_size \
                    """
            params = {'batch_size': batch_size}
        else:
            query = """
                    SELECT a.article_id,
                           a.title,
                           ab.abstract,
                           ao.random_rank
                    FROM articles a
                             JOIN abstracts ab ON a.article_id = ab.article_id
                             JOIN articles_order ao ON a.article_id = ao.article_id
                    WHERE ao.random_rank > :last_rank
                    ORDER BY ao.random_rank LIMIT :batch_size \
                    """
            params = {'last_rank': int(self.last_random_rank), 'batch_size': batch_size}

        try:
            logger.info(f"Querying articles with cursor at rank {self.last_random_rank}")

            with create_engine(self.source_db_url).connect() as conn:
                result = conn.execute(text(query), params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df.empty:
                logger.info("No more records found - batch processing complete")
                return False

            logger.info(f"Retrieved {len(df)} records from database")

            # Update cursor position
            if len(df) > 0:
                self.last_random_rank = df['random_rank'].iloc[-1]
                logger.info(f"Updated cursor to random_rank: {self.last_random_rank}")
            else:
                return False

            # Get already processed IDs for this batch
            processed_ids = self._get_processed_ids()

            # Filter out already processed articles
            unprocessed_df = df[~df['article_id'].isin(processed_ids)].copy()

            if len(unprocessed_df) == 0:
                logger.info("All articles in this batch already processed, continuing to next batch")
                return True  # Continue to next batch

            logger.info(f"Processing {len(unprocessed_df)} unprocessed articles from batch")

            # Prepare texts and article IDs
            texts = []
            article_ids = []

            for _, row in unprocessed_df.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else ""
                abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
                combined_text = f"{title} . {abstract}".strip()

                texts.append(combined_text)
                article_ids.append(int(row['article_id']))

            # Process batch - only parallelize the encoding part
            if self.use_multiprocessing:
                batch_result = self._encode_texts_parallel(texts, article_ids)
            else:
                batch_result = self._encode_texts_sequential(texts, article_ids)

            self._save_embeddings_batch(batch_result)

            logger.info(f"Processed and saved {len(batch_result)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error processing batch with cursor at rank {self.last_random_rank}: {e}")
            return False

    def process_all_articles_from_db(self):
        """Process all articles from database using cursor-based pagination"""
        if not self.source_db_url:
            raise ValueError("source_db_url must be provided to process from database")

        logger.info("Getting total record count...")
        total_records = self.get_total_records()
        logger.info(f"Total records to process: {total_records}")

        if self.use_multiprocessing:
            logger.info(f"Using {self.n_workers} worker processes")
        else:
            logger.info(f"Using sequential processing with {torch.get_num_threads()} threads")

        batch_count = 0
        total_processed = 0

        # Get initial count of processed embeddings
        initial_processed = len(self._get_processed_ids())
        logger.info(f"Starting with {initial_processed} already processed embeddings")

        with tqdm(total=total_records, desc="Processing articles") as pbar:
            while True:
                if not self.process_batch_from_db(self.batch_size):
                    break

                batch_count += 1

                # Update progress every 10 batches
                if batch_count % 10 == 0:
                    current_processed = len(self._get_processed_ids())
                    newly_processed = current_processed - initial_processed
                    pbar.update(min(self.batch_size * 10, newly_processed - total_processed))
                    total_processed = newly_processed

                    logger.info(f"Batch {batch_count}: {current_processed:,} total embeddings stored, "
                                f"{newly_processed:,} newly processed this session. "
                                f"Cursor at rank: {self.last_random_rank}")

        # Final update
        final_processed = len(self._get_processed_ids())
        logger.info(f"Processing complete! Total embeddings in database: {final_processed:,}")

    def get_embedding(self, article_id):
        """Retrieve embedding for a specific article"""
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT embedding FROM embeddings WHERE article_id = :article_id",
                       {'article_id': int(article_id)})
        result = cursor.fetchone()

        conn.close()

        if result:
            return pickle.loads(result[0])
        return None

    def get_embeddings_batch(self, article_ids):
        """Retrieve embeddings for multiple articles"""
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        # Convert article_ids to integers
        article_ids_int = [int(aid) for aid in article_ids]
        placeholders = ','.join([':id' + str(i) for i in range(len(article_ids_int))])
        params = {f'id{i}': aid for i, aid in enumerate(article_ids_int)}

        cursor.execute(f"SELECT article_id, embedding FROM embeddings WHERE article_id IN ({placeholders})", params)
        results = cursor.fetchall()

        conn.close()

        embeddings_dict = {}
        for article_id, embedding_blob in results:
            embeddings_dict[article_id] = pickle.loads(embedding_blob)

        return embeddings_dict

    def test_embedding_integrity(self):
        """Test that stored embeddings match freshly computed ones"""
        logger.info("Testing embedding integrity...")

        # Get a random article from the database
        conn = sqlite3.connect(self.embeddings_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT article_id FROM embeddings ORDER BY RANDOM() LIMIT 1")
        result = cursor.fetchone()

        if not result:
            logger.warning("No embeddings found in database for testing")
            return False

        test_article_id = result[0]
        conn.close()

        # Get the stored embedding
        stored_embedding = self.get_embedding(test_article_id)
        if stored_embedding is None:
            logger.error(f"Could not retrieve stored embedding for article {test_article_id}")
            return False

        # Get the original text for this article
        with create_engine(self.source_db_url).connect() as conn:
            result = conn.execute(text("""
                                       SELECT a.title, ab.abstract
                                       FROM articles a
                                                JOIN abstracts ab ON a.article_id = ab.article_id
                                       WHERE a.article_id = :article_id
                                       """), {'article_id': test_article_id})

            row = result.fetchone()
            if not row:
                logger.error(f"Could not find original text for article {test_article_id}")
                return False

            title = str(row[0]) if row[0] else ""
            abstract = str(row[1]) if row[1] else ""
            combined_text = f"{title} . {abstract}".strip()

        # Compute fresh embedding
        if not hasattr(self, 'model') or self.model is None:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        fresh_embedding = self.model.encode([combined_text], show_progress_bar=False)[0]

        # Compare embeddings using cosine similarity
        dot_product = np.dot(stored_embedding, fresh_embedding)
        norm_stored = np.linalg.norm(stored_embedding)
        norm_fresh = np.linalg.norm(fresh_embedding)
        cosine_similarity = dot_product / (norm_stored * norm_fresh)

        # Check if embeddings are nearly identical (should be > 0.999)
        threshold = 0.999
        is_match = cosine_similarity > threshold

        logger.info(f"Embedding integrity test for article {test_article_id}:")
        logger.info(f"  Cosine similarity: {cosine_similarity:.6f}")
        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Test result: {'PASS' if is_match else 'FAIL'}")

        if not is_match:
            logger.error(f"Embedding integrity test FAILED! Similarity {cosine_similarity:.6f} < {threshold}")
            logger.info(f"  Text: {combined_text[:100]}...")
            logger.info(f"  Stored embedding shape: {stored_embedding.shape}")
            logger.info(f"  Fresh embedding shape: {fresh_embedding.shape}")

        return is_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute article text embeddings.")
    parser.add_argument("--db-path", default="articles.db", help="Read-only source corpus.")
    parser.add_argument("--output-db", default="output/article_embeddings.db")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-multiprocessing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    processor = TextEmbeddingProcessor(
        embeddings_db_path=args.output_db,
        source_db_url=f"sqlite:///{args.db_path}",
        batch_size=args.batch_size,
        n_workers=args.workers,
        use_multiprocessing=not args.no_multiprocessing,
    )
    processor.process_all_articles_from_db()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())