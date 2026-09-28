#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUTPUT_DIR = "output/article_text_ordered_parquet"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Export ordered article text from SQLite to a chunked Parquet dataset."
    )
    parser.add_argument("--db-path", default="articles.db", help="Path to SQLite database.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output Parquet dataset directory.")
    parser.add_argument("--batch-size", type=int, default=100000, help="Rows per Parquet part.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--start-rank", type=int, default=0, help="Start after this random_rank.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum rows to export.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing parts and start over.")
    parser.add_argument("--sqlite-cache-mb", type=int, default=512, help="SQLite page-cache size in MiB.")
    return parser.parse_args(argv)


def atomic_write_json(path: Path, data: dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_manifest(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_output_dir(output_dir: Path, overwrite: bool, resume: bool):
    manifest_path = output_dir / "_manifest.json"

    if overwrite and output_dir.exists():
        for path in output_dir.glob("part_*.parquet"):
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and not resume:
        existing_parts = list(output_dir.glob("part_*.parquet"))
        if existing_parts or manifest_path.exists():
            raise FileExistsError(
                f"{output_dir} already contains export files. Use --resume or --overwrite."
            )

    return manifest_path


def configure_sqlite(conn: sqlite3.Connection, cache_mb: int):
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(f"PRAGMA cache_size = {-max(cache_mb, 1) * 1024}")
    conn.execute("""
                 CREATE TEMP TABLE IF NOT EXISTS temp_export_order
                 (
                     article_id  INTEGER PRIMARY KEY,
                     random_rank INTEGER NOT NULL
                 )
                 """)


def get_max_random_rank(conn: sqlite3.Connection):
    row = conn.execute("SELECT MAX(random_rank) FROM articles_order").fetchone()
    return int(row[0] or 0)


def fetch_batch(conn: sqlite3.Connection, last_random_rank: int, batch_size: int):
    conn.execute("DELETE FROM temp_export_order")
    conn.execute(
        """
        INSERT INTO temp_export_order (article_id, random_rank)
        SELECT ao.article_id, ao.random_rank
        FROM articles_order ao
        WHERE ao.random_rank > ?
        ORDER BY ao.random_rank
        LIMIT ?
        """,
        (int(last_random_rank), int(batch_size)),
    )

    row = conn.execute("SELECT COUNT(*), MAX(random_rank) FROM temp_export_order").fetchone()
    ordered_count = int(row[0] or 0)
    batch_last_rank = int(row[1] or last_random_rank)
    if ordered_count == 0:
        return pd.DataFrame(), batch_last_rank

    cursor = conn.execute(
        """
        SELECT batch.article_id,
               a.publication_year,
               a.title,
               ab.abstract,
               batch.random_rank
        FROM temp_export_order batch
                 JOIN articles a ON batch.article_id = a.article_id
                 LEFT JOIN abstracts ab ON batch.article_id = ab.article_id
        ORDER BY batch.random_rank
        """
    )
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    return df, batch_last_rank


def normalize_dataframe(df: pd.DataFrame):
    if df.empty:
        return df

    df["article_id"] = df["article_id"].astype("int64")
    df["random_rank"] = df["random_rank"].astype("int64")
    df["publication_year"] = df["publication_year"].astype("Int32")
    df["title"] = df["title"].fillna("").astype("string")
    df["abstract"] = df["abstract"].fillna("").astype("string")
    return df


def write_parquet_part(df: pd.DataFrame, path: Path, compression: str):
    schema = pa.schema(
        [
            ("article_id", pa.int64()),
            ("publication_year", pa.int32()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("random_rank", pa.int64()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    )


def export_dataset(args):
    output_dir = Path(args.output_dir)
    manifest_path = prepare_output_dir(output_dir, args.overwrite, args.resume)
    manifest = load_manifest(manifest_path) if args.resume else None

    last_random_rank = args.start_rank
    next_part = 0
    total_rows = 0

    if manifest:
        last_random_rank = int(manifest["last_random_rank"])
        next_part = int(manifest["next_part"])
        total_rows = int(manifest["total_rows"])
        print(f"Resuming from random_rank {last_random_rank:,}, part {next_part:,}")

    started_at = time.time()
    with sqlite3.connect(args.db_path) as conn:
        configure_sqlite(conn, args.sqlite_cache_mb)
        max_random_rank = get_max_random_rank(conn)
        print(f"Max random_rank: {max_random_rank:,}")

        while last_random_rank < max_random_rank:
            if args.limit is not None:
                remaining = args.limit - total_rows
                if remaining <= 0:
                    break
                batch_size = min(args.batch_size, remaining)
            else:
                batch_size = args.batch_size

            batch_started_at = time.time()
            df, batch_last_rank = fetch_batch(conn, last_random_rank, batch_size)
            if df.empty:
                break

            df = normalize_dataframe(df)
            part_path = output_dir / f"part_{next_part:06d}.parquet"
            write_parquet_part(df, part_path, args.compression)

            rows_written = len(df)
            total_rows += rows_written
            last_random_rank = batch_last_rank
            next_part += 1

            manifest_data = {
                "db_path": args.db_path,
                "output_dir": str(output_dir),
                "compression": args.compression,
                "batch_size": args.batch_size,
                "last_random_rank": last_random_rank,
                "next_part": next_part,
                "total_rows": total_rows,
                "max_random_rank": max_random_rank,
                "updated_at": time.time(),
            }
            atomic_write_json(manifest_path, manifest_data)

            elapsed = time.time() - batch_started_at
            total_elapsed = max(time.time() - started_at, 1e-9)
            print(
                f"Wrote {part_path.name}: {rows_written:,} rows, "
                f"rank {last_random_rank:,}/{max_random_rank:,}, "
                f"{elapsed:.1f}s batch, {total_rows / total_elapsed:.1f} rows/s overall"
            )

            del df

    print(f"Exported {total_rows:,} rows to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
