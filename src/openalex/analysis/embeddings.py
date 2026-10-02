"""Build and read article text-embedding artifacts.

The source corpus is always opened read-only. Embeddings are written to a
separate SQLite database so long runs can resume without modifying the corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import quote

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT_DIR = Path("output/embeddings")
DEFAULT_WORKERS = 32
DEFAULT_CPU_ENCODE_BATCH_SIZE = 32
DEFAULT_GPU_ENCODE_BATCH_SIZE = 256
DATABASE_NAME = "embeddings.db"
TEXT_FORMAT = "{title} . {abstract}"
_SQLITE_IN_LIMIT = 900

EncoderFactory = Callable[[str, str | None], object]

_worker_encoder = None
_worker_encode_batch_size = 32


def _sentence_transformer(model_name: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding support is not installed. Run "
            "`python -m pip install -e '.[embeddings]'`."
        ) from exc
    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def _init_worker(model_name: str, device: str | None, encode_batch_size: int) -> None:
    global _worker_encoder, _worker_encode_batch_size
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _worker_encoder = _sentence_transformer(model_name, device)
    _worker_encode_batch_size = encode_batch_size


def _encode_worker(texts: list[str]) -> np.ndarray:
    if _worker_encoder is None:
        raise RuntimeError("Embedding worker was not initialized")
    return np.asarray(
        _worker_encoder.encode(
            texts,
            batch_size=_worker_encode_batch_size,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def _detect_accelerator_devices() -> list[str]:
    """Return visible accelerator devices in preferred execution order."""
    try:
        import torch
    except ImportError:
        return []
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return ["mps"]
    return []


def _resolve_runtime(
    device: str | None,
    workers: int | None,
    encode_batch_size: int | None,
) -> tuple[str, list[str], int, int]:
    """Resolve automatic CPU, single-accelerator, or multi-GPU execution."""
    requested = (device or "auto").lower()
    detected = (
        []
        if requested == "cpu"
        else _detect_accelerator_devices()
    )
    if requested == "auto":
        devices = detected or ["cpu"]
    elif requested == "cpu":
        devices = ["cpu"]
    elif requested == "cuda":
        devices = [value for value in detected if value.startswith("cuda:")]
        if not devices:
            raise RuntimeError("CUDA was requested but no CUDA GPU is available")
    elif requested.startswith("cuda:"):
        if requested not in detected:
            raise RuntimeError(f"{requested} was requested but is not available")
        devices = [requested]
    elif requested == "mps":
        if "mps" not in detected:
            raise RuntimeError("MPS was requested but is not available")
        devices = ["mps"]
    else:
        devices = [device or requested]

    accelerated = devices[0] != "cpu"
    if workers is None:
        resolved_workers = len(devices) if accelerated else DEFAULT_WORKERS
    else:
        if workers < 1:
            raise ValueError("--workers must be >= 1")
        resolved_workers = workers

    if accelerated:
        if resolved_workers > len(devices):
            raise ValueError(
                f"--workers={resolved_workers} exceeds the {len(devices)} "
                f"available accelerator device(s)"
            )
        devices = devices[:resolved_workers]
    else:
        devices = ["cpu"] * resolved_workers

    if encode_batch_size is None:
        resolved_batch_size = (
            DEFAULT_GPU_ENCODE_BATCH_SIZE
            if any(value.startswith("cuda:") for value in devices)
            else DEFAULT_CPU_ENCODE_BATCH_SIZE
        )
    else:
        if encode_batch_size < 1:
            raise ValueError("--encode-batch-size must be >= 1")
        resolved_batch_size = encode_batch_size

    runtime_device = (
        "cuda"
        if len(devices) > 1 and all(value.startswith("cuda:") for value in devices)
        else devices[0]
    )
    return runtime_device, devices, resolved_workers, resolved_batch_size


def article_text(title: object, abstract: object) -> str:
    """Return the exact text supplied to the embedding model."""
    title_text = "" if title is None else str(title)
    abstract_text = "" if abstract is None else str(abstract)
    return f"{title_text} . {abstract_text}".strip()


def encode_embedding(value: np.ndarray) -> bytes:
    """Serialize a float32 vector in the legacy-compatible representation."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Embedding must be one-dimensional, got {vector.shape}")
    return pickle.dumps(vector, protocol=pickle.HIGHEST_PROTOCOL)


def decode_embedding(value: bytes, *, dimension: int | None = None) -> np.ndarray:
    """Deserialize and validate an embedding BLOB."""
    try:
        vector = np.asarray(pickle.loads(value), dtype=np.float32)
    except Exception as exc:
        raise ValueError("Invalid embedding BLOB") from exc
    if vector.ndim != 1:
        raise ValueError(f"Embedding must be one-dimensional, got {vector.shape}")
    if dimension is not None and vector.shape != (dimension,):
        raise ValueError(
            f"Embedding has dimension {vector.size}, expected {dimension}"
        )
    return vector


class EmbeddingStore:
    """Read embeddings from a manifest artifact or a legacy database file."""

    def __init__(self, path: str | Path):
        source = Path(path).expanduser().resolve()
        self.root = source if source.is_dir() else source.parent
        self.manifest_path = self.root / "manifest.json"
        self.manifest = _read_json(self.manifest_path) if self.manifest_path.is_file() else {}
        if source.is_dir():
            database_name = self.manifest.get("database", DATABASE_NAME)
            self.database_path = source / str(database_name)
        else:
            self.database_path = source
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Embedding database not found: {self.database_path}")
        version = self.manifest.get("artifact_version")
        if version is not None and int(version) != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported embedding artifact version {version}; expected {ARTIFACT_VERSION}"
            )
        dimension = self.manifest.get("dimension")
        self.dimension = int(dimension) if dimension is not None else None

    def _connect(self) -> sqlite3.Connection:
        return _connect_readonly(self.database_path)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def article_ids(self) -> list[int]:
        return list(self.iter_article_ids())

    def iter_article_ids(self) -> Iterator[int]:
        """Stream stored article IDs in deterministic order."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT article_id FROM embeddings ORDER BY article_id"
            )
            for row in rows:
                yield int(row[0])

    def get_embedding(self, article_id: int | str) -> np.ndarray | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT embedding FROM embeddings WHERE article_id = ?",
                (str(article_id),),
            ).fetchone()
        if row is None:
            return None
        return decode_embedding(row[0], dimension=self.dimension)

    def get_embeddings_batch(
        self, article_ids: Iterable[int | str]
    ) -> dict[int, np.ndarray]:
        ids = [int(article_id) for article_id in article_ids]
        found: dict[int, np.ndarray] = {}
        with self._connect() as connection:
            for chunk in _chunks(ids, _SQLITE_IN_LIMIT):
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT article_id, embedding FROM embeddings "
                    f"WHERE article_id IN ({placeholders})",
                    [str(article_id) for article_id in chunk],
                )
                for article_id, blob in rows:
                    found[int(article_id)] = decode_embedding(
                        blob, dimension=self.dimension
                    )
        return found

    def iter_batches(
        self,
        batch_size: int = 10_000,
        *,
        start_after: int | None = None,
    ) -> Iterator[tuple[list[int], np.ndarray]]:
        """Stream every stored embedding without loading the artifact at once."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        with self._connect() as connection:
            if start_after is None:
                cursor = connection.execute(
                    """
                    SELECT article_id, embedding
                    FROM embeddings
                    ORDER BY article_id
                    """
                )
            else:
                cursor = connection.execute(
                    """
                    SELECT article_id, embedding
                    FROM embeddings
                    WHERE article_id > ?
                    ORDER BY article_id
                    """,
                    (int(start_after),),
                )
            while rows := cursor.fetchmany(batch_size):
                article_ids = [int(row[0]) for row in rows]
                vectors = np.vstack(
                    [
                        decode_embedding(row[1], dimension=self.dimension)
                        for row in rows
                    ]
                )
                yield article_ids, vectors


def build_embeddings(
    db_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 1000,
    encode_batch_size: int | None = None,
    workers: int | None = None,
    device: str | None = None,
    resume: bool = False,
    encoder=None,
    encoder_factory: EncoderFactory = _sentence_transformer,
) -> dict[str, object]:
    """Encode every article with an abstract into a resumable artifact."""
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if encoder is not None:
        if workers not in {None, 1}:
            raise ValueError("An injected encoder can only be used with workers=1")
        if encode_batch_size is not None and encode_batch_size < 1:
            raise ValueError("--encode-batch-size must be >= 1")
        runtime_device = device or "injected"
        execution_devices = [runtime_device]
        workers = 1
        encode_batch_size = encode_batch_size or DEFAULT_CPU_ENCODE_BATCH_SIZE
    else:
        (
            runtime_device,
            execution_devices,
            workers,
            encode_batch_size,
        ) = _resolve_runtime(device, workers, encode_batch_size)
    logger.info(
        "Embedding runtime: device=%s workers=%s encode_batch_size=%s",
        runtime_device,
        workers,
        encode_batch_size,
    )

    source = Path(db_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("--output-dir must not contain the source database")
    database = output / DATABASE_NAME
    manifest_path = output / "manifest.json"
    config = {
        "artifact_version": ARTIFACT_VERSION,
        "database": DATABASE_NAME,
        "model": model_name,
        "text_format": TEXT_FORMAT,
        "source_database": str(source),
        "source_size": source.stat().st_size,
        "encode_batch_size": encode_batch_size,
        "device": runtime_device,
    }

    existing = _read_json(manifest_path) if manifest_path.is_file() else None
    if not resume and (database.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"{output} already contains embeddings. Use --resume or a new --output-dir."
        )
    if resume and existing is not None:
        _validate_resume(existing, config)
    if resume and database.exists() and existing is None:
        logger.warning("Resuming a legacy embedding database without a manifest")

    output.mkdir(parents=True, exist_ok=True)
    _initialize_database(database)
    stored_count, last_article_id = _embedding_progress(database)
    dimension = (
        int(existing["dimension"])
        if existing is not None and existing.get("dimension") is not None
        else _stored_dimension(database)
    )
    total = _source_count(source)
    manifest = {
        **config,
        "batch_size": batch_size,
        "complete": False,
        "dimension": dimension,
        "last_article_id": last_article_id,
        "rows": stored_count,
        "source_rows": total,
        "workers": workers,
    }
    _atomic_json(manifest_path, manifest)

    local_encoder = encoder
    executors: list[ProcessPoolExecutor] = []

    processed = stored_count
    try:
        for rows in _source_batches(source, batch_size):
            existing_ids = _existing_ids(database, [int(row[0]) for row in rows])
            rows = [row for row in rows if int(row[0]) not in existing_ids]
            if not rows:
                continue
            article_ids = [int(row[0]) for row in rows]
            texts = [article_text(row[1], row[2]) for row in rows]
            if workers == 1:
                if local_encoder is None:
                    local_encoder = encoder_factory(model_name, device)
                vectors = np.asarray(
                    local_encoder.encode(
                        texts,
                        batch_size=encode_batch_size,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                dimension = _validate_encoded_batch(
                    vectors,
                    len(article_ids),
                    dimension,
                )
                _store_batch(database, article_ids, vectors)
                processed += len(article_ids)
                last_article_id = max(last_article_id or -1, article_ids[-1])
                _update_progress_manifest(
                    manifest_path,
                    manifest,
                    dimension=dimension,
                    last_article_id=last_article_id,
                    rows=processed,
                )
                logger.info("Stored %s/%s article embeddings", processed, total)
            else:
                if not executors:
                    if all(value == "cpu" for value in execution_devices):
                        executors = [
                            ProcessPoolExecutor(
                                max_workers=workers,
                                initializer=_init_worker,
                                initargs=(model_name, "cpu", encode_batch_size),
                            )
                        ]
                    else:
                        executors = [
                            ProcessPoolExecutor(
                                max_workers=1,
                                initializer=_init_worker,
                                initargs=(model_name, worker_device, encode_batch_size),
                                mp_context=get_context("spawn"),
                            )
                            for worker_device in execution_devices
                        ]
                chunk_indices = [
                    indices.tolist()
                    for indices in np.array_split(
                        np.arange(len(texts)), min(workers, len(texts))
                    )
                    if len(indices)
                ]
                id_chunks = [
                    [article_ids[index] for index in indices]
                    for indices in chunk_indices
                ]
                text_chunks = [
                    [texts[index] for index in indices]
                    for indices in chunk_indices
                ]
                if len(executors) == 1:
                    encoded_chunks = executors[0].map(_encode_worker, text_chunks)
                else:
                    futures = [
                        executor.submit(_encode_worker, chunk)
                        for executor, chunk in zip(
                            executors, text_chunks, strict=True
                        )
                    ]
                    encoded_chunks = (
                        future.result() for future in futures
                    )
                for chunk_ids, chunk_vectors in zip(
                    id_chunks, encoded_chunks, strict=True
                ):
                    dimension = _validate_encoded_batch(
                        chunk_vectors,
                        len(chunk_ids),
                        dimension,
                    )
                    _store_batch(database, chunk_ids, chunk_vectors)
                    processed += len(chunk_ids)
                    last_article_id = max(last_article_id or -1, chunk_ids[-1])
                    _update_progress_manifest(
                        manifest_path,
                        manifest,
                        dimension=dimension,
                        last_article_id=last_article_id,
                        rows=processed,
                    )
                    logger.info("Stored %s/%s article embeddings", processed, total)
    finally:
        for executor in executors:
            executor.shutdown()

    final_count, final_article_id = _embedding_progress(database)
    manifest.update(
        {
            "complete": final_count == total,
            "dimension": dimension,
            "last_article_id": final_article_id,
            "rows": final_count,
        }
    )
    _atomic_json(manifest_path, manifest)
    if not manifest["complete"]:
        raise RuntimeError(f"Stored {final_count} embeddings for {total} source articles")
    return manifest


def _source_batches(database: Path, batch_size: int) -> Iterator[list[sqlite3.Row]]:
    with _connect_readonly(database) as connection:
        cursor = connection.execute(
            """
            SELECT a.article_id, a.title, ab.abstract
            FROM articles a
            JOIN abstracts ab ON ab.article_id = a.article_id
            ORDER BY a.article_id
            """
        )
        while rows := cursor.fetchmany(batch_size):
            yield rows


def _source_count(database: Path) -> int:
    with _connect_readonly(database) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM articles a
                JOIN abstracts ab ON ab.article_id = a.article_id
                """
            ).fetchone()[0]
        )


def _existing_ids(database: Path, article_ids: Sequence[int]) -> set[int]:
    existing: set[int] = set()
    with sqlite3.connect(database) as connection:
        for chunk in _chunks(article_ids, _SQLITE_IN_LIMIT):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT article_id FROM embeddings WHERE article_id IN ({placeholders})",
                [str(article_id) for article_id in chunk],
            )
            existing.update(int(row[0]) for row in rows)
    return existing


def _validate_encoded_batch(
    vectors: np.ndarray,
    expected_rows: int,
    dimension: int | None,
) -> int:
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            f"Encoder returned shape {vectors.shape} for {expected_rows} texts"
        )
    batch_dimension = int(vectors.shape[1])
    if dimension is not None and dimension != batch_dimension:
        raise ValueError(
            f"Encoder returned dimension {batch_dimension}, expected {dimension}"
        )
    return batch_dimension


def _update_progress_manifest(
    path: Path,
    manifest: dict,
    *,
    dimension: int,
    last_article_id: int,
    rows: int,
) -> None:
    manifest.update(
        {
            "dimension": dimension,
            "last_article_id": last_article_id,
            "rows": rows,
        }
    )
    _atomic_json(path, manifest)


def _initialize_database(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings(
                article_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _embedding_progress(database: Path) -> tuple[int, int | None]:
    with sqlite3.connect(database) as connection:
        count, last_id = connection.execute(
            "SELECT COUNT(*), MAX(CAST(article_id AS INTEGER)) FROM embeddings"
        ).fetchone()
    return int(count), int(last_id) if last_id is not None else None


def _stored_dimension(database: Path) -> int | None:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT embedding FROM embeddings LIMIT 1").fetchone()
    return int(decode_embedding(row[0]).size) if row else None


def _store_batch(
    database: Path, article_ids: Sequence[int], vectors: np.ndarray
) -> None:
    payload = [
        (int(article_id), sqlite3.Binary(encode_embedding(vector)))
        for article_id, vector in zip(article_ids, vectors, strict=True)
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO embeddings(article_id, embedding)
            VALUES (?, ?)
            """,
            payload,
        )


def _validate_resume(manifest: dict, config: dict) -> None:
    for key in (
        "artifact_version",
        "database",
        "model",
        "text_format",
        "source_database",
        "source_size",
    ):
        if manifest.get(key) != config[key]:
            raise ValueError(
                f"Existing embedding artifact used {key}={manifest.get(key)!r}, "
                f"not {config[key]!r}. Use a new --output-dir."
            )


def _connect_readonly(database: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(database.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _chunks(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a manifest-backed article text-embedding artifact."
    )
    parser.add_argument("--db-path", type=Path, default=Path("articles.db"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="Encoder batch size (auto: 32 on CPU/MPS, 256 on CUDA).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count (auto: 32 on CPU, one per visible GPU).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override such as cpu, mps, cuda, or cuda:0 (default: auto).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    manifest = build_embeddings(
        args.db_path,
        args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        workers=args.workers,
        device=args.device,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
