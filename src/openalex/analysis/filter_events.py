"""Classify extracted event keywords as genuine terms or spurious artefacts.

Each keyword is shown to GPT-6 Luna with the three other keywords whose
co-occurrence profiles are most similar. Similarity is the cosine of
L2-normalized co-occurrence rows, the same similarity the keyword dendrogram
uses before complete linkage. A genuine keyword is one whose frequency could
have moved over about the last 15 years for scientific reasons. An artefact is
a term whose change is better explained by language or writing, such as a
non-English word or a shift in how papers are phrased.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openalex.website.build import l2_normalize_cooccurrence, load_event_artifacts

logger = logging.getLogger(__name__)

MODEL = "gpt-6-luna"
DEFAULT_NEIGHBORS = 3
DEFAULT_WORKERS = 8
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_RETRIES = 4
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
LABELS = ("genuine", "artefact")
_COSINE_BATCH = 256

SYSTEM_PROMPT = """You classify keywords extracted from social-science papers because their frequency changed sharply over about the last 15 years.

Label a keyword genuine when it is a real scientific or substantive term whose frequency could have moved for scientific reasons: a concept, method, topic, population, place, institution, or technology entering or leaving the literature.

Label a keyword artefact when the change is better explained by a corpus or writing artefact than by science. Artefacts include non-English words, boilerplate, spelling or hyphenation changes, section headings, statistical notation, generic function words, OCR or metadata debris, and similar shifts in how papers are written rather than what they study.

You will see the keyword and the other keywords with the most similar co-occurrence profiles. Those neighbors are context. Judge the keyword itself.

Return the label and one sentence explaining it."""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Neighbor:
    """One co-occurrence neighbor and its row-cosine similarity."""

    ngram: str
    similarity: float


@dataclass(frozen=True)
class KeywordContext:
    """An extracted keyword and the neighbors shown to the model."""

    ngram: str
    neighbors: tuple[Neighbor, ...]
    event: Mapping[str, str]


@dataclass(frozen=True)
class Classification:
    """Model verdict for one extracted keyword."""

    ngram: str
    label: str
    reason: str


Classify = Callable[[KeywordContext], Classification]


def top_correlated_indices(matrix, k: int) -> list[list[tuple[int, float]]]:
    """Return the ``k`` other rows with the highest dendrogram cosine.

    Rows with zero L2 norm are omitted, matching the dendrogram's drop of
    those keywords. Ties keep the earlier row.
    """
    if k < 1:
        raise ValueError("--neighbors must be >= 1")
    filtered = matrix.astype(np.float64).tocsr()
    if filtered.shape[0] != filtered.shape[1]:
        raise ValueError("Co-occurrence matrix must be square")
    count = filtered.shape[0]
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    if count < 2:
        return neighbors
    norms = np.sqrt(np.asarray(filtered.multiply(filtered).sum(axis=1)).ravel())
    positive = np.flatnonzero(norms > 0)
    if positive.size < 2:
        return neighbors
    normalized = l2_normalize_cooccurrence(filtered[positive][:, positive])
    width = normalized.shape[0]
    take = min(k, width - 1)
    for start in range(0, width, _COSINE_BATCH):
        end = min(start + _COSINE_BATCH, width)
        scores = (normalized[start:end] @ normalized.T).toarray()
        for local in range(end - start):
            row = scores[local].copy()
            row[start + local] = -np.inf
            order = np.argsort(-row, kind="stable")
            chosen = [int(index) for index in order if np.isfinite(row[index])][:take]
            neighbors[int(positive[start + local])] = [
                (int(positive[index]), float(row[index])) for index in chosen
            ]
    return neighbors


def render_prompt(context: KeywordContext) -> str:
    """Show the keyword and its most correlated keywords."""
    lines = [f"Keyword: {context.ngram}", "Most correlated keywords:"]
    if context.neighbors:
        for rank, neighbor in enumerate(context.neighbors, start=1):
            lines.append(f"{rank}. {neighbor.ngram} ({neighbor.similarity:.3f})")
    else:
        lines.append("(none)")
    return "\n".join(lines)


def classification_from_response(ngram: str, response) -> Classification:
    """Read a genuine/artefact verdict from a Responses API result."""
    for text in _iter_response_texts(response):
        payload = _parse_classification(text)
        if payload is not None:
            return Classification(ngram=ngram, label=payload["label"], reason=payload["reason"])
    raise RuntimeError(f"Model response for {ngram!r} did not match the classification schema")


def classify_keyword(
    context: KeywordContext,
    *,
    client,
    model: str = MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Classification:
    """Ask the model to classify one keyword, retrying transient API errors."""
    if max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    prompt = render_prompt(context)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "classification",
                        "strict": True,
                        "schema": CLASSIFICATION_SCHEMA,
                    }
                },
            )
            return classification_from_response(context.ngram, response)
        except Exception as error:
            last_error = error
            if attempt + 1 >= max_retries or not _retryable(error):
                raise
            delay = min(2**attempt, 30)
            logger.warning(
                "Retrying %s after %s (%s/%s)",
                context.ngram,
                error,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
    raise RuntimeError(f"Classification failed for {context.ngram!r}") from last_error


def filter_events(
    events_dir: str | Path,
    output_dir: str | Path,
    *,
    neighbors: int = DEFAULT_NEIGHBORS,
    model: str = MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    workers: int = DEFAULT_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    resume: bool = False,
    client=None,
    classify: Classify | None = None,
) -> dict[str, object]:
    """Classify each extracted event and write the verdicts."""
    if neighbors < 1:
        raise ValueError("--neighbors must be >= 1")
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"--reasoning-effort must be one of {', '.join(REASONING_EFFORTS)}")
    events_root = Path(events_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output == events_root:
        raise ValueError("--output-dir must be different from --events-dir")

    artifacts = load_event_artifacts(events_root)
    rows = _event_rows(events_root)
    contexts = keyword_contexts(artifacts, rows, neighbors)
    checkpoint = output / "classifications.jsonl"
    table = output / "classifications.csv"
    if not resume and (checkpoint.exists() or table.exists()):
        raise FileExistsError(
            f"{output} already contains classifications. Use --resume or a new --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    completed = _load_checkpoint(checkpoint) if resume and checkpoint.exists() else {}
    pending = [context for context in contexts if context.ngram not in completed]
    logger.info(
        "Classifying %s keywords with %s (%s already done, %s neighbors)",
        len(contexts),
        model,
        len(completed),
        neighbors,
    )
    if pending and classify is None:
        model_client = _openai_client(client)

        def classify(context: KeywordContext) -> Classification:
            return classify_keyword(
                context,
                client=model_client,
                model=model,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
            )

    verdicts = _classify_pending(pending, classify, checkpoint, completed, workers=workers)
    ordered = [verdicts[context.ngram] for context in contexts]
    _write_table(table, contexts, ordered, neighbors, _event_fields(rows))
    summary = _summary(output, ordered, model=model, neighbors=neighbors, reasoning_effort=reasoning_effort)
    _atomic_json(output / "manifest.json", summary)
    logger.info(
        "Classified %s keywords: %s genuine, %s artefact",
        summary["keywords"],
        summary["genuine"],
        summary["artefact"],
    )
    return summary


def keyword_contexts(
    artifacts: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    neighbors: int,
) -> list[KeywordContext]:
    """Pair each extracted keyword with its top co-occurrence neighbors."""
    vocabulary = [str(keyword) for keyword in artifacts["vocabulary"]]
    index_by_keyword = {keyword: index for index, keyword in enumerate(vocabulary)}
    ranked = top_correlated_indices(artifacts["matrix"], neighbors)
    seen: set[str] = set()
    contexts: list[KeywordContext] = []
    missing = 0
    for row in rows:
        ngram = row["ngram"]
        if ngram in seen:
            continue
        seen.add(ngram)
        index = index_by_keyword.get(ngram)
        if index is None:
            missing += 1
            chosen: list[tuple[int, float]] = []
        else:
            chosen = ranked[index]
        contexts.append(
            KeywordContext(
                ngram=ngram,
                neighbors=tuple(
                    Neighbor(ngram=vocabulary[neighbor], similarity=similarity)
                    for neighbor, similarity in chosen
                ),
                event={key: value for key, value in row.items() if key != "ngram"},
            )
        )
    if missing:
        logger.warning("%s extracted keywords are absent from the co-occurrence vocabulary", missing)
    if not contexts:
        raise ValueError("events.csv does not list any keywords")
    return contexts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify extracted event keywords as genuine scientific terms or "
            "spurious artefacts with GPT-6 Luna."
        )
    )
    parser.add_argument("--events-dir", type=Path, default=Path("output/events"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/filtered_events"))
    parser.add_argument("--model", default=MODEL, help="OpenAI model id. Default is GPT-6 Luna.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
        help="Most correlated keywords to show the model for each event.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip keywords already stored in classifications.jsonl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    summary = filter_events(
        args.events_dir,
        args.output_dir,
        neighbors=args.neighbors,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        workers=args.workers,
        max_retries=args.max_retries,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _classify_pending(
    pending: Sequence[KeywordContext],
    classify: Classify | None,
    checkpoint: Path,
    completed: dict[str, Classification],
    *,
    workers: int,
) -> dict[str, Classification]:
    verdicts = dict(completed)
    if not pending:
        return verdicts
    if classify is None:
        raise RuntimeError("No classifier is available for pending keywords")
    lock = threading.Lock()

    def store(result: Classification) -> None:
        with lock:
            verdicts[result.ngram] = result
            _append_checkpoint(checkpoint, result)

    if workers == 1:
        for index, context in enumerate(pending, start=1):
            store(classify(context))
            if index % 25 == 0 or index == len(pending):
                logger.info("Classified %s/%s pending keywords", index, len(pending))
        return verdicts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(classify, context): context for context in pending}
        finished = 0
        try:
            for future in as_completed(futures):
                store(future.result())
                finished += 1
                if finished % 25 == 0 or finished == len(pending):
                    logger.info("Classified %s/%s pending keywords", finished, len(pending))
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return verdicts


def _iter_response_texts(response) -> Iterator[str]:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                yield text
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        yield output_text


def _parse_classification(text: str) -> dict[str, str] | None:
    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        label = payload.get("label")
        reason = payload.get("reason")
        if label not in LABELS or not isinstance(reason, str) or not reason.strip():
            continue
        return {"label": label, "reason": reason.strip()}
    return None


def _retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    name = type(error).__name__
    return name in {"APIConnectionError", "APITimeoutError", "RateLimitError"}


def _openai_client(client):
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required for filter-events. "
            "Install the analysis extra: python -m pip install -e '.[analysis]'"
        ) from error
    return OpenAI()


def _event_rows(events_dir: Path) -> list[dict[str, str]]:
    path = events_dir / "events.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} was not found. Re-run event extraction first.")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ngram" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an ngram column")
        return list(reader)


def _event_fields(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    return [field for field in rows[0] if field != "ngram"]


def _load_checkpoint(path: Path) -> dict[str, Classification]:
    completed: dict[str, Classification] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error
            ngram = payload.get("ngram")
            label = payload.get("label")
            reason = payload.get("reason")
            if not isinstance(ngram, str) or label not in LABELS or not isinstance(reason, str):
                raise ValueError(f"{path} line {line_number} is not a classification")
            completed[ngram] = Classification(ngram=ngram, label=label, reason=reason)
    return completed


def _append_checkpoint(path: Path, result: Classification) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"ngram": result.ngram, "label": result.label, "reason": result.reason},
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()


def _write_table(
    path: Path,
    contexts: Sequence[KeywordContext],
    verdicts: Sequence[Classification],
    neighbors: int,
    event_fields: Sequence[str],
) -> None:
    neighbor_fields = [
        name
        for rank in range(1, neighbors + 1)
        for name in (f"correlated_{rank}", f"similarity_{rank}")
    ]
    fieldnames = ["ngram", "label", "reason", *neighbor_fields, *event_fields]
    by_ngram = {context.ngram: context for context in contexts}
    rows = []
    for verdict in verdicts:
        context = by_ngram[verdict.ngram]
        row: dict[str, object] = {
            "ngram": verdict.ngram,
            "label": verdict.label,
            "reason": verdict.reason,
        }
        for rank in range(neighbors):
            if rank < len(context.neighbors):
                neighbor = context.neighbors[rank]
                row[f"correlated_{rank + 1}"] = neighbor.ngram
                row[f"similarity_{rank + 1}"] = f"{neighbor.similarity:.6f}"
            else:
                row[f"correlated_{rank + 1}"] = ""
                row[f"similarity_{rank + 1}"] = ""
        for field in event_fields:
            row[field] = context.event.get(field, "")
        rows.append(row)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _summary(
    output: Path,
    verdicts: Sequence[Classification],
    *,
    model: str,
    neighbors: int,
    reasoning_effort: str,
) -> dict[str, object]:
    genuine = sum(verdict.label == "genuine" for verdict in verdicts)
    return {
        "artefact": len(verdicts) - genuine,
        "checkpoint": "classifications.jsonl",
        "classifications": "classifications.csv",
        "genuine": genuine,
        "keywords": len(verdicts),
        "model": model,
        "neighbors": neighbors,
        "output_dir": str(output),
        "reasoning_effort": reasoning_effort,
        "similarity": "l2-normalized-cooccurrence-cosine",
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
