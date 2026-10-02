"""Assign BERTopic topics to articles with precomputed text embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sqlite3
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from openalex.analysis.author_aggregation import file_sha256
from openalex.analysis.embeddings import DEFAULT_MODEL, EmbeddingStore

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 2
DEFAULT_OUTPUT_DIR = Path("output/topics")
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_CLASSIFICATION_BATCH_SIZE = 10_000
DEFAULT_CLASSIFIER_CV_FOLDS = 3
DEFAULT_CLASSIFIER_JOBS = 4
_SQLITE_IN_LIMIT = 900

TopicModelFactory = Callable[..., object]
ClassifierTrainer = Callable[..., tuple[object, dict[str, Any], list[dict[str, Any]]]]


def _topic_model_factory(
    *,
    random_seed: int,
    min_cluster_size: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    umap_components: int,
    use_keybert_representation: bool,
    embedding_model_name: str,
):
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "Topic-modeling support is not installed. Run "
            "`python -m pip install -e '.[topics]'`."
        ) from exc

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=umap_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_seed,
        low_memory=True,
        verbose=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=10_000,
    )
    representation_model = KeyBERTInspired() if use_keybert_representation else None
    return BERTopic(
        embedding_model=embedding_model_name,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=None,
        calculate_probabilities=True,
        verbose=True,
    )


def assign_topics(
    db_path: str | Path,
    embeddings_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    random_seed: int = 42,
    min_cluster_size: int = 25,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    umap_components: int = 5,
    outlier_threshold: float = 0.1,
    hierarchy: bool = True,
    visualizations: bool = True,
    save_model: bool = True,
    classification_batch_size: int = DEFAULT_CLASSIFICATION_BATCH_SIZE,
    classifier_cv_folds: int = DEFAULT_CLASSIFIER_CV_FOLDS,
    classifier_jobs: int = DEFAULT_CLASSIFIER_JOBS,
    classifier_test_size: float = 0.2,
    model_factory: TopicModelFactory = _topic_model_factory,
    classifier_trainer: ClassifierTrainer | None = None,
) -> dict[str, object]:
    """Discover BERTopic labels, train an MLP, and classify the full artifact."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("--sample-size must be >= 1 or 'all'")
    if min_cluster_size < 2:
        raise ValueError("--min-cluster-size must be >= 2")
    if n_neighbors < 2:
        raise ValueError("--n-neighbors must be >= 2")
    if umap_components < 2:
        raise ValueError("--umap-components must be >= 2")
    if not 0 <= outlier_threshold <= 1:
        raise ValueError("--outlier-threshold must be between 0 and 1")
    if classification_batch_size < 1:
        raise ValueError("--classification-batch-size must be >= 1")
    if classifier_cv_folds < 2:
        raise ValueError("--classifier-cv-folds must be >= 2")
    if classifier_jobs == 0:
        raise ValueError("--classifier-jobs must not be zero")
    if not 0 < classifier_test_size < 1:
        raise ValueError("--classifier-test-size must be between 0 and 1")

    source = Path(db_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    store = EmbeddingStore(embeddings_path)
    if store.manifest and not bool(store.manifest.get("complete")):
        raise ValueError(
            f"Embedding artifact is incomplete: {store.manifest_path}. "
            "Resume the embedding run before assigning topics."
        )
    if output == source or source.is_relative_to(output):
        raise ValueError("--output-dir must not contain the source database")
    if _output_exists(output):
        raise FileExistsError(
            f"{output} already contains topic results. Use a new --output-dir."
        )
    logger.info("Fingerprinting source corpus %s", source)
    source_sha256 = file_sha256(source)
    embedding_source_sha256 = store.manifest.get("source_sha256")
    if (
        embedding_source_sha256 is not None
        and embedding_source_sha256 != source_sha256
    ):
        raise ValueError(
            "The embedding artifact was built from a different source corpus"
        )

    selected_ids = select_topic_sample(
        source,
        store,
        sample_size,
    )
    article_ids, documents, embeddings = load_topic_inputs(source, store, selected_ids)
    if not article_ids:
        raise ValueError("No articles have both embeddings and source documents")
    if len(article_ids) < min_cluster_size:
        raise ValueError(
            f"Only {len(article_ids)} aligned articles are available, fewer than "
            f"min_cluster_size={min_cluster_size}"
        )

    topic_model = model_factory(
        embedding_model_name=str(store.manifest.get("model", DEFAULT_MODEL)),
        random_seed=random_seed,
        min_cluster_size=min_cluster_size,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        umap_components=umap_components,
        use_keybert_representation=True,
    )
    original_topics, probabilities = topic_model.fit_transform(documents, embeddings)
    original_topics = np.asarray(original_topics, dtype=np.int64)
    discovered_topics = len(
        {int(topic) for topic in original_topics if int(topic) != -1}
    )
    logger.info(
        "BERTopic discovered %s topics and %s outliers in the sample",
        discovered_topics,
        int(np.count_nonzero(original_topics == -1)),
    )

    if np.any(original_topics == -1):
        reduced_topics = np.asarray(
            topic_model.reduce_outliers(
                documents,
                original_topics.tolist(),
                strategy="embeddings",
                embeddings=embeddings,
                threshold=outlier_threshold,
            ),
            dtype=np.int64,
        )
    else:
        reduced_topics = original_topics.copy()
    if reduced_topics.shape != original_topics.shape:
        raise ValueError("BERTopic returned a misaligned topic assignment")
    if not np.array_equal(reduced_topics, original_topics):
        topic_model.update_topics(documents, topics=reduced_topics.tolist())
    assignment_probabilities = topic_probabilities(probabilities, reduced_topics)

    output.mkdir(parents=True, exist_ok=True)
    topic_info = topic_model.get_topic_info()
    labels = topic_labels(topic_info)
    sample_classifications_path = output / "sample_topic_classifications.csv"
    _write_classifications(
        sample_classifications_path,
        article_ids,
        reduced_topics,
        assignment_probabilities,
        labels,
    )

    trainer = classifier_trainer or train_topic_classifier
    classifier, classifier_metrics, cv_records = trainer(
        embeddings,
        reduced_topics,
        random_seed=random_seed,
        cv_folds=classifier_cv_folds,
        jobs=classifier_jobs,
        test_size=classifier_test_size,
    )
    classifier_path = output / "topic_classifier.joblib"
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Topic classifier support requires joblib. Install the topics extras."
        ) from exc
    joblib.dump(classifier, classifier_path)
    classifier_metrics_path = output / "classifier_metrics.json"
    _atomic_json(classifier_metrics_path, classifier_metrics)
    classifier_cv_path = output / "classifier_cv_results.csv"
    _write_classifier_cv_results(classifier_cv_path, cv_records)

    classifications_path = output / "article_topic_classifications.parquet"
    classified_articles, full_topic_counts = _write_full_classifications(
        classifications_path,
        store,
        classifier,
        batch_size=classification_batch_size,
    )
    topic_list_path = output / "topic_list.csv"
    topic_records = _write_topic_list(
        topic_list_path,
        topic_info,
        topic_counts=full_topic_counts,
    )
    detailed_path = output / "detailed_topics.csv"
    _write_detailed_topics(detailed_path, topic_model, topic_records)

    files = [
        classifications_path.name,
        sample_classifications_path.name,
        topic_list_path.name,
        detailed_path.name,
        classifier_path.name,
        classifier_metrics_path.name,
        classifier_cv_path.name,
    ]
    if hierarchy:
        hierarchical_topics = topic_model.hierarchical_topics(documents)
        hierarchy_path = output / "topic_hierarchy.csv"
        hierarchical_topics.to_csv(hierarchy_path, index=False)
        files.append(hierarchy_path.name)
        if visualizations:
            _write_optional_visualization(
                output,
                files,
                "topic_hierarchy.html",
                lambda: topic_model.visualize_hierarchy(
                    hierarchical_topics=hierarchical_topics
                ),
            )

    if save_model:
        model_path = output / "bertopic_model"
        topic_model.save(model_path, serialization="pickle")
        files.append(model_path.name)

    if visualizations:
        _write_optional_visualization(
            output,
            files,
            "topic_words.html",
            lambda: topic_model.visualize_barchart(top_n_topics=20),
        )
        _write_optional_visualization(
            output,
            files,
            "intertopic_distance.html",
            topic_model.visualize_topics,
        )
        probability_values = (
            np.asarray(probabilities) if probabilities is not None else np.empty(0)
        )
        if probability_values.ndim == 2 and len(probability_values):
            _write_optional_visualization(
                output,
                files,
                "topic_distribution_sample.html",
                lambda: topic_model.visualize_distribution(
                    probability_values[0]
                ),
            )

    unique_topics = sorted({int(topic) for topic in reduced_topics if int(topic) != -1})
    outliers = int(full_topic_counts.get(-1, 0))
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "article_assignments": classifications_path.name,
        "articles": classified_articles,
        "sample_articles": len(article_ids),
        "sample_assignments": sample_classifications_path.name,
        "classifier": {
            "algorithm": "MLPClassifier",
            "cross_validation_folds": classifier_metrics["cross_validation_folds"],
            "cv_macro_f1": classifier_metrics["cv_macro_f1"],
            "held_out_macro_f1": classifier_metrics["held_out_macro_f1"],
            "held_out_weighted_f1": classifier_metrics[
                "held_out_weighted_f1"
            ],
            "model": classifier_path.name,
        },
        "classification_batch_size": classification_batch_size,
        "embedding_artifact": str(Path(embeddings_path).expanduser().resolve()),
        "embedding_manifest_sha256": _manifest_digest(store.manifest_path),
        "files": files,
        "hierarchy": hierarchy,
        "min_cluster_size": min_cluster_size,
        "min_dist": min_dist,
        "n_neighbors": n_neighbors,
        "outlier_threshold": outlier_threshold,
        "outliers": outliers,
        "random_seed": random_seed,
        "sample_size": sample_size,
        "source_database": str(source),
        "source_sha256": source_sha256,
        "source_size": source.stat().st_size,
        "topics": len(unique_topics),
        "umap_components": umap_components,
        "umap_metric": metric,
        "visualizations": visualizations,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _write_optional_visualization(
    output: Path,
    files: list[str],
    filename: str,
    build_figure: Callable[[], Any],
) -> None:
    """Write one optional plot without invalidating core topic outputs."""
    try:
        figure = build_figure()
        figure.write_html(output / filename)
    except Exception:
        logger.exception("Skipping optional topic visualization %s", filename)
        return
    files.append(filename)


def finalize_existing_topics(
    db_path: str | Path,
    output_dir: str | Path,
    embeddings_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate completed core files and publish a failed late-stage run."""
    source = Path(db_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Topic manifest already exists: {manifest_path}")

    required = {
        "article_assignments": output
        / "article_topic_classifications.parquet",
        "sample_assignments": output / "sample_topic_classifications.csv",
        "topic_list": output / "topic_list.csv",
        "classifier_model": output / "topic_classifier.joblib",
        "classifier_metrics": output / "classifier_metrics.json",
        "classifier_cv": output / "classifier_cv_results.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot finalize partial topic output; missing: "
            + ", ".join(missing)
        )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Topic recovery requires pyarrow. Install the topics extras."
        ) from exc

    parquet = pq.ParquetFile(required["article_assignments"])
    expected_schema = {
        "article_id": pa.int64(),
        "topic": pa.int32(),
        "probability": pa.float32(),
    }
    for name, expected_type in expected_schema.items():
        field = parquet.schema_arrow.field(name)
        if field.type != expected_type:
            raise ValueError(
                f"Topic column {name!r} has type {field.type}, "
                f"expected {expected_type}"
            )

    topic_counts: Counter[int] = Counter()
    previous_article_id: int | None = None
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=["article_id", "topic"]
    ):
        article_ids = batch.column(0).to_numpy(zero_copy_only=False)
        topics = batch.column(1).to_numpy(zero_copy_only=False)
        if len(article_ids) == 0:
            continue
        if (
            (previous_article_id is not None and article_ids[0] <= previous_article_id)
            or np.any(article_ids[1:] <= article_ids[:-1])
        ):
            raise ValueError(
                "Topic classifications must be strictly ordered by article_id"
            )
        previous_article_id = int(article_ids[-1])
        topic_counts.update(int(topic) for topic in topics)

    with required["topic_list"].open(newline="", encoding="utf-8") as handle:
        labeled_topics = {
            int(row["Topic"]) for row in csv.DictReader(handle)
        }
    missing_labels = sorted(set(topic_counts) - labeled_topics)
    if missing_labels:
        raise ValueError(
            "Topic labels are missing topics: "
            + ", ".join(str(topic) for topic in missing_labels)
        )
    with required["sample_assignments"].open(
        newline="", encoding="utf-8"
    ) as handle:
        sample_articles = sum(1 for _row in csv.DictReader(handle))
    with required["classifier_metrics"].open(encoding="utf-8") as handle:
        classifier_metrics = json.load(handle)

    logger.info("Fingerprinting source corpus %s", source)
    source_sha256 = file_sha256(source)
    files = sorted(
        path.name
        for path in output.iterdir()
        if path.name != "manifest.json" and not path.name.endswith(".tmp")
    )
    candidate_embedding_manifest = (
        Path(embeddings_path).expanduser().resolve() / "manifest.json"
        if embeddings_path is not None
        else None
    )
    embedding_manifest_path = (
        candidate_embedding_manifest
        if candidate_embedding_manifest is not None
        and candidate_embedding_manifest.is_file()
        else None
    )
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "article_assignments": required["article_assignments"].name,
        "articles": int(parquet.metadata.num_rows),
        "classifier": {
            **classifier_metrics,
            "model": required["classifier_model"].name,
        },
        "complete": True,
        "embedding_artifact": (
            str(Path(embeddings_path).expanduser().resolve())
            if embedding_manifest_path is not None
            else None
        ),
        "embedding_manifest_sha256": (
            _manifest_digest(embedding_manifest_path)
            if embedding_manifest_path is not None
            else None
        ),
        "files": files,
        "hierarchy": (output / "topic_hierarchy.csv").is_file(),
        "outliers": int(topic_counts.get(-1, 0)),
        "recovered_from_partial_run": True,
        "sample_articles": sample_articles,
        "sample_assignments": required["sample_assignments"].name,
        "source_database": str(source),
        "source_sha256": source_sha256,
        "source_size": source.stat().st_size,
        "topics": len(set(topic_counts) - {-1}),
        "visualizations": any(
            path.suffix == ".html" for path in output.iterdir()
        ),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def select_topic_sample(
    source: Path,
    store: EmbeddingStore,
    sample_size: int | None,
) -> list[int]:
    """Select sample membership exclusively from the corpus random order."""
    if sample_size is None:
        return list(store.iter_article_ids())
    with _connect_readonly(source) as connection:
        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'articles_order'
            """
        ).fetchone()
        if not table_exists:
            raise ValueError(
                "The source database has no articles_order table. Create it "
                "with `openalex random-order <db-path>` before assigning topics."
            )
        ordered_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM articles_order"
            ).fetchone()[0]
        )
        embedding_count = store.count()
        if ordered_count != embedding_count:
            raise ValueError(
                "articles_order is stale: it contains "
                f"{ordered_count} rows but the embedding artifact contains "
                f"{embedding_count}. Rebuild it with `openalex random-order "
                "<db-path>`."
            )
        rows = connection.execute(
            """
            SELECT article_id
            FROM articles_order
            ORDER BY random_rank
            LIMIT ?
            """,
            (sample_size,),
        )
        selected = [int(row[0]) for row in rows]
    logger.info(
        "Selected %s articles from the corpus articles_order table",
        len(selected),
    )
    return sorted(selected)


def load_topic_inputs(
    database: Path,
    store: EmbeddingStore,
    selected_ids: Sequence[int],
) -> tuple[list[int], list[str], np.ndarray]:
    """Load documents and embeddings in exactly the same article order."""
    documents_by_id: dict[int, str] = {}
    with _connect_readonly(database) as connection:
        for chunk in _chunks(selected_ids, _SQLITE_IN_LIMIT):
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""
                SELECT a.article_id, a.title, ab.abstract
                FROM articles a
                JOIN abstracts ab ON ab.article_id = a.article_id
                WHERE a.article_id IN ({placeholders})
                """,
                list(chunk),
            )
            for article_id, title, abstract in rows:
                documents_by_id[int(article_id)] = _article_text(title, abstract)

    embeddings_by_id = store.get_embeddings_batch(selected_ids)
    article_ids = [
        article_id
        for article_id in selected_ids
        if article_id in documents_by_id and article_id in embeddings_by_id
    ]
    documents = [documents_by_id[article_id] for article_id in article_ids]
    if article_ids:
        embeddings = np.vstack([embeddings_by_id[article_id] for article_id in article_ids])
    else:
        embeddings = np.empty((0, store.dimension or 0), dtype=np.float32)
    return article_ids, documents, embeddings


def topic_probabilities(probabilities, topics: np.ndarray) -> np.ndarray:
    """Return one confidence per final assignment."""
    if probabilities is None:
        return np.full(len(topics), np.nan, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        if values.shape != topics.shape:
            raise ValueError("BERTopic probabilities do not align with assignments")
        return values
    if values.ndim != 2 or values.shape[0] != len(topics):
        raise ValueError("BERTopic probabilities do not align with assignments")
    result = np.empty(len(topics), dtype=float)
    for index, topic in enumerate(topics):
        result[index] = (
            values[index, int(topic)]
            if 0 <= int(topic) < values.shape[1]
            else np.nan
        )
    return result


def train_topic_classifier(
    embeddings: np.ndarray,
    topics: np.ndarray,
    *,
    random_seed: int,
    cv_folds: int,
    jobs: int,
    test_size: float,
) -> tuple[object, dict[str, Any], list[dict[str, Any]]]:
    """Select and evaluate an MLP, then refit it on the complete BERTopic sample."""
    try:
        from sklearn.base import clone
        from sklearn.metrics import classification_report, f1_score
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Topic classifier support requires scikit-learn. "
            "Install the topics extras."
        ) from exc

    all_features = np.asarray(embeddings, dtype=np.float32)
    all_targets = np.asarray(topics, dtype=np.int64)
    semantic_mask = all_targets != -1
    excluded_outliers = int(np.count_nonzero(~semantic_mask))
    features = all_features[semantic_mask]
    targets = all_targets[semantic_mask]
    classes, class_counts = np.unique(targets, return_counts=True)
    if len(classes) < 2:
        raise ValueError("At least two BERTopic classes are required to train the MLP")
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Every BERTopic class needs at least two sampled articles for "
            "stratified classifier evaluation"
        )

    minimum_test_fraction = len(classes) / len(targets)
    effective_test_size = max(test_size, minimum_test_fraction)
    if effective_test_size >= 1:
        raise ValueError("The BERTopic sample is too small for a held-out test set")
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=effective_test_size,
        random_state=random_seed,
        stratify=targets,
    )
    _, training_class_counts = np.unique(train_targets, return_counts=True)
    effective_cv_folds = min(cv_folds, int(training_class_counts.min()))
    if effective_cv_folds < 2:
        raise ValueError(
            "The BERTopic sample is too small for stratified cross-validation"
        )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    batch_size=512,
                    early_stopping=True,
                    max_iter=200,
                    n_iter_no_change=10,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    parameter_grid = {
        "mlp__hidden_layer_sizes": [(128,), (256,), (256, 128)],
        "mlp__alpha": [1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-3],
    }
    search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring={"macro_f1": "f1_macro", "weighted_f1": "f1_weighted"},
        refit="macro_f1",
        cv=effective_cv_folds,
        n_jobs=jobs,
        return_train_score=False,
        verbose=2,
    )
    logger.info(
        "Selecting an MLP with %s-fold cross-validation over %s candidates",
        effective_cv_folds,
        len(parameter_grid["mlp__hidden_layer_sizes"])
        * len(parameter_grid["mlp__alpha"]),
    )
    search.fit(train_features, train_targets)
    test_predictions = search.best_estimator_.predict(test_features)
    held_out_macro_f1 = float(
        f1_score(test_targets, test_predictions, average="macro")
    )
    held_out_weighted_f1 = float(
        f1_score(test_targets, test_predictions, average="weighted")
    )
    report = classification_report(
        test_targets,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )
    best_params = {
        key.removeprefix("mlp__"): value
        for key, value in search.best_params_.items()
    }
    cv_records = []
    for index, params in enumerate(search.cv_results_["params"]):
        cv_records.append(
            {
                "params": {
                    key.removeprefix("mlp__"): value
                    for key, value in params.items()
                },
                "mean_macro_f1": float(
                    search.cv_results_["mean_test_macro_f1"][index]
                ),
                "std_macro_f1": float(
                    search.cv_results_["std_test_macro_f1"][index]
                ),
                "mean_weighted_f1": float(
                    search.cv_results_["mean_test_weighted_f1"][index]
                ),
                "rank_macro_f1": int(
                    search.cv_results_["rank_test_macro_f1"][index]
                ),
            }
        )
    metrics = _json_safe(
        {
            "algorithm": "MLPClassifier",
            "selection_metric": "macro_f1",
            "cross_validation_folds": effective_cv_folds,
            "cv_macro_f1": float(search.best_score_),
            "held_out_macro_f1": held_out_macro_f1,
            "held_out_weighted_f1": held_out_weighted_f1,
            "held_out_articles": len(test_targets),
            "training_articles": len(train_targets),
            "classifier_labeled_articles": len(targets),
            "bertopic_sample_articles": len(all_targets),
            "excluded_sample_outliers": excluded_outliers,
            "best_params": best_params,
            "classification_report": report,
        }
    )
    logger.info(
        "Selected MLP %s; CV macro-F1 %.4f; held-out macro-F1 %.4f; "
        "held-out weighted-F1 %.4f",
        best_params,
        search.best_score_,
        held_out_macro_f1,
        held_out_weighted_f1,
    )
    classifier = clone(search.best_estimator_)
    classifier.set_params(mlp__early_stopping=False)
    classifier.fit(features, targets)
    return classifier, metrics, cv_records


def _write_full_classifications(
    path: Path,
    store: EmbeddingStore,
    classifier,
    *,
    batch_size: int,
) -> tuple[int, Counter]:
    """Predict every embedding with the MLP into compressed Parquet row groups."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Full-corpus topic export requires pyarrow. Install the topics extras."
        ) from exc

    classes = np.asarray(classifier.classes_, dtype=np.int64)
    counts: Counter = Counter()
    processed = 0
    total = store.count()
    schema = pa.schema(
        [
            pa.field("article_id", pa.int64(), nullable=False),
            pa.field("topic", pa.int32(), nullable=False),
            pa.field("probability", pa.float32(), nullable=False),
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=["topic"],
            write_statistics=True,
        ) as writer:
            for article_ids, embeddings in store.iter_batches(batch_size):
                probabilities = np.asarray(
                    classifier.predict_proba(embeddings), dtype=np.float32
                )
                best_indices = probabilities.argmax(axis=1)
                predicted_topics = classes[best_indices].astype(
                    np.int32, copy=False
                )
                predicted_probabilities = probabilities[
                    np.arange(len(article_ids)), best_indices
                ].astype(np.float32, copy=False)
                counts.update(
                    {
                        int(topic): int(count)
                        for topic, count in zip(
                            *np.unique(predicted_topics, return_counts=True),
                            strict=True,
                        )
                    }
                )
                table = pa.Table.from_arrays(
                    [
                        pa.array(article_ids, type=pa.int64()),
                        pa.array(predicted_topics, type=pa.int32()),
                        pa.array(predicted_probabilities, type=pa.float32()),
                    ],
                    schema=schema,
                )
                writer.write_table(table, row_group_size=len(article_ids))
                processed += len(article_ids)
                logger.info(
                    "Classified %s/%s article embeddings",
                    processed,
                    total,
                )
        if processed != total:
            raise RuntimeError(
                f"Classified {processed} embeddings, expected {total}"
            )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return processed, counts


def _write_classifier_cv_results(
    path: Path, records: Sequence[dict[str, Any]]
) -> None:
    fieldnames = [
        "rank_macro_f1",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_weighted_f1",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda value: value["rank_macro_f1"]):
            writer.writerow(
                {
                    **{key: record[key] for key in fieldnames if key != "params"},
                    "params": json.dumps(
                        _json_safe(record["params"]), sort_keys=True
                    ),
                }
            )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def topic_labels(topic_info) -> dict[int, str]:
    labels: dict[int, str] = {-1: "Outlier"}
    for row in topic_info.to_dict("records"):
        labels[int(row["Topic"])] = str(row.get("Name", row["Topic"]))
    return labels


def _write_classifications(
    path: Path,
    article_ids: Sequence[int],
    topics: np.ndarray,
    probabilities: np.ndarray,
    labels: dict[int, str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["article_id", "topic", "probability", "topic_label"],
        )
        writer.writeheader()
        for article_id, topic, probability in zip(
            article_ids, topics, probabilities, strict=True
        ):
            writer.writerow(
                {
                    "article_id": int(article_id),
                    "topic": int(topic),
                    "probability": "" if np.isnan(probability) else float(probability),
                    "topic_label": labels.get(int(topic), f"Topic {int(topic)}"),
                }
            )


def _write_topic_list(
    path: Path,
    topic_info,
    topics: np.ndarray | None = None,
    *,
    topic_counts: Counter | None = None,
) -> list[dict]:
    records = topic_info.to_dict("records")
    if topic_counts is not None:
        counts = {int(topic): int(count) for topic, count in topic_counts.items()}
    elif topics is not None:
        counts = {
            int(topic): int(count)
            for topic, count in zip(
                *np.unique(topics, return_counts=True), strict=True
            )
        }
    else:
        raise ValueError("topics or topic_counts is required")
    normalized = []
    for row in records:
        normalized_row = dict(row)
        normalized_row["Count"] = counts.get(int(row["Topic"]), 0)
        normalized.append(normalized_row)
    fieldnames = list(normalized[0]) if normalized else ["Topic", "Count", "Name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    return normalized


def _write_detailed_topics(path: Path, topic_model, topic_records: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Topic", "Count", "Name", "Top_Words"]
        )
        writer.writeheader()
        for row in topic_records:
            topic = int(row["Topic"])
            if topic == -1:
                continue
            words = topic_model.get_topic(topic) or []
            writer.writerow(
                {
                    "Topic": topic,
                    "Count": int(row.get("Count", 0)),
                    "Name": row.get("Name", f"Topic {topic}"),
                    "Top_Words": "; ".join(
                        f"{word}:{float(score):.3f}" for word, score in words[:20]
                    ),
                }
            )


def _article_text(title: object, abstract: object) -> str:
    title_text = "" if title is None else str(title)
    abstract_text = "" if abstract is None else str(abstract)
    return f"{title_text} . {abstract_text}".strip()


def _connect_readonly(database: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(database.resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _chunks(values: Sequence[int], size: int):
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _manifest_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _output_exists(output: Path) -> bool:
    if not output.exists():
        return False
    return any(output.iterdir()) if output.is_dir() else True


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sample_size(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample size must be an integer or 'all'") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("sample size must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign BERTopic topics using a text-embedding artifact."
    )
    parser.add_argument("--db-path", type=Path, default=Path("articles.db"))
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("output/embeddings"),
        help="Embedding artifact directory or legacy embeddings database.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=_sample_size, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--outlier-threshold", type=float, default=0.1)
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=DEFAULT_CLASSIFICATION_BATCH_SIZE,
        help="Embedding rows predicted per full-corpus MLP batch.",
    )
    parser.add_argument(
        "--classifier-cv-folds",
        type=int,
        default=DEFAULT_CLASSIFIER_CV_FOLDS,
        help="Stratified folds used to select MLP hyperparameters.",
    )
    parser.add_argument(
        "--classifier-jobs",
        type=int,
        default=DEFAULT_CLASSIFIER_JOBS,
        help="Parallel cross-validation jobs; use -1 for every CPU.",
    )
    parser.add_argument(
        "--classifier-test-size",
        type=float,
        default=0.2,
        help="Held-out fraction used only for final classifier evaluation.",
    )
    parser.add_argument("--no-hierarchy", action="store_true")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Validate core files left by a late-stage failure and write the "
            "manifest without refitting"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    if args.finalize_existing:
        manifest = finalize_existing_topics(
            args.db_path,
            args.output_dir,
            args.embeddings_dir,
        )
    else:
        manifest = assign_topics(
            args.db_path,
            args.embeddings_dir,
            args.output_dir,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            min_cluster_size=args.min_cluster_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            umap_components=args.umap_components,
            outlier_threshold=args.outlier_threshold,
            hierarchy=not args.no_hierarchy,
            visualizations=not args.no_visualizations,
            save_model=not args.no_save_model,
            classification_batch_size=args.classification_batch_size,
            classifier_cv_folds=args.classifier_cv_folds,
            classifier_jobs=args.classifier_jobs,
            classifier_test_size=args.classifier_test_size,
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
