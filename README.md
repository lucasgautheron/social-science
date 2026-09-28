# OpenAlex corpus tools

This repository packages the OpenAlex corpus workflows as the `openalex`
Python package. The existing SQLite corpus is an immutable input to analysis
and website commands; packaging and site builds do not migrate or modify it.

## Install

```bash
python -m pip install -e '.[analysis,website,parquet,download,aws,dev]'
```

Heavy sentence-transformer support is separate:

```bash
python -m pip install -e '.[embeddings]'
```

List commands with `openalex --help`, then inspect any command directly:

```bash
openalex events --help
openalex build-website --help
```

## Corpus and event pipeline

Database-producing and maintenance commands require an explicit writable
target. Analysis commands read `articles.db` without changing it.

Create a deterministic order on a disposable/writable database:

```bash
openalex random-order /path/to/writable-copy.db
```

Extract temporal event keywords and website-ready sparse artifacts:

```bash
openalex events \
  --db-path /path/to/articles.db \
  --output-dir output/events
```

The output contains:

- `events.csv` and `events_by_year.csv`;
- `event_cooccurrence.npz`, `event_vocabulary.npy`, and
  `event_document_frequency.npy`;
- chunked Boolean paper-keyword incidence and publication years under
  `incidence/`;
- `manifest.json`, which versions and describes the artifacts.

Incidence is retained so a cluster frequency counts a paper once even when the
paper contains multiple keywords in that cluster.

## Website

Build the two-page site:

```bash
openalex build-website \
  --events-dir output/events \
  --output-dir output/website
python -m http.server --directory output/website 8000
```

`index.html` ranks keywords by document frequency. `dendrogram.html` shows the
complete-linkage keyword hierarchy and a sidebar with exact yearly paper
frequency plus cluster membership.

The builder filters words below `--min-document-frequency` (default 10),
normalizes nonzero co-occurrence rows to unit L2 norm, uses cosine distance,
and cuts complete linkage at `--cluster-similarity 0.5`. If `H` is the cut
membership matrix, it also exports `H.T @ M @ H` and verifies that the total
count is preserved.

Publish a fresh orphan `gh-pages` commit with:

```bash
scripts/deploy-gh-pages.sh output/events
```

## AWS

Provision once:

```bash
openalex-aws setup --db-path /path/to/articles.db
```

Run any installed command by placing it after `--`:

```bash
openalex-aws submit -- openalex events --output-dir output/events
openalex-aws status
openalex-aws logs
openalex-aws artifacts
openalex-aws download
```

Use `--no-database` before `--` for commands that need no corpus. The worker
caches the S3 database under `/mnt/aws-runner` on the instance's fast local
NVMe storage and validates the cached URI, size, ETag, and VersionId before
reuse. Downloads are atomic and `--force-db-download` explicitly refreshes the
cache. The cache is reused across calls while the instance remains running,
but EC2 instance-store data is lost on stop or termination and is then
repopulated from S3. Normal runs receive a read-only symlink to the cached
database and write only to an isolated per-run `output/` directory.

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check src tests
```
