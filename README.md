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

## Cluster events

Cluster those keywords with a nested degree-corrected stochastic block model.
`graph-tool` is installed from conda-forge, separately from the pip extras:

```bash
conda install -c conda-forge graph-tool
openalex cluster-events \
  --events-dir output/events \
  --output-dir output/event_clusters
```

The fit treats each off-diagonal co-occurrence count as an undirected edge
multiplicity and keeps the shortest description length across `--restarts`.
Level 0 is the finest partition; coarser cuts of the same hierarchy are written
too. A paper that contains several keywords from one cluster contributes once
to that cluster's yearly count.

The default vocabulary is the keywords listed in `events.csv`. `--keywords all`
clusters every keyword in the co-occurrence vocabulary. `--level` selects which
hierarchy depth is reported as `group` in `keyword_groups.csv`.

## Website

Build the three-page site:

```bash
openalex build-website \
  --events-dir output/events \
  --clusters-dir output/event_clusters \
  --output-dir output/website
python -m http.server --directory output/website 8000
```

`index.html` ranks keywords by document frequency. `dendrogram.html` shows the
complete-linkage keyword hierarchy and a sidebar with exact yearly paper
frequency plus cluster membership. `graph.html` switches between the keyword
co-occurrence network and its blockmodel clusters. Nodes share cluster colors
and have area proportional to keyword document frequency (summed for cluster
nodes). Cluster coordinates are the document-frequency-weighted barycenters of
their displayed keywords.

The builder filters words below `--min-document-frequency` (default 10),
normalizes nonzero co-occurrence rows to unit L2 norm, uses cosine distance,
and cuts complete linkage at `--cluster-similarity 0.5`. If `H` is the cut
membership matrix, it also exports `H.T @ M @ H` and verifies that the total
count is preserved. The graph page keeps the 5,000 most frequent blockmodel
keywords and their 10,000 strongest co-occurrence edges by default; use
`--max-graph-keywords` and `--max-graph-edges` to change those limits.

Omitting `--clusters-dir` still builds the site, but the graph page displays
instructions instead of a network.

Publish a fresh orphan `gh-pages` commit with:

```bash
OPENALEX_CLUSTERS_DIR=output/event_clusters \
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

`status` reads the durable run state from S3. Add `--live` to query the worker
directly through SSM and display the process state, fast-storage usage, database
cache, and recent launcher/stdout/stderr lines:

```bash
openalex-aws status --live --lines 30
```

Live output normalizes carriage-return progress displays and byte-caps each
log so the SSM response still includes the current stderr tail.

Enable email notifications for successful and failed runs once:

```bash
openalex-aws notifications --email lucas.gautheron@gmail.com
```

AWS sends a subscription confirmation email that must be accepted before
notifications are delivered. Messages include the run ID, command, exit code,
and S3 links for status, logs, and output. Bootstrap failures are also reported
and no longer leave the durable status incorrectly marked as running.

To upgrade a completed pre-refactor run without repeating its temporal
counting pass, keep the original instance running and point the new command at
its legacy checkpoint:

```bash
openalex-aws submit -- openalex events \
  --checkpoint-path /mnt/aws-runner/repo/output/variations_checkpoint.pkl \
  --output-dir output/events \
  --rebuild-artifacts
```

This reuses the version-1 event counts and reruns only the corpus pass needed
to produce co-occurrence plus exact paper/year incidence. If the instance was
stopped and the excluded checkpoint was not separately downloaded, its local
NVMe copy no longer exists and the counting pass cannot be recovered.

Use `--no-database` before `--` for commands that need no corpus. The worker
caches the S3 database under `/mnt/aws-runner` on the instance's fast local
NVMe storage. Before launching a run, the worker discovers the EC2
instance-store devices, stripes multiple devices as RAID 0, mounts the result
at `/mnt/aws-runner`, and refuses to fall back to root EBS. It validates the
cached URI, size, ETag, and VersionId before reuse. Downloads are atomic and
`--force-db-download` explicitly refreshes the cache. The cache is reused
across calls while the instance remains running, but EC2 instance-store data
is lost on stop or termination and is then repopulated from S3. Normal runs
receive a read-only symlink to the cached database and write only to an
isolated per-run `output/` directory.

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
ruff check src tests
```
