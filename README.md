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

Install BERTopic support separately when assigning topics:

```bash
python -m pip install -e '.[topics]'
```

List commands with `openalex --help`, then inspect any command directly:

```bash
openalex events --help
openalex build-website --help
```

## Text embeddings and topics

Build local multilingual MiniLM embeddings from each article's title and
abstract. The corpus is opened read-only; vectors and their versioned manifest
are written to a separate, resumable artifact directory:

```bash
openalex embeddings \
  --db-path /path/to/articles.db \
  --output-dir output/embeddings \
  --resume
```

Embedding execution is selected automatically. CPU hosts use 32 persistent
worker processes by default. CUDA hosts use one persistent process per visible
GPU (one process on the L4 `g6.8xlarge`) and a default encode batch size of 256;
MPS uses one process. Each process owns one model, and encoded chunks are
committed to SQLite and checkpointed in order. Override detection with
`--device`, `--workers`, or `--encode-batch-size`.

`output/embeddings/embeddings.db` supports exact retrieval by article ID.
`manifest.json` records the model, dimension, text format, and completion
state. The artifact can be staged for a remote run with
`openalex-aws submit --input output/embeddings -- ...`.

Assign BERTopic topics from those embeddings:

```bash
openalex topics \
  --db-path /path/to/articles.db \
  --embeddings-dir output/embeddings \
  --output-dir output/topics
```

BERTopic discovers topics on a deterministic random sample of 100,000
articles selected exclusively from the corpus `articles_order` table. The
command rejects missing or stale random-order tables instead of silently using
a different sample. Topic discovery uses EOM cluster selection with a minimum
topic size of 25, tuned for the 100,000-article sample; use
`--min-cluster-size` to change the minimum. BERTopic does not apply a second
automatic topic-reduction pass. A standardized MLP is then selected by
stratified cross-validation,
evaluated on a held-out portion of that sample, refit on all sample labels, and
used to classify every stored embedding in chunks. Results include full-corpus
and sample assignments, macro/weighted F1 metrics, cross-validation results,
the fitted classifier, topic labels and words, a hierarchy, visualizations,
the saved BERTopic model, and a manifest tied to the embedding artifact. The
canonical `article_topic_classifications.parquet` contains MLP predictions for
every article—including the training sample—as `int64` article IDs, `int32`
topics, and `float32` confidence, compressed with Zstandard. The separate
sample CSV preserves BERTopic's training targets for auditing. Use
`--sample-size` to change the discovery sample and `--no-hierarchy`,
`--no-visualizations`, or `--no-save-model` to omit optional outputs.
These full-corpus classifier outputs use topic artifact schema version 2.

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

Cluster those keywords with the same complete-linkage dendrogram as the
website. Each keyword is an L2-normalized co-occurrence row, and distance is
cosine distance. The cosine similarity threshold is the coarsest cut that
still reaches `--n-clusters` (default 20): every join kept in a cluster has
similarity at least that high. Equal merge heights can skip a count; the cut
then uses the next finer partition. A paper that contains several keywords
from one cluster contributes once to that cluster's yearly count.

```bash
openalex cluster-events \
  --events-dir output/events \
  --output-dir output/event_clusters
```

The default vocabulary is the keywords listed in `events.csv`. When
`filtered_events/classifications.csv` sits beside that events directory, or
`--filtered-dir` points at filter-events output, only keywords labelled
genuine are clustered. `--keywords all` keeps the full co-occurrence
vocabulary. `--level` selects which hierarchy depth is reported as `group`
in `keyword_groups.csv`. The dendrogram cut has a single level, 0.

## Filter events

Classify each extracted keyword as a genuine scientific term or a spurious
artefact. The classifier is GPT-6 Luna (`gpt-6-luna`). Each prompt shows the
keyword and the three other keywords with the most similar co-occurrence
profiles: cosine similarity of L2-normalized co-occurrence rows, the same
similarity used to build the keyword dendrogram.

```bash
openalex filter-events \
  --events-dir output/events \
  --output-dir output/filtered_events
```

The command reads `OPENAI_API_KEY`. Requests start at least 200 ms apart
(`--request-interval`), which keeps a run under GPT-6 Luna's Tier 1 limit of
500 requests and 500,000 tokens per minute. `classifications.csv` records the
label, a short reason, and those three neighbors. `classifications.jsonl` is
the checkpoint; `--resume` skips keywords already classified.

`cluster-events` and `build-website` use the genuine keywords from that file
when it is available. `--keywords all` on the cluster command keeps every
co-occurrence keyword.

## New coauthorship links

Build every author pair in its first coauthorship year, its distance in the
cumulative network through the preceding year, and a uniquely attributable
event cluster:

```bash
openalex new-links \
  --db-path /path/to/articles.db \
  --events-dir output/events \
  --clusters-dir output/event_clusters \
  --output-dir output/new_links
```

All new links are added to the cumulative graph. Results retain simple random
samples of up to 2,000 no-cluster links per year and 2,000 links per
attributed cluster and year; change these with `--no-cluster-sample` and
`--cluster-sample`. A separate sample of up to 2,000 cluster-paper pair
observations per year supplies all-link reference distributions; change it
with `--baseline-sample`. Reproduce every sample with `--sampling-seed`. They are
partitioned as `years/<year>.npz`, with int32 `author_i` and `author_j` indices
into `author_ids.npy`, int32 exact `distance`, int32 `cluster_id`, and float64
`sampling_weight`. Distance `-1` means the authors were disconnected. Cluster
`-1` means no unique cluster is present on every paper responsible for that
first-year link. Use `sampling_weight` when estimating totals. Exact
population/sample counts and inclusion probabilities are in `manifest.json`.
Event artifacts produced before article-id incidence sidecars were added must
be regenerated. New-link artifacts before version 7 lack year-matched
reference samples and must also be rebuilt.

Both `network` and `new-links` skip papers with more than 16 authors by
default; change this with `--max-authors`. Exact distances use bidirectional
BFS for sources with few targets and grouped BFS otherwise. `new-links` runs
16 source-aligned distance workers by default; change this with
`--distance-workers`. Each worker uses scratch space proportional to the
author count while sharing the cumulative graph. Per-year search
counts, visited nodes, inspected edges, and timing are recorded under
`distance_stats` in the new-link manifest.

Plot cluster paper counts on a logarithmic x-axis against connected-only mean
distance:

```bash
openalex visualize-new-links \
  --new-links-dir output/new_links \
  --output-dir output/new_link_visualizations \
  --bootstrap-replicates 500 \
  --permutation-replicates 1000
```

This writes separate scatter plots for first links and for all coauthor-pair
observations on cluster papers. In the latter, pairs already linked before the
paper contribute distance one. All connected observations use one uniform
reservoir of up to `--cluster-sample` observations per cluster across all
years. Exact connected denominators and reservoir-weighted distance sums keep
the plotted means design-unbiased. (The first-link output above remains sampled
per cluster and year.) Disconnected pairs are excluded from both means and
reported separately in `cluster_link_distance_summary.csv`.

Each cluster receives its own year-matched reference. First-link references
use no-cluster links; all-link references use cluster-paper observations from
the other clusters (leave-one-cluster-out). Within-year reference
distributions are post-stratified to the selected cluster's edge-year mix, so
the comparison does not inherit the corpus-wide year distribution. The output
reports disconnection risk separately from finite connected distances and
reports repeat-collaboration share separately for all links.

Finite-distribution effects include the signed mean-hop shift and
Wasserstein-1 distance. Uncertainty uses year-stratified nonparametric
bootstrap intervals; year-stratified permutation tests produce Monte Carlo
p-values, with Benjamini-Hochberg q-values across clusters. Set
`--inference-seed` for reproducible inference; `--inference-workers` controls
the parallel cluster comparisons (default 8). These are year-adjusted
descriptive comparisons, not causal cluster effects: author seniority,
network degree, team composition, and other cluster-specific factors remain
possible confounders. Points are gray;
within each of four size quantiles, the most positive and negative residuals
from a regression on log paper count are highlighted in red and labeled with
the cluster's highest-frequency keyword.

## Website

Build the site:

```bash
openalex build-website \
  --events-dir output/events \
  --clusters-dir output/event_clusters \
  --new-link-visualizations-dir output/new_link_visualizations \
  --db-path /path/to/articles.db \
  --output-dir output/website
python -m http.server --directory output/website 8000
```

The site root opens `clusters.html`, which lists every cluster from
`cluster-events`. A cluster's size is the number of documents
that contain any of its keywords, divided by the total number of documents,
and each row plots that share by year. `link-distances.html` interactively
plots cluster paper count on a logarithmic x-axis against the two mean-distance
measures from `visualize-new-links`. Its points use one neutral style rather
than the static plots' residual highlights; hovering a point shows the cluster,
its share-of-papers curve by year, the mean connected distance by year for the
selected measure, and its discrete connected-link
distance distribution (1, 2, ...) overlaid with that cluster's year-matched
reference. The details also report disconnection and distribution effect
sizes with FDR-adjusted q-values. The
cluster search matches labels, member keywords, and cluster ids, and filters
the scatter points immediately.

When genuine classifications are available, the cluster list uses those
keywords. Artefact-labelled members are left out of a cluster.

The builder filters words below `--min-document-frequency` (default 10),
normalizes nonzero co-occurrence rows to unit L2 norm, uses cosine distance,
and cuts complete linkage at `--cluster-similarity 0.5`. If `H` is the cut
membership matrix, it also exports `H.T @ M @ H` and verifies that the total
count is preserved.

Omitting `--clusters-dir` still builds the site, but the cluster page explains
that no clusters were supplied. `--db-path` divides each yearly frequency
by the number of articles published that year. The count is one read-only
`GROUP BY publication_year` over `idx_publication_year`, so the article rows
are not read. Without `--db-path`, the curves use the processed-paper totals
stored in the event manifest. Omitting `--new-link-visualizations-dir` leaves
the link-distance page in an empty state. `--trends` is the cluster-trends
CSV that classifies those points. Without it, the builder uses
`cluster_trends.csv` inside `--clusters-dir` when that file exists.
`scripts/deploy-website.slurm` passes the file only in that case.

Publish a fresh orphan `gh-pages` commit with:

```bash
OPENALEX_CLUSTERS_DIR=output/event_clusters \
OPENALEX_NEW_LINK_VISUALIZATIONS_DIR=output/new_link_visualizations \
OPENALEX_DB_PATH=/path/to/articles.db \
OPENALEX_CLUSTER_TRENDS=output/event_clusters/cluster_trends.csv \
  scripts/deploy-gh-pages.sh output/events
```

## AWS

Provision the default CPU worker once:

```bash
openalex-aws setup --worker cpu --db-path /path/to/articles.db
```

`setup` creates and maintains the default EC2 role and instance profile with
SSM access and S3 access limited to the configured bucket prefix. When an
existing worker lacks that profile, rerunning `setup` attaches it in place; the
instance is not replaced. Pass `--iam-instance-profile` to use a pre-existing
custom profile instead.

The CPU profile is the existing `i4i.8xlarge`. An existing flat state file is
backed up and migrated without changing its instance ID. Provision the optional
L4 profile independently; this keeps the CPU instance, whether running or
stopped:

```bash
openalex-aws setup --worker gpu --skip-db-upload
openalex-aws workers
```

The GPU profile defaults to `g6.8xlarge` and AWS's current Amazon Linux 2023
NVIDIA-driver DLAMI. `workers` lists both instance IDs and their live states.
Lifecycle commands affect only the selected worker:

```bash
openalex-aws start --worker gpu --wait
openalex-aws pause --worker gpu --wait
openalex-aws start --worker cpu --wait
```

Run any installed command by placing it after `--`. CPU is the default worker:

```bash
openalex-aws submit --worker cpu -- openalex events --output-dir output/events
openalex-aws status
openalex-aws logs
openalex-aws artifacts
openalex-aws download
```

Run embeddings on the L4 with one CUDA process:

```bash
openalex-aws submit --worker gpu -- openalex embeddings \
  --db-path articles.db \
  --output-dir output/embeddings \
  --resume
```

Every run records its worker, instance ID, and instance type. Live status,
artifact refresh, and cancellation therefore continue to target the original
instance even after another worker is selected. The local last-run file retains
the latest run globally and per worker; pass `--worker cpu` or `--worker gpu`
to inspect that worker's latest run.

`download` writes artifacts into the local `output/` directory by default,
preserving their paths. It asks for confirmation when the destination already
exists; pass `--yes` to confirm non-interactively or `--output-dir` to choose a
different destination.

Stage local artifact directories from the submitting machine with repeatable
`--input` options. Each directory must contain `manifest.json`; it is
content-addressed in S3, cached on the worker's NVMe storage, and exposed at
the same relative path inside the isolated run:

```bash
openalex-aws submit \
  --input output/events \
  --input output/event_clusters \
  -- openalex new-links \
  --events-dir output/events \
  --clusters-dir output/event_clusters
```

The first submission publishes and downloads each artifact. Later submissions
with the same manifests reuse both the S3 objects and the worker cache without
transferring their contents again. The cache is repopulated from S3 after the
instance is stopped or terminated. Staged paths beneath `output/` are excluded
when run results are synchronized, so inputs are not copied into every run's
output prefix.

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
