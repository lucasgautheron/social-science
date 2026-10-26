#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 EVENTS_DIR [GIT_REMOTE]" >&2
  exit 2
fi

ROOT="${OPENALEX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVENTS_INPUT="$1"
REMOTE="${2:-origin}"
if [[ "$EVENTS_INPUT" = /* ]]; then
  EVENTS_DIR="$EVENTS_INPUT"
else
  EVENTS_DIR="$(cd "$(dirname "$EVENTS_INPUT")" && pwd)/$(basename "$EVENTS_INPUT")"
fi
if [[ ! -f "$EVENTS_DIR/manifest.json" ]]; then
  echo "Event manifest not found: $EVENTS_DIR/manifest.json" >&2
  exit 2
fi

TEMP_ROOT="$(mktemp -d)"
SITE="$TEMP_ROOT/site"
trap 'rm -rf "$TEMP_ROOT"' EXIT

BUILD_ARGS=(
  --events-dir "$EVENTS_DIR"
  --output-dir "$SITE"
)
if [[ -n "${OPENALEX_CLUSTERS_DIR:-}" ]]; then
  if [[ ! -f "$OPENALEX_CLUSTERS_DIR/manifest.json" ]]; then
    echo "Cluster manifest not found: $OPENALEX_CLUSTERS_DIR/manifest.json" >&2
    exit 2
  fi
  BUILD_ARGS+=(--clusters-dir "$OPENALEX_CLUSTERS_DIR")
fi
if [[ -n "${OPENALEX_DB_PATH:-}" ]]; then
  if [[ ! -f "$OPENALEX_DB_PATH" ]]; then
    echo "Database not found: $OPENALEX_DB_PATH" >&2
    exit 2
  fi
  BUILD_ARGS+=(--db-path "$OPENALEX_DB_PATH")
fi
if [[ -n "${OPENALEX_CLUSTER_TRENDS:-}" ]]; then
  if [[ ! -f "$OPENALEX_CLUSTER_TRENDS" ]]; then
    echo "Cluster trends not found: $OPENALEX_CLUSTER_TRENDS" >&2
    exit 2
  fi
  BUILD_ARGS+=(--trends "$OPENALEX_CLUSTER_TRENDS")
fi
if [[ -n "${OPENALEX_NEW_LINK_VISUALIZATIONS_DIR:-}" ]]; then
  if [[ ! -f "$OPENALEX_NEW_LINK_VISUALIZATIONS_DIR/cluster_link_distance_summary.csv" ]]; then
    echo "New-link summary not found: $OPENALEX_NEW_LINK_VISUALIZATIONS_DIR/cluster_link_distance_summary.csv" >&2
    exit 2
  fi
  BUILD_ARGS+=(
    --new-link-visualizations-dir "$OPENALEX_NEW_LINK_VISUALIZATIONS_DIR"
  )
fi
if [[ -n "${OPENALEX_CLUSTER_TOPIC_VISUALIZATIONS_DIR:-}" ]]; then
  if [[ ! -f "$OPENALEX_CLUSTER_TOPIC_VISUALIZATIONS_DIR/cluster_topic_summary.csv" ]]; then
    echo "Cluster-topic summary not found: $OPENALEX_CLUSTER_TOPIC_VISUALIZATIONS_DIR/cluster_topic_summary.csv" >&2
    exit 2
  fi
  BUILD_ARGS+=(
    --cluster-topic-visualizations-dir "$OPENALEX_CLUSTER_TOPIC_VISUALIZATIONS_DIR"
  )
fi
if [[ -n "${OPENALEX_INCIDENCE_WORKERS:-}" ]]; then
  if [[ ! "$OPENALEX_INCIDENCE_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "OPENALEX_INCIDENCE_WORKERS must be a positive integer" >&2
    exit 2
  fi
  BUILD_ARGS+=(--incidence-workers "$OPENALEX_INCIDENCE_WORKERS")
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "${PYTHON:-python}" -m openalex.website.build \
  "${BUILD_ARGS[@]}"
bash "$ROOT/scripts/publish-gh-pages.sh" "$SITE" "$REMOTE"
