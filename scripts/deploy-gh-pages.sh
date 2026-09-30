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
PUBLISH="$TEMP_ROOT/gh-pages"
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
if [[ -n "${OPENALEX_NEW_LINK_VISUALIZATIONS_DIR:-}" ]]; then
  if [[ ! -f "$OPENALEX_NEW_LINK_VISUALIZATIONS_DIR/cluster_link_distance_summary.csv" ]]; then
    echo "New-link summary not found: $OPENALEX_NEW_LINK_VISUALIZATIONS_DIR/cluster_link_distance_summary.csv" >&2
    exit 2
  fi
  BUILD_ARGS+=(
    --new-link-visualizations-dir "$OPENALEX_NEW_LINK_VISUALIZATIONS_DIR"
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

git init --initial-branch=gh-pages "$PUBLISH" >/dev/null
if name="$(git -C "$ROOT" config user.name 2>/dev/null)" && [[ -n "$name" ]]; then
  git -C "$PUBLISH" config user.name "$name"
fi
if email="$(git -C "$ROOT" config user.email 2>/dev/null)" && [[ -n "$email" ]]; then
  git -C "$PUBLISH" config user.email "$email"
fi
cp -R "$SITE"/. "$PUBLISH"/

python - "$PUBLISH" <<'PY'
import os
from pathlib import Path
import sys

limit = int(os.environ.get("GITHUB_FILE_LIMIT", 100 * 1024 * 1024))
root = Path(sys.argv[1])
too_large = [
    path for path in root.rglob("*")
    if path.is_file() and ".git" not in path.parts and path.stat().st_size >= limit
]
if too_large:
    print("Refusing to publish files at or over GitHub's 100MB limit:", file=sys.stderr)
    for path in too_large:
        print(f"  {path.relative_to(root)}", file=sys.stderr)
    raise SystemExit(1)
PY

git -C "$PUBLISH" add --all
git -C "$PUBLISH" commit --quiet -m "Update event website"
REMOTE_URL="$(git -C "$ROOT" remote get-url "$REMOTE")"
git -C "$PUBLISH" remote add origin "$REMOTE_URL"
git -C "$PUBLISH" push --force --set-upstream origin HEAD:gh-pages
git -C "$ROOT" fetch --force "$REMOTE" "gh-pages:gh-pages"
echo "Published event website from $EVENTS_DIR to $REMOTE/gh-pages."
