#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 SITE_DIR [GIT_REMOTE]" >&2
  exit 2
fi

ROOT="${OPENALEX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SITE_INPUT="$1"
REMOTE="${2:-origin}"
if [[ "$SITE_INPUT" = /* ]]; then
  SITE="$SITE_INPUT"
else
  SITE="$(cd "$(dirname "$SITE_INPUT")" && pwd)/$(basename "$SITE_INPUT")"
fi
if [[ ! -f "$SITE/index.html" ]]; then
  echo "Website entry point not found: $SITE/index.html" >&2
  exit 2
fi

TEMP_ROOT="$(mktemp -d)"
PUBLISH="$TEMP_ROOT/gh-pages"
trap 'rm -rf "$TEMP_ROOT"' EXIT

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
echo "Published website from $SITE to $REMOTE/gh-pages."
