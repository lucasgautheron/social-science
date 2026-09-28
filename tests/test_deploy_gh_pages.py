import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "scripts" / "deploy-gh-pages.sh"


def run(command, *, cwd, env=None, check=True):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def git(cwd, *args):
    return run(["git", *args], cwd=cwd)


def test_deploy_replaces_gh_pages_history(tmp_path):
    root = tmp_path / "source"
    remote = tmp_path / "remote.git"
    events = tmp_path / "events"
    fake_python = tmp_path / "python"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", "--initial-branch=main", str(root))
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "README").write_text("source\n", encoding="utf-8")
    git(root, "add", "README")
    git(root, "commit", "-m", "initial")
    git(root, "remote", "add", "origin", str(remote))
    events.mkdir()
    (events / "manifest.json").write_text("{}\n", encoding="utf-8")
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output.mkdir(parents=True)
counter = Path(os.environ["SITE_COUNTER"])
value = int(counter.read_text() if counter.exists() else "0") + 1
counter.write_text(str(value), encoding="utf-8")
(output / "index.html").write_text(f"site {value}\\n", encoding="utf-8")
(output / "dendrogram.html").write_text("tree\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "OPENALEX_ROOT": str(root),
            "PYTHON": str(fake_python),
            "SITE_COUNTER": str(tmp_path / "counter"),
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    run([str(DEPLOY), str(events)], cwd=root, env=env)
    first = git(remote, "rev-parse", "gh-pages").stdout.strip()
    run([str(DEPLOY), str(events)], cwd=root, env=env)
    second = git(remote, "rev-parse", "gh-pages").stdout.strip()
    assert first != second
    assert git(remote, "rev-list", "--count", "gh-pages").stdout.strip() == "1"
