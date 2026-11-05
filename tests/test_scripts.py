"""Tests for scripts/preflight.sh and scripts/build_target.sh.

perf and the compiler are replaced by small fake executables, and a local git repository stands
in for the upstream, so these run without perf, hardware counters, a compiler or network.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO / "scripts" / "preflight.sh"
BUILD = REPO / "scripts" / "build_target.sh"
BASH = shutil.which("bash")
URL = "https://example.invalid/upstream.git"

FAKE_PERF = r"""#!/bin/sh
mode=${FAKE_PERF:-hw}
if [ "$1" = --version ]; then
  [ "$mode" = broken ] && { echo "WARNING: perf not found for kernel 9.9.9"; exit 2; }
  echo "perf version 6.9.0"; exit 0
fi
if [ "$mode" = none ]; then
  echo "Error: Access to performance monitoring and observability operations is limited." >&2; exit 255
fi
case "$*" in
  *TopdownL1*)
    [ "$mode" = topdown ] && { echo "0.4,,tma_retiring,1000,100.00,," >&2; exit 0; }
    echo "Cannot find metric or group \`TopdownL1'" >&2; exit 234 ;;
  *cycles,instructions*)
    if [ "$mode" = sw ]; then
      printf '<not supported>,,cycles,0,100.00,,\n<not supported>,,instructions,0,100.00,,\n' >&2
    else
      printf '123456,,cycles:u,1000,100.00,,\n98765,,instructions:u,1000,100.00,0.80,insn per cycle\n' >&2
    fi ;;
  *task-clock*) echo "0.52,msec,task-clock,520000,100.00,0.45,CPUs utilized" >&2 ;;
esac
exit 0
"""

FAKE_CC = r"""#!/bin/sh
[ "$1" = --version ] && { echo "fakecc 1.0"; exit 0; }
out=a.out
while [ $# -gt 0 ]; do [ "$1" = -o ] && { out=$2; shift; }; shift; done
printf '#!/bin/sh\necho ran\n' > "$out" && chmod +x "$out"
"""

TOOLS = ["uname", "awk", "nproc", "id", "head", "tail", "cat", "ls", "grep", "sed", "dirname", "mkdir"]


def executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


# --- preflight ------------------------------------------------------------------

def preflight(tmp_path, perf="hw", compilers=("gcc", "g++")):
    """Run preflight.sh with a PATH holding only basic tools, fake compilers and (maybe) fake perf."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in TOOLS:
        (bin_dir / tool).symlink_to(shutil.which(tool))
    for c in compilers:
        executable(bin_dir / c, f'#!/bin/sh\necho "{c} (Fake) 13.0"\n')
    if perf != "missing":
        executable(bin_dir / "perf", FAKE_PERF)
    out = tmp_path / "pf" / "preflight.json"
    env = {"PATH": str(bin_dir), "FAKE_PERF": perf, "PYTHON": sys.executable}
    r = subprocess.run([BASH, str(PREFLIGHT), "--output", str(out)], env=env, capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_preflight_hardware_and_topdown(tmp_path):
    r, res = preflight(tmp_path, perf="topdown")
    assert r.returncode == 0, r.stderr
    assert res["ok"] and res["errors"] == []
    assert res["events"] == {"cycles": True, "instructions": True, "software": True, "topdown": True}
    assert res["perf"] == {"available": True, "version": "perf version 6.9.0"}
    assert res["compilers"] == {"gcc": "gcc (Fake) 13.0", "g++": "g++ (Fake) 13.0"}
    assert res["host"]["cpus"] >= 1 and res["host"]["arch"]
    assert res["perf_event_paranoid"] == int(Path("/proc/sys/kernel/perf_event_paranoid").read_text())
    assert not any("software" in w or "topdown" in w for w in res["warnings"])


def test_preflight_warns_when_topdown_unsupported(tmp_path):
    r, res = preflight(tmp_path, perf="hw")
    assert r.returncode == 0 and res["ok"]
    assert res["events"]["cycles"] and not res["events"]["topdown"]
    assert any("topdown metrics" in w for w in res["warnings"])
    assert not any("only software events" in w for w in res["warnings"])
    assert "warning: topdown metrics" in r.stderr


def test_preflight_warns_on_software_only_events(tmp_path):
    r, res = preflight(tmp_path, perf="sw")
    assert r.returncode == 0 and res["ok"]
    assert res["events"] == {"cycles": False, "instructions": False, "software": True, "topdown": False}
    assert any("only software events are available" in w for w in res["warnings"])
    assert any("topdown metrics" in w for w in res["warnings"])


@pytest.mark.parametrize("perf,message", [
    ("missing", "perf not found on PATH"),
    ("broken", "perf is on PATH but does not run: WARNING: perf not found for kernel"),
    ("none", "perf cannot measure any events"),
])
def test_preflight_fails_without_usable_perf(tmp_path, perf, message):
    r, res = preflight(tmp_path, perf=perf)
    assert r.returncode == 1
    assert not res["ok"] and any(message in e for e in res["errors"])
    assert f"error: {message}" in r.stderr


def test_preflight_fails_without_compilers(tmp_path):
    r, res = preflight(tmp_path, compilers=())
    assert r.returncode == 1
    assert "no C compiler found (need gcc or clang)" in res["errors"]
    assert "no C++ compiler found (need g++ or clang++)" in res["errors"]


# --- build_target ---------------------------------------------------------------

def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def project(tmp_path):
    """An upstream repo, a pre-seeded checkout of it in .work/lulesh/src, and a config for it."""
    up = tmp_path / "upstream"
    up.mkdir()
    git("init", "-q", "-b", "master", cwd=up)
    (up / "main.c").write_text("int main(void) { return 0; }\n")
    git("add", ".", cwd=up)
    git("commit", "-q", "-m", "v1", cwd=up)
    commit = git("rev-parse", "HEAD", cwd=up)

    work = tmp_path / ".work"
    src = work / "lulesh" / "src"
    src.parent.mkdir(parents=True)
    git("clone", "-q", str(up), str(src), cwd=tmp_path)
    git("remote", "set-url", "origin", URL, cwd=src)  # the config must name an https URL

    cc = executable(tmp_path / "fakecc", FAKE_CC)
    config = {
        "target": "lulesh", "repo_url": URL, "commit": commit, "compiler": str(cc),
        "flags": "-O2 -g -fno-omit-frame-pointer",
        "build": ["mkdir -p build", 'echo "greeting=$GREETING"; "$SAGE_COMPILER" $SAGE_FLAGS -o build/app src/main.c'],
        "binary": "build/app", "run": "{binary}", "env": {"GREETING": "hi"},
    }
    return {"root": tmp_path, "work": work, "src": src, "commit": commit, "config": config}


def build(project, *extra, config=None):
    path = project["root"] / "cfg.yaml"
    path.write_text(yaml.safe_dump(config or project["config"]))
    args = [BASH, str(BUILD), "lulesh", "--config", str(path), "--work", str(project["work"]), *extra]
    return subprocess.run(args, cwd=project["root"], env={**os.environ, "PYTHON": sys.executable},
                          capture_output=True, text=True)


def test_build_records_commit_compiler_flags_and_log(project):
    r = build(project)
    assert r.returncode == 0, r.stderr
    work = project["work"] / "lulesh"
    assert (work / "build" / "app").is_file()
    info = json.loads((work / "build-info.json").read_text())
    assert info["commit"] == project["commit"]
    assert info["repo_url"] == URL
    assert info["compiler_version"] == "fakecc 1.0"
    assert info["flags"] == "-O2 -g -fno-omit-frame-pointer"
    assert info["build_commands"] == project["config"]["build"]
    assert info["binary"] == str(work / "build" / "app")
    assert info["build_log"] == str(work / "build.log")
    log = (work / "build.log").read_text()
    assert f"# commit:   {project['commit']}" in log and "# flags:    -O2 -g -fno-omit-frame-pointer" in log
    assert "$ mkdir -p build" in log and "greeting=hi" in log


def test_build_checks_out_the_pinned_commit(project):
    (project["src"] / "main.c").write_text("int main(void) { return 2; }\n")
    git("commit", "-qam", "local", cwd=project["src"])
    assert build(project).returncode == 0
    assert git("rev-parse", "HEAD", cwd=project["src"]) == project["commit"]


def test_failed_build_is_nonzero_and_points_to_log(project):
    build(project)  # a successful build first: its build-info.json must not survive a failure
    cfg = {**project["config"], "build": ['echo compiling; "$SAGE_COMPILER" $SAGE_FLAGS; echo boom >&2; exit 7']}
    r = build(project, config=cfg)
    log = project["work"] / "lulesh" / "build.log"
    assert r.returncode == 1
    assert "build failed (exit 7)" in r.stderr
    assert f"full log: {log}" in r.stderr
    assert "  | boom" in r.stderr
    assert "boom" in log.read_text()
    assert not (project["work"] / "lulesh" / "build-info.json").exists()


def test_build_refuses_edited_upstream_source(project):
    (project["src"] / "main.c").write_text("// edited\n")
    r = build(project)
    assert r.returncode == 1 and "local changes to tracked upstream files" in r.stderr


def test_build_fails_if_build_edits_upstream_source(project):
    cfg = {**project["config"], "build": project["config"]["build"] + ["echo '// x' >> src/main.c"]}
    r = build(project, config=cfg)
    assert r.returncode == 1 and "the build modified tracked upstream files" in r.stderr


def test_build_errors(project):
    git("remote", "set-url", "origin", "https://example.invalid/other.git", cwd=project["src"])
    r = build(project)
    assert r.returncode == 1 and "was cloned from https://example.invalid/other.git" in r.stderr
    git("remote", "set-url", "origin", URL, cwd=project["src"])

    r = build(project, config={**project["config"], "compiler": "no-such-cc"})
    assert r.returncode == 1 and "compiler 'no-such-cc' not found" in r.stderr

    r = build(project, config={**project["config"], "flags": "-O0 -g"})
    assert r.returncode == 2 and "flags must enable optimization" in r.stderr


def test_clean_starts_from_scratch(project):
    stale = project["work"] / "lulesh" / "stale.o"
    stale.write_text("old")
    r = build(project, "--clean")  # the clone of the (unreachable) URL fails, but only after cleaning
    assert r.returncode == 1 and "git clone of" in r.stderr
    assert not stale.exists()


def test_build_usage_errors(tmp_path):
    r = subprocess.run([BASH, str(BUILD)], capture_output=True, text=True)
    assert r.returncode == 2 and "usage:" in r.stderr
    r = subprocess.run([BASH, str(BUILD), "lulesh", "--bogus"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown option" in r.stderr
