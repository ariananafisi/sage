"""Tests for scripts/profile_target.sh and scripts/annotate_symbol.sh.

A fake `perf` replays saved perf output from tests/fixtures (and can be told to fail), a fake
`mpirun` starts N copies with OMPI_COMM_WORLD_RANK set, and the "application" is a shell script,
so these need no perf access, hardware counters, MPI or compiler.
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
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROFILE = REPO / "scripts" / "profile_target.sh"
ANNOTATE = REPO / "scripts" / "annotate_symbol.sh"
BASH = shutil.which("bash")

# Fake perf: `stat ... -o FILE -- CMD` runs CMD 3 times and writes the stat fixture; `record ... -o FILE
# -- CMD` runs CMD and writes FILE; `report -i FILE` prints the report fixture; `annotate -s SYM` prints
# the annotate fixture for ljForce only. FAKE_PERF_FAIL=<subcommand> makes that subcommand fail.
FAKE_PERF = r"""#!/usr/bin/env bash
sub=$1; shift
if [[ $sub == --version ]]; then echo "perf version 6.9.0-fake"; exit 0; fi
if [[ ${FAKE_PERF_FAIL:-} == "$sub" ]]; then echo "perf $sub: fake failure: Permission denied" >&2; exit 3; fi
out="" in="" sym=""
while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in -o) out=$2; shift ;; -i) in=$2; shift ;; -s) sym=$2; shift ;; esac
  shift
done
[[ ${1:-} == -- ]] && shift
case $sub in
  stat)
    rc=0; for i in 1 2 3; do "$@" || rc=$?; done
    cp "$FIXTURES/perf-stat-software.txt" "$out"; exit $rc ;;
  record)
    "$@"; rc=$?
    echo "[ perf record: Captured and wrote 0.048 MB $out (448 samples) ]" >&2
    printf 'PERFILE2 fake for %s\n' "$*" > "$out"; exit $rc ;;
  report)
    [[ -s $in ]] || { echo "perf report: no $in" >&2; exit 1; }
    cat "${FAKE_REPORT:-$FIXTURES/perf-report.txt}" ;;
  annotate)
    [[ $sym == ljForce ]] && cat "$FIXTURES/perf-annotate.txt"
    [[ $sym == ljForce ]] || echo "Error: The perf.data data has no samples!" >&2 ;;
esac
exit 0
"""

FAKE_MPIRUN = r"""#!/usr/bin/env bash
[[ $1 == -np ]] || { echo "fake mpirun: expected -np" >&2; exit 2; }
n=$2; shift 2
pids=()
for ((r = 0; r < n; r++)); do OMPI_COMM_WORLD_RANK=$r "$@" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit $rc
"""

APP = r"""#!/bin/sh
echo "app stdout rank=${OMPI_COMM_WORLD_RANK:-none} threads=${OMP_NUM_THREADS:-unset} args=$*"
echo "app stderr line" >&2
[ "$1" = --fail ] && exit 5
exit 0
"""


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def env(tmp_path):
    """A project root with a 'built' target (serial and MPI), fake perf/mpirun, and a config."""
    root = tmp_path / "project"
    work = root / ".work" / "comd"
    executable(work / "src/bin/CoMD-serial", APP)
    executable(work / "src/bin/CoMD-mpi", APP)
    (work / "build-info.json").write_text(json.dumps({"variant": "serial", "commit": "c" * 40}))
    (work / "build-info-mpi.json").write_text(json.dumps({"variant": "mpi", "commit": "c" * 40}))
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "perf", FAKE_PERF)
    executable(bin_dir / "mpirun", FAKE_MPIRUN)
    config = {
        "target": "comd", "repo_url": "https://example.invalid/comd.git", "commit": "c" * 40,
        "compiler": "gcc", "flags": "-O3 -g -fno-omit-frame-pointer",
        "build": ['make CC="$SAGE_COMPILER" OPTFLAGS="$SAGE_FLAGS"'], "binary": "src/bin/CoMD-serial",
        "run": "{binary} -N 5", "env": {"OMP_NUM_THREADS": 3},
        "mpi": {"compiler": "mpicc", "build": ['make CC="$SAGE_COMPILER" OPTFLAGS="$SAGE_FLAGS"'],
                "binary": "src/bin/CoMD-mpi", "launcher": "mpirun -np {ranks}", "ranks": 2,
                "run": "{binary} -i 2 -j 1 -k 1", "env": {"OMP_NUM_THREADS": 1}},
    }
    (root / "comd.yaml").write_text(yaml.safe_dump(config))
    environ = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHON": sys.executable,
               "FIXTURES": str(FIXTURES), "FAKE_API_KEY": "sk-should-not-leak"}
    environ.pop("FAKE_PERF_FAIL", None)
    return {"root": root, "work": work, "runs": root / "runs", "environ": environ}


def profile(env, *args, **extra_env):
    cmd = [BASH, str(PROFILE), "comd", *args, "--config", str(env["root"] / "comd.yaml")]
    return subprocess.run(cmd, cwd=env["root"], env={**env["environ"], **extra_env}, capture_output=True, text=True)


def annotate(env, *args):
    return subprocess.run([BASH, str(ANNOTATE), *args], cwd=env["root"], env=env["environ"],
                          capture_output=True, text=True)


def metadata(env, run_id):
    return json.loads((env["runs"] / run_id / "metadata.json").read_text())


# --- profile_target.sh: serial -------------------------------------------------

def test_one_call_produces_the_full_artifact_set(env):
    r = profile(env, "base", "--label", "first baseline")
    assert r.returncode == 0, r.stderr
    run = env["runs"] / "base"
    for name in ("metadata.json", "stdout.log", "stderr.log", "perf-stat.txt", "perf.data", "perf-report.txt"):
        assert (run / name).is_file() and (run / name).stat().st_size > 0, name

    stdout = (run / "stdout.log").read_text()
    assert stdout.count("app stdout rank=none threads=3 args=-N 5") == 4  # 3 stat repeats + 1 record
    assert "===== perf stat: perf stat -r 3 -d -o" in stdout and "===== perf record:" in stdout
    assert "app stderr line" in (run / "stderr.log").read_text()

    meta = metadata(env, "base")
    binary = env["work"] / "src/bin/CoMD-serial"
    assert meta["status"] == "ok" and meta["error"] is None
    assert meta["run_id"] == "base" and meta["label"] == "first baseline" and meta["variant"] == "serial"
    assert meta["run_command"] == f"{binary} -N 5"
    assert meta["command_line"].startswith(f"{PROFILE} comd base --label first\\ baseline --config ")
    assert meta["perf"]["version"] == "perf version 6.9.0-fake"
    assert meta["perf"]["commands"]["stat"].startswith("perf stat -r 3 -d -o ")
    assert meta["perf"]["commands"]["report"] == "perf report --stdio --no-children --sort overhead,dso,sym -g none"
    assert meta["perf"]["stat"]["runs"] == 3
    assert meta["perf"]["report"]["top"][0]["symbol"] == "ljForce"
    wall = meta["wall_clock"]
    assert wall["repeats"] == 3 and wall["app_mean_s"] == 0.0931
    assert wall["total_s"] > 0 and wall["perf_stat_s"] > 0 and wall["perf_record_s"] > 0
    assert meta["threads"]["variables"]["OMP_NUM_THREADS"] == "3" and meta["threads"]["mpi_ranks"] is None
    assert meta["environment"]["FAKE_API_KEY"] == "<redacted>"
    assert "sk-should-not-leak" not in (run / "metadata.json").read_text()
    assert meta["build"]["variant"] == "serial"
    assert meta["host"]["logical_cpus"] >= 1
    assert {"perf.data", "perf-report.txt", "stdout.log"} <= {a["path"] for a in meta["artifacts"]}


def test_run_command_override(env):
    assert profile(env, "custom", "--cmd", "{binary} -x 99").returncode == 0
    meta = metadata(env, "custom")
    assert meta["run_command"] == f"{env['work'] / 'src/bin/CoMD-serial'} -x 99"
    assert "args=-x 99" in (env["runs"] / "custom" / "stdout.log").read_text()


def test_runs_directory_option(env, tmp_path):
    other = tmp_path / "elsewhere"
    assert profile(env, "exp1", "--runs", str(other)).returncode == 0
    assert (other / "exp1" / "metadata.json").is_file() and not (env["runs"] / "exp1").exists()


# --- profile_target.sh: MPI ----------------------------------------------------

def test_mpi_run_writes_perf_data_and_report_per_rank(env):
    r = profile(env, "mpi", "--ranks", "2")
    assert r.returncode == 0, r.stderr
    run = env["runs"] / "mpi"
    for rank in (0, 1):
        assert (run / f"rank-{rank}" / "perf.data").is_file()
        assert "ljForce" in (run / f"rank-{rank}" / "perf-report.txt").read_text()
    assert not (run / "perf.data").exists() and not (run / "perf-report.txt").exists()
    assert (run / "perf-stat.txt").is_file()
    stdout = (run / "stdout.log").read_text()
    assert "rank=0 threads=1" in stdout and "rank=1 threads=1" in stdout

    meta = metadata(env, "mpi")
    assert meta["variant"] == "mpi" and meta["mpi"] == {"ranks": 2, "launcher": "mpirun -np 2"}
    assert set(meta["perf"]["rank_reports"]) == {"0", "1"} and meta["perf"]["report"] is None
    assert meta["perf"]["commands"]["stat"].endswith(f"-- mpirun -np 2 {env['work'] / 'src/bin/CoMD-mpi'} -i 2 -j 1 -k 1")
    assert meta["threads"]["mpi_ranks"] == 2 and meta["threads"]["variables"]["OMP_NUM_THREADS"] == "1"
    assert meta["build"]["variant"] == "mpi"


def test_mpi_rank_count_must_match_default_command(env):
    r = profile(env, "mpi3", "--ranks", "3")
    assert r.returncode == 2 and "written for 2 ranks; pass --cmd" in r.stderr
    r = profile(env, "mpi3", "--ranks", "3", "--cmd", "{binary} -i 3 -j 1 -k 1")
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (env["runs"] / "mpi3").glob("rank-*")) == ["rank-0", "rank-1", "rank-2"]


def test_mpi_needs_launcher(env):
    config = yaml.safe_load((env["root"] / "comd.yaml").read_text())
    config["mpi"]["launcher"] = "no-such-mpirun -np {ranks}"
    (env["root"] / "comd.yaml").write_text(yaml.safe_dump(config))
    r = profile(env, "nompi", "--ranks", "2")
    assert r.returncode == 1 and "launcher 'no-such-mpirun' not found" in r.stderr
    assert not (env["runs"] / "nompi").exists()


# --- profile_target.sh: failures -----------------------------------------------

@pytest.mark.parametrize("step,missing", [("stat", "perf.data"), ("record", "perf-report.txt")])
def test_failed_perf_command_stops_with_useful_error(env, step, missing):
    r = profile(env, "broken", FAKE_PERF_FAIL=step)
    run = env["runs"] / "broken"
    assert r.returncode == 1
    assert f"perf {step} failed (exit 3)" in r.stderr
    assert "fake failure: Permission denied" in r.stderr  # the tail of stderr.log is shown
    assert f"stderr: {run / 'stderr.log'}" in r.stderr and f"metadata: {run / 'metadata.json'}" in r.stderr
    assert not (run / missing).exists()  # later steps did not run
    meta = metadata(env, "broken")
    assert meta["status"] == "failed" and f"perf {step} failed" in meta["error"]


def test_failed_perf_report_stops(env):
    r = profile(env, "noreport", FAKE_PERF_FAIL="report")
    assert r.returncode == 1 and "perf report failed" in r.stderr
    assert metadata(env, "noreport")["status"] == "failed"


def test_report_without_samples_fails(env, tmp_path):
    empty = tmp_path / "empty-report.txt"
    empty.write_text("# Samples: 0  of event 'cycles'\n")
    r = profile(env, "nosamples", FAKE_REPORT=str(empty))
    assert r.returncode == 1 and "found no samples" in r.stderr


def test_application_failure_stops_the_run(env):
    r = profile(env, "appfail", "--cmd", "{binary} --fail")
    assert r.returncode == 1 and "perf stat failed (exit 5)" in r.stderr
    assert metadata(env, "appfail")["status"] == "failed"


def test_precondition_errors_write_nothing(env):
    assert profile(env, "base").returncode == 0
    r = profile(env, "base")
    assert r.returncode == 2 and "already exists" in r.stderr

    r = profile(env, "bad/id")
    assert r.returncode == 2 and "run id must be" in r.stderr

    (env["work"] / "src/bin/CoMD-serial").unlink()
    r = profile(env, "unbuilt")
    assert r.returncode == 1 and "is not built; run: scripts/build_target.sh comd" in r.stderr
    assert not (env["runs"] / "unbuilt").exists()

    r = profile(env, "x", "--ranks", "zero")
    assert r.returncode == 2 and "--ranks must be a positive integer" in r.stderr
    r = subprocess.run([BASH, str(PROFILE), "comd"], capture_output=True, text=True)
    assert r.returncode == 2 and "usage:" in r.stderr


# --- annotate_symbol.sh ----------------------------------------------------------

def test_annotate_saves_symbol_output(env):
    assert profile(env, "base").returncode == 0
    r = annotate(env, "base", "ljForce")
    out = env["runs"] / "base" / "annotate" / "ljForce.txt"
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "runs/base/annotate/ljForce.txt"
    assert "Percent" in out.read_text() and "<ljForce>" in out.read_text()


def test_annotate_unknown_symbol(env):
    assert profile(env, "base").returncode == 0
    r = annotate(env, "base", "noSuchFunction")
    assert r.returncode == 1 and "no samples for symbol 'noSuchFunction'" in r.stderr and "perf-report.txt" in r.stderr
    assert not list((env["runs"] / "base").glob("annotate/*"))


def test_annotate_mpi_run_needs_rank(env):
    assert profile(env, "mpi", "--ranks", "2").returncode == 0
    r = annotate(env, "mpi", "ljForce")
    assert r.returncode == 2 and "is an MPI run; pass --rank (ranks: 0 1 )" in r.stderr
    r = annotate(env, "mpi", "ljForce", "--rank", "1")
    assert r.returncode == 0 and (env["runs"] / "mpi" / "rank-1" / "annotate" / "ljForce.txt").is_file()
    r = annotate(env, "mpi", "ljForce", "--rank", "7")
    assert r.returncode == 2 and "has no rank 7" in r.stderr


def test_annotate_errors(env, tmp_path):
    r = annotate(env, "missing", "ljForce")
    assert r.returncode == 2 and "not found" in r.stderr
    assert profile(env, "base").returncode == 0
    r = subprocess.run([BASH, str(ANNOTATE), "base", "ljForce"], cwd=env["root"],
                       env={**env["environ"], "FAKE_PERF_FAIL": "annotate"}, capture_output=True, text=True)
    assert r.returncode == 1 and "perf annotate failed (exit 3)" in r.stderr
    custom = tmp_path / "out" / "lj.txt"
    assert annotate(env, "base", "ljForce", "--output", str(custom)).returncode == 0 and custom.is_file()
