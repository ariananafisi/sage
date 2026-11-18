import subprocess
from pathlib import Path

import pytest

from sage.guard import HIDDEN_REASON, Policy, check_bash, check_command, check_execution, source_changes, tracked

CWD = "/work"
PROTECTED = {
    "/work/src/lulesh.cc",
    "/work/src/lulesh.h",
    "/work/src/CMakeLists.txt",
    "/work/src/src-mpi/ljForce.c",
    "/work/src/src-mpi/Makefile.vanilla",
}


# --- source edits and other profilers (check_command) ------------------------------

@pytest.mark.parametrize("cmd", [
    "perf --version",
    "cat /proc/sys/kernel/perf_event_paranoid",
    "perf report --stdio --no-children > hot.txt",
    "perf annotate --stdio -s CalcHourglassControlForElems | head -100",
    "cmake -S src -B build -DWITH_MPI=OFF -DCMAKE_CXX_FLAGS='-O3 -g -fno-omit-frame-pointer'",
    "cmake --build build -j8 2>&1 | tail -20",
    "cd src/src-mpi && cp Makefile.vanilla Makefile && make DO_MPI=OFF CC=gcc",
    "grep -n 'for (' src/lulesh.cc | tee loops.txt",
    "cat src/lulesh.cc > /dev/null",
    "wc -l src/*.cc",
    "git -C src log --oneline -3",
    "git -C src status --porcelain",
    "git --no-pager -C src diff",
    "cat > findings.json <<'EOF'\n{}\nEOF",
])
def test_allows_reading_and_building(cmd):
    assert check_command(cmd, PROTECTED, CWD) is None


@pytest.mark.parametrize("cmd", [
    "valgrind --tool=callgrind ./a.out",
    "gprof ./lulesh2.0 gmon.out",
    "cmake -S src -B build -DCMAKE_CXX_FLAGS=-pg",
    "strace -c ./a.out",
    "likwid-perfctr -g MEM ./a.out",
    "bash -c 'nsys profile ./a.out'",
])
def test_denies_other_profilers(cmd):
    assert "perf is the only" in check_command(cmd, PROTECTED, CWD)


@pytest.mark.parametrize("cmd", [
    "sed -i 's/a/b/' src/lulesh.cc",
    "perl -pi -e 's/a/b/' src/lulesh.cc",
    "echo '// x' >> src/lulesh.h",
    "cat > src/lulesh.cc <<'EOF'\nint main(){}\nEOF",
    "cd src && echo x > lulesh.cc",
    "cp /tmp/fast.cc src/lulesh.cc",
    "mv src/lulesh.cc src/lulesh.cc.bak",
    "rm -f src/*.cc",
    "sudo tee src/CMakeLists.txt < new.txt",
    "truncate -s 0 src/src-mpi/ljForce.c",
    "dd if=/dev/zero of=src/lulesh.h count=1",
    "cp Makefile src/src-mpi/Makefile.vanilla",
    "find src -name '*.c' -exec rm {} \\;",
    "git -C src checkout .",
    "git -C src update-index --assume-unchanged lulesh.cc",
    "git -C src stash",
    "git add -A",
    "vim src/lulesh.cc",
    "patch -p1 < opt.diff",
    "bash -c 'echo x > src/lulesh.cc'",
    "eval 'rm src/lulesh.h'",
])
def test_denies_source_modification(cmd):
    reason = check_command(cmd, PROTECTED, CWD)
    assert reason is not None and "perf is the only" not in reason


# --- running the application (check_execution) -----------------------------------------

@pytest.fixture
def policy(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    return Policy(src=tmp_path / ".work" / "lulesh" / "src", run_dir=run_dir,
                  binaries=frozenset({"lulesh2.0", "CoMD-serial"}), max_experiments=1)


@pytest.mark.parametrize("cmd", [
    "perf report -i baseline/perf.data --stdio --no-children -g caller",
    "perf annotate -i baseline/perf.data --stdio -s ljForce",
    "perf script -i baseline/perf.data | head",
    "perf list sw",
    "perf --version",
    "/w/.work/lulesh/build/lulesh2.0 -h",
    "/w/.work/comd/src/bin/CoMD-serial --help | head -40",
    "/repo/scripts/build_target.sh lulesh --config c.yaml --work /w/.work",
    "{run}/../../x/profile_target.sh lulesh baseline --runs {run} --work /w --config c.yaml",
    "bash /repo/scripts/profile_target.sh lulesh baseline --runs {run} --config c.yaml",
    "/repo/scripts/profile_target.sh lulesh size-60 --runs {run}/experiments --cmd '{{binary}} -s 60'",
    "cd {run} && /repo/scripts/profile_target.sh lulesh threads-1 --runs experiments --cmd '{{binary}}'",
    "/repo/scripts/annotate_symbol.sh baseline ljForce --runs {run}",
    "/repo/scripts/profile_target.sh lulesh threads-1 --runs $PWD/experiments --env OMP_NUM_THREADS=1",
    "/repo/scripts/profile_target.sh lulesh threads-1 --env OMP_NUM_THREADS=1 --runs ${{PWD}}/experiments",
])
def test_allowed_executions(policy, cmd):
    command = cmd.format(run=policy.run_dir)
    cwd = str(policy.run_dir)
    assert check_execution(command, policy, cwd) is None


@pytest.mark.parametrize("cmd,message", [
    ("perf stat -e task-clock ./lulesh2.0 -s 10", "'perf stat' runs the application outside"),
    ("perf record -g -o x.data -- /w/build/lulesh2.0", "'perf record' runs the application"),
    ("cd /w && perf top", "'perf top'"),
    ("sudo perf trace ls", "'perf trace'"),
    ("/w/build/lulesh2.0 -s 10 -i 5", "Running lulesh2.0 directly is not allowed"),
    ("OMP_NUM_THREADS=1 /w/build/lulesh2.0", "Running lulesh2.0 directly"),
    ("timeout 10 /w/bin/CoMD-serial -N 2", "Running CoMD-serial directly"),
    ("bash -c '/w/build/lulesh2.0 -s 5'", "Running lulesh2.0 directly"),
    ("mpirun -np 4 /w/bin/CoMD-mpi", "'mpirun' is not allowed directly"),
    ("/repo/scripts/profile_target.sh lulesh mine --runs /tmp/elsewhere", "Artifact sets must go to"),
    ("/repo/scripts/profile_target.sh lulesh other --runs {run}", "Artifact sets must go to"),
    ("/repo/scripts/profile_target.sh lulesh baseline", "Artifact sets must go to"),
])
def test_denied_executions(policy, cmd, message):
    reason = check_execution(cmd.format(run=policy.run_dir), policy, str(policy.run_dir))
    assert reason is not None and message in reason, reason


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_experiment_limit(policy, limit):
    policy = Policy(src=policy.src, run_dir=policy.run_dir, binaries=policy.binaries, max_experiments=limit)
    cmd = f"/repo/scripts/profile_target.sh lulesh exp --runs {policy.experiments} --cmd '{{binary}}'"
    for used in range(limit):
        assert check_execution(cmd, policy, str(policy.run_dir)) is None
        (policy.experiments / f"exp-{used}").mkdir(parents=True)
    reason = check_execution(cmd, policy, str(policy.run_dir))
    assert f"Experiment limit reached: {limit} of {limit} experiment run(s) used" in reason
    # the baseline is not an experiment
    baseline = f"/repo/scripts/profile_target.sh lulesh baseline --runs {policy.run_dir}"
    assert check_execution(baseline, policy, str(policy.run_dir)) is None


def test_experiment_count_ignores_files(policy):
    policy.experiments.mkdir()
    (policy.experiments / "notes.txt").write_text("x")
    assert policy.experiments_used() == 0


# --- check_bash: what the agent's shell tool uses -------------------------------------------

def test_check_bash_combines_both_checks(policy):
    cwd = str(policy.run_dir)
    assert check_bash("perf report -i baseline/perf.data --stdio", policy, cwd) is None
    assert "'perf stat' runs the application" in check_bash("perf stat ls", policy, cwd)
    assert "perf is the only" in check_bash("valgrind ls", policy, cwd)


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    git("init", "-q", cwd=src)
    (src / "a.c").write_text("int a;\n")
    (src / "sub" / "b.h").write_text("int b;\n")
    git("add", ".", cwd=src)
    git("commit", "-qm", "init", cwd=src)
    return src, git("rev-parse", "HEAD", cwd=src)


def test_checkout_created_mid_run_is_protected(policy, checkout):
    src, _ = checkout
    p = Policy(src=src, run_dir=policy.run_dir, binaries=policy.binaries, max_experiments=1)
    assert tracked(src) == {str(src.resolve() / "a.c"), str(src.resolve() / "sub" / "b.h")}
    assert "Source files are read-only" in check_bash(f"echo x > {src}/a.c", p, str(policy.run_dir))
    assert tracked(src.parent / "missing") == set()


# --- source_changes ---------------------------------------------------------------------------

def test_source_changes_clean_and_generated_files(checkout):
    src, commit = checkout
    (src / "a.o").write_text("object")
    (src / "Makefile").write_text("generated")
    assert source_changes(src, commit) == []
    assert source_changes(src.parent / "never-cloned", commit) == []


def test_source_changes_detects_edits_deletions_and_head(checkout):
    src, commit = checkout
    (src / "a.c").write_text("int a = 1;\n")
    (src / "sub" / "b.h").unlink()
    problems = source_changes(src, commit)
    assert f"{src / 'a.c'} was modified" in problems and f"{src / 'sub/b.h'} was deleted" in problems

    git("checkout", "-q", "--", ".", cwd=src)
    (src / "c.c").write_text("x")
    git("add", "c.c", cwd=src)
    git("commit", "-qm", "two", cwd=src)
    assert any("checkout HEAD is" in p for p in source_changes(src, commit))


def test_source_changes_sees_through_index_tricks(checkout):
    src, commit = checkout
    git("update-index", "--assume-unchanged", "a.c", cwd=src)
    (src / "a.c").write_text("int a = 2;\n")
    assert git("status", "--porcelain", cwd=src) == ""  # git itself no longer reports it
    assert source_changes(src, commit) == [f"{src / 'a.c'} was modified"]


def test_source_changes_without_git_metadata(checkout):
    src, commit = checkout
    subprocess.run(["rm", "-rf", str(src / ".git")], check=True)
    assert source_changes(src, commit) == [f"{src} is no longer a git checkout; its source cannot be verified"]


def test_hidden_paths(policy, tmp_path):
    suites = (tmp_path / "repo" / "suites").resolve()
    results = (tmp_path / "out" / "results.json").resolve()
    p = Policy(src=policy.src, run_dir=policy.run_dir, binaries=policy.binaries, max_experiments=1,
               hidden=(suites, results))
    cwd = str(policy.run_dir)
    for cmd in [f"cat {suites}/comd.yaml", "cd /x/repo && cat suites/comd.yaml", "ls ../../suites",
                "grep -r ljForce suites", f"cat {results}", "cat ../../results.json", "find / -name results.json"]:
        assert check_bash(cmd, p, cwd) == HIDDEN_REASON, cmd
    for cmd in ["cat baseline/perf-report.txt", "echo my-suites-notes", "ls results.jsonl"]:
        assert check_bash(cmd, p, cwd) is None, cmd
    assert p.is_hidden(suites / "patches" / "x.patch") and p.is_hidden(results)
    assert not p.is_hidden(tmp_path / "repo" / "sage")
