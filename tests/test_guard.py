import asyncio

import pytest

from sage.guard import changed, check_command, make_guard, snapshot

CWD = "/work"
PROTECTED = {
    "/work/src/lulesh.cc",
    "/work/src/lulesh.h",
    "/work/src/CMakeLists.txt",
    "/work/src/src-mpi/ljForce.c",
    "/work/src/src-mpi/Makefile.vanilla",
}


@pytest.mark.parametrize("cmd", [
    "perf --version",
    "cat /proc/sys/kernel/perf_event_paranoid",
    "perf stat -e cycles,instructions,cache-misses ./build/lulesh2.0 -s 30 -i 50",
    "perf record -g -o perf.data ./build/lulesh2.0 -s 30",
    "perf report --stdio --no-children > hot.txt",
    "perf annotate --stdio -s CalcHourglassControlForElems | head -100",
    "cmake -S src -B build -DWITH_MPI=OFF -DCMAKE_CXX_FLAGS='-O3 -g -fno-omit-frame-pointer'",
    "cmake --build build -j8 2>&1 | tail -20",
    "cd src/src-mpi && cp Makefile.vanilla Makefile && make DO_MPI=OFF CC=gcc",
    "grep -n 'for (' src/lulesh.cc | tee loops.txt",
    "cat src/lulesh.cc > /dev/null",
    "wc -l src/*.cc",
    "OMP_NUM_THREADS=4 /usr/bin/time -v ./build/lulesh2.0 -s 20",
])
def test_allows_profiling_and_building(cmd):
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
    "vim src/lulesh.cc",
    "patch -p1 < opt.diff",
    "bash -c 'echo x > src/lulesh.cc'",
    "eval 'rm src/lulesh.h'",
])
def test_denies_source_modification(cmd):
    reason = check_command(cmd, PROTECTED, CWD)
    assert reason is not None and "perf is the only" not in reason


def run_hook(tool_name, tool_input):
    hook = make_guard(PROTECTED)
    return asyncio.run(hook({"tool_name": tool_name, "tool_input": tool_input, "cwd": CWD}, "id", None))


def test_hook_allows_with_empty_output():
    assert run_hook("Bash", {"command": "perf stat ls"}) == {}
    assert run_hook("Read", {"file_path": "/work/src/lulesh.cc"}) == {}


def test_hook_denies_in_sdk_format():
    out = run_hook("Bash", {"command": "sed -i s/x/y/ src/lulesh.cc"})["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "deny"
    assert out["permissionDecisionReason"]
    assert run_hook("Write", {"file_path": "/work/x"})["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_snapshot_detects_edits_and_deletions_only(tmp_path):
    (tmp_path / "a.cc").write_text("int a;")
    (tmp_path / "b.h").write_text("int b;")
    (tmp_path / "Makefile.vanilla").write_text("CC=cc")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.c").write_text("ignored")
    snap = snapshot(tmp_path)
    assert {p.split("/")[-1] for p in snap} == {"a.cc", "b.h", "Makefile.vanilla"}
    assert changed(snap) == []

    (tmp_path / "new.o").write_text("build output")
    (tmp_path / "Makefile").write_text("copied template")
    (tmp_path / "notes.txt").write_text("changed but not source")
    assert changed(snap) == []

    (tmp_path / "a.cc").write_text("int a = 1;")
    (tmp_path / "b.h").unlink()
    assert [p.split("/")[-1] for p in changed(snap)] == ["a.cc", "b.h"]
