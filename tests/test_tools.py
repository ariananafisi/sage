"""The agent's tools (bash, read_file, search, write_findings), run for real in a temporary run directory."""

import json
import os
import time
from pathlib import Path

import pytest

from sage import tools as tools_mod
from sage.guard import Policy
from sage.tools import SCHEMAS, DeadlineExceeded, Tools, truncate


@pytest.fixture
def tools(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    policy = Policy(src=tmp_path / ".work" / "lulesh" / "src", run_dir=run_dir,
                    binaries=frozenset({"lulesh2.0"}), max_experiments=1)
    return Tools(policy=policy, env={"OMP_NUM_THREADS": "3"}, deadline=time.monotonic() + 600)


def test_schemas_are_the_four_tools():
    assert [s["function"]["name"] for s in SCHEMAS] == ["bash", "read_file", "search", "write_findings"]
    for s in SCHEMAS:
        params = s["function"]["parameters"]
        assert params["type"] == "object" and set(params["required"]) <= set(params["properties"])
        assert params["additionalProperties"] is False


# --- bash ----------------------------------------------------------------------------

def test_bash_runs_in_run_dir_with_env(tools):
    out = tools.call("bash", json.dumps({"command": "pwd; echo $OMP_NUM_THREADS; echo err >&2; exit 3"}))
    assert out.startswith("exit code 3\n")
    assert str(tools.run_dir) in out and "\n3\n" in out and "err" in out


def test_bash_cd_does_not_persist(tools):
    tools.bash("mkdir sub && cd sub")
    assert str(tools.run_dir) + "\n" in tools.bash("pwd")


def test_bash_refuses_guarded_commands(tools):
    assert tools.bash("perf stat -e task-clock ls").startswith("DENIED: 'perf stat' runs the application")
    assert tools.bash("valgrind ls").startswith("DENIED: 'valgrind' is not allowed")
    assert tools.bash("./lulesh2.0 -s 5").startswith("DENIED: Running lulesh2.0 directly")
    assert not (tools.run_dir / "x").exists()


def test_bash_output_is_truncated(tools):
    out = tools.bash("python3 -c \"print('a' * 50000)\"")
    assert len(out) < tools_mod.MAX_OUTPUT + 200 and "characters omitted" in out


def test_bash_command_timeout_kills_process_group(tools, monkeypatch):
    monkeypatch.setattr(tools_mod, "MAX_COMMAND_SECONDS", 1)
    start = time.monotonic()
    out = tools.bash("echo started; (sleep 30; touch late) & sleep 30")
    assert time.monotonic() - start < 10
    assert out.startswith("TIMED OUT after 1 s") and "started" in out
    time.sleep(0.2)
    assert not (tools.run_dir / "late").exists()


def test_bash_past_the_deadline_raises(tools):
    tools.deadline = time.monotonic() + 0.5
    with pytest.raises(DeadlineExceeded):
        tools.bash("sleep 20")
    tools.deadline = time.monotonic() - 1
    with pytest.raises(DeadlineExceeded):
        tools.bash("true")


# --- read_file -------------------------------------------------------------------------

def test_read_file_numbers_lines_and_pages(tools):
    (tools.run_dir / "a.txt").write_text("".join(f"line {i}\n" for i in range(1, 11)))
    out = tools.read_file("a.txt")
    assert "(10 lines)" in out and "     1\tline 1" in out and "    10\tline 10" in out
    page = tools.read_file("a.txt", offset=4, limit=2)
    assert "     4\tline 4" in page and "     5\tline 5" in page and "line 6" not in page and "[5 more lines]" in page


def test_read_file_errors(tools):
    assert tools.read_file("missing.txt").startswith("ERROR:")
    (tools.run_dir / "bin").write_bytes(b"\x7fELF\0\0\0")
    assert "binary file" in tools.read_file("bin")
    assert "is not a file" in tools.read_file(".")


def test_read_file_absolute_path(tools, tmp_path):
    (tmp_path / "src.c").write_text("int x;\n")
    assert "     1\tint x;" in tools.read_file(str(tmp_path / "src.c"))


# --- search ----------------------------------------------------------------------------

def test_search_regex_glob_and_skips(tools):
    src = tools.run_dir / "src"
    (src / ".git").mkdir(parents=True)
    (src / "a.c").write_text("void ljForce(void) {\n  for (int i = 0; i < n; i++) {}\n}\n")
    (src / "b.h").write_text("void ljForce(void);\n")
    (src / ".git" / "x.c").write_text("ljForce")
    (src / "obj.o").write_bytes(b"\0ljForce")
    out = tools.search(r"ljForce\(", "src")
    assert f"{src / 'a.c'}:1: void ljForce(void) {{" in out and f"{src / 'b.h'}:1:" in out
    assert ".git" not in out and "obj.o" not in out
    assert "b.h" not in tools.search("ljForce", "src", glob="*.c")
    assert "a.c:2:" in tools.search("FOR \\(", "src", ignore_case=True)
    assert tools.search("nothing-here", "src") == "no matches"
    assert tools.search("(", "src").startswith("ERROR: invalid regular expression")
    assert tools.search("x", "missing").startswith("ERROR:")
    assert f"{src / 'a.c'}:1:" in tools.search("ljForce", "src/a.c")


def test_search_stops_at_limit(tools):
    (tools.run_dir / "many.txt").write_text("hit\n" * 500)
    out = tools.search("hit", ".")
    assert out.count(":") >= tools_mod.MAX_SEARCH_RESULTS and "stopped at 200 matches" in out


# --- write_findings and dispatch -------------------------------------------------------------

def test_write_findings_writes_only_findings_json(tools):
    out = tools.call("write_findings", json.dumps({"content": '{"a": 1}'}))
    assert out.startswith("saved") and "validate" in out
    assert (tools.run_dir / "findings.json").read_text() == '{"a": 1}'
    assert sorted(p.name for p in tools.run_dir.iterdir()) == ["findings.json"]


def test_write_findings_refuses_symlink(tools, tmp_path):
    (tmp_path / "elsewhere.json").write_text("{}")
    (tools.run_dir / "findings.json").symlink_to(tmp_path / "elsewhere.json")
    assert tools.write_findings("{}").startswith("ERROR")
    assert (tmp_path / "elsewhere.json").read_text() == "{}"


@pytest.mark.parametrize("name,arguments,message", [
    ("bash", "{not json", "not valid JSON"),
    ("bash", "[1]", "must be a JSON object"),
    ("delete_everything", "{}", "unknown tool 'delete_everything'"),
    ("bash", '{"cmd": "ls"}', "bad arguments for bash"),
    ("bash", '{"command": ""}', "non-empty string"),
    ("write_findings", '{"content": 5}', "must be a string"),
])
def test_call_errors_are_returned_as_text(tools, name, arguments, message):
    assert message in tools.call(name, arguments)


def test_truncate_keeps_head_and_tail():
    text = "H" * 100 + "M" * 1000 + "T" * 100
    out = truncate(text, limit=300)
    assert out.startswith("H" * 100) and out.endswith("T" * 100) and "omitted" in out
    assert truncate("short", limit=300) == "short"


def test_read_and_search_skip_hidden_paths(tools, tmp_path):
    secret = tmp_path / "suites"
    secret.mkdir()
    (secret / "answers.yaml").write_text("function: ljForce\n")
    (tmp_path / "visible.txt").write_text("ljForce\n")
    tools.policy = Policy(src=tools.policy.src, run_dir=tools.run_dir, binaries=tools.policy.binaries,
                          max_experiments=1, hidden=(secret.resolve(),))
    assert "outside the analysis" in tools.read_file(str(secret / "answers.yaml"))
    assert "outside the analysis" in tools.search("ljForce", str(secret))
    out = tools.search("ljForce", str(tmp_path))
    assert "visible.txt" in out and "answers.yaml" not in out
