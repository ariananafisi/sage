import asyncio
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from sage import agent


@pytest.fixture
def app_dir(tmp_path):
    d = tmp_path / "LULESH"
    d.mkdir()
    (d / "lulesh.cc").write_text("int main() { return 0; }\n")
    (d / "lulesh.h").write_text("#pragma once\n")
    (d / "CMakeLists.txt").write_text("project(LULESH)\n")
    return d


@pytest.fixture
def fake_perf(monkeypatch):
    real_which = agent.shutil.which
    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/perf" if name == "perf" else real_which(name))


def fake_query(calls, side_effect=None):
    async def query(*, prompt, options):
        calls.append((prompt, options))
        if side_effect:
            side_effect(Path(options.cwd))
        yield AssistantMessage(content=[TextBlock(text="checking perf"),
                                        ToolUseBlock(id="t1", name="Bash", input={"command": "perf --version"})],
                               model="test")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=2, session_id="s", result="## Hotspots\nCalcFBHourglassForceForElems 30%")
    return query


def test_options_single_agent_shell_and_read_only(tmp_path):
    opts = agent.build_options("comd", tmp_path, {"/x/a.c"}, model=None, max_turns=10)
    assert opts.tools == ["Bash", "Read", "Glob", "Grep"]
    assert opts.allowed_tools == opts.tools
    assert not opts.agents and not opts.mcp_servers
    assert opts.permission_mode == "dontAsk"
    assert opts.setting_sources == []
    assert opts.cwd == str(tmp_path)
    assert opts.max_turns == 10
    assert "perf is the only profiler" in opts.system_prompt
    assert "CoMD" in opts.system_prompt
    [matcher] = opts.hooks["PreToolUse"]
    assert matcher.matcher is None and len(matcher.hooks) == 1


def test_run_writes_report_and_leaves_source_untouched(app_dir, tmp_path, fake_perf, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(agent, "query", fake_query(calls))
    workdir = tmp_path / "run"
    assert asyncio.run(agent.run("lulesh", app_dir, workdir, goal="memory")) == 0

    [(prompt, options)] = calls
    assert "LULESH" in prompt and "memory" in prompt
    assert (workdir / "src" / "lulesh.cc").read_text() == (app_dir / "lulesh.cc").read_text()
    assert "Hotspots" in (workdir / "report.md").read_text()
    out = capsys.readouterr()
    assert "Hotspots" in out.out
    assert "[Bash] perf --version" in out.err


def test_run_allows_new_build_files_in_copy(app_dir, tmp_path, fake_perf, monkeypatch):
    def build(cwd):
        (cwd / "src" / "lulesh.o").write_text("obj")
        (cwd / "src" / "Makefile").write_text("generated")
    monkeypatch.setattr(agent, "query", fake_query([], build))
    assert asyncio.run(agent.run("lulesh", app_dir, tmp_path / "run")) == 0


def test_run_fails_if_source_modified(app_dir, tmp_path, fake_perf, monkeypatch, capsys):
    def tamper(cwd):
        (cwd / "src" / "lulesh.cc").write_text("// optimized\n")
    monkeypatch.setattr(agent, "query", fake_query([], tamper))
    assert asyncio.run(agent.run("lulesh", app_dir, tmp_path / "run")) == 2
    assert "source was modified" in capsys.readouterr().err


def test_run_requires_perf(app_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent, "query", lambda **kw: pytest.fail("agent should not start"))
    assert asyncio.run(agent.run("lulesh", app_dir, tmp_path / "run")) == 1


def test_run_rejects_workdir_inside_app(app_dir, fake_perf):
    assert asyncio.run(agent.run("lulesh", app_dir, app_dir / "run")) == 1


def test_cli_parses_arguments(app_dir, tmp_path, monkeypatch):
    seen = {}

    async def fake_run(app, app_dir_, workdir, model, max_turns, goal):
        seen.update(app=app, workdir=workdir, model=model, max_turns=max_turns, goal=goal)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    assert agent.main(["comd", str(app_dir), "--workdir", str(tmp_path / "w"), "--max-turns", "5"]) == 0
    assert seen == {"app": "comd", "workdir": tmp_path / "w", "model": None, "max_turns": 5, "goal": None}

    with pytest.raises(SystemExit):
        agent.main(["hpcg", str(app_dir)])
