import asyncio
import subprocess
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from sage import agent
from sage.config import TargetConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
BUILD = "mkdir -p build && printf '#!/bin/sh\\necho run\\n' > build/app && chmod +x build/app && echo $GREETING > build/env.txt"


def sh(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    """A local git repo standing in for the upstream application."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    sh("init", "-q", "-b", "master", cwd=repo)
    (repo / "main.c").write_text("int main(void) { return 0; }\n")
    sh("add", ".", cwd=repo)
    sh("commit", "-q", "-m", "v1", cwd=repo)
    return repo


@pytest.fixture
def cfg(upstream):
    return TargetConfig(target="lulesh", repo_url=str(upstream), commit=sh("rev-parse", "HEAD", cwd=upstream),
                        build=[BUILD], binary="build/app", run="{binary} -s 5", env={"GREETING": "hi"})


@pytest.fixture
def fake_perf(monkeypatch):
    real_which = agent.shutil.which
    monkeypatch.setattr(agent.shutil, "which", lambda name: "/usr/bin/perf" if name == "perf" else real_which(name))


def fake_query(calls, side_effect=None):
    async def query(*, prompt, options):
        calls.append((prompt, options))
        if side_effect:
            side_effect()
        yield AssistantMessage(content=[TextBlock(text="checking perf"),
                                        ToolUseBlock(id="t1", name="Bash", input={"command": "perf --version"})],
                               model="test")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=2, session_id="s", result="## Hotspots\nCalcFBHourglassForceForElems 30%")
    return query


# --- checkout and build -------------------------------------------------------

def test_checkout_clones_pins_and_reuses(cfg, upstream, tmp_path):
    src = tmp_path / "work" / "src"
    agent.checkout(cfg, src)
    assert sh("rev-parse", "HEAD", cwd=src) == cfg.commit

    (upstream / "main.c").write_text("int main(void) { return 1; }\n")
    sh("commit", "-qam", "v2", cwd=upstream)
    sh("checkout", "-q", "master", cwd=src)
    sh("pull", "-q", cwd=src)
    agent.checkout(cfg, src)  # moves back to the pinned commit
    assert sh("rev-parse", "HEAD", cwd=src) == cfg.commit


def test_checkout_rejects_local_changes(cfg, tmp_path):
    src = tmp_path / "src"
    agent.checkout(cfg, src)
    (src / "main.c").write_text("// edited\n")
    with pytest.raises(agent.SetupError, match="local changes"):
        agent.checkout(cfg, src)


def test_checkout_rejects_wrong_origin_and_non_git_dir(cfg, tmp_path):
    src = tmp_path / "src"
    agent.checkout(cfg, src)
    with pytest.raises(agent.SetupError, match="was cloned from"):
        agent.checkout(TargetConfig(**{**cfg.__dict__, "repo_url": "https://example.com/other.git"}), src)
    other = tmp_path / "other"
    other.mkdir()
    (other / "x").write_text("")
    with pytest.raises(agent.SetupError, match="not a git checkout"):
        agent.checkout(cfg, other)


def test_checkout_unknown_commit(cfg, tmp_path):
    with pytest.raises(agent.SetupError):
        agent.checkout(TargetConfig(**{**cfg.__dict__, "commit": "0" * 40}), tmp_path / "src")


def test_build_runs_commands_with_env(cfg, tmp_path):
    binary = agent.build(cfg, tmp_path)
    assert binary == tmp_path / "build" / "app"
    assert (tmp_path / "build" / "env.txt").read_text().strip() == "hi"
    assert "$ mkdir -p build" in (tmp_path / "build.log").read_text()


def test_build_failures(cfg, tmp_path):
    with pytest.raises(agent.SetupError, match="build command failed"):
        agent.build(TargetConfig(**{**cfg.__dict__, "build": ["echo oops; exit 3"]}), tmp_path)
    assert "oops" in (tmp_path / "build.log").read_text()
    with pytest.raises(agent.SetupError, match="not an executable"):
        agent.build(TargetConfig(**{**cfg.__dict__, "build": ["true"]}), tmp_path)


# --- agent options and run ----------------------------------------------------

def test_options_single_agent_shell_and_read_only(cfg, tmp_path):
    opts = agent.build_options(cfg, tmp_path / "work", tmp_path / "run", {"/x/a.c"}, model=None, max_turns=10)
    assert opts.tools == ["Bash", "Read", "Glob", "Grep"]
    assert opts.allowed_tools == opts.tools
    assert not opts.agents and not opts.mcp_servers
    assert opts.permission_mode == "dontAsk"
    assert opts.setting_sources == []
    assert opts.cwd == str(tmp_path / "run")
    assert opts.add_dirs == [str(tmp_path / "work")]
    assert opts.env == {"GREETING": "hi"}
    assert opts.max_turns == 10
    assert "perf is the only profiler" in opts.system_prompt and "LULESH" in opts.system_prompt
    [matcher] = opts.hooks["PreToolUse"]
    assert matcher.matcher is None and len(matcher.hooks) == 1


def test_run_end_to_end_layout(cfg, tmp_path, fake_perf, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(agent, "query", fake_query(calls))
    root = tmp_path / "project"
    assert asyncio.run(agent.run(cfg, root, goal="memory")) == 0

    work = root / ".work" / "lulesh"
    assert sh("rev-parse", "HEAD", cwd=work / "src") == cfg.commit
    assert (work / "build" / "app").is_file()
    [run_dir] = (root / "runs").iterdir()
    assert run_dir.name.startswith("lulesh-")
    assert "Hotspots" in (run_dir / "report.md").read_text()

    [(prompt, options)] = calls
    assert f"Default run command: {work}/build/app -s 5" in prompt
    assert str(work / "src") in prompt and cfg.commit in prompt and "memory" in prompt
    assert options.cwd == str(run_dir)
    out = capsys.readouterr()
    assert "Hotspots" in out.out and "[Bash] perf --version" in out.err


def test_run_allows_generated_files_in_checkout(cfg, tmp_path, fake_perf, monkeypatch):
    root = tmp_path / "project"
    gen = lambda: (root / ".work/lulesh/src/generated.h").write_text("// generated")
    monkeypatch.setattr(agent, "query", fake_query([], gen))
    assert asyncio.run(agent.run(cfg, root)) == 0


def test_run_fails_if_tracked_source_modified(cfg, tmp_path, fake_perf, monkeypatch, capsys):
    root = tmp_path / "project"
    tamper = lambda: (root / ".work/lulesh/src/main.c").write_text("// optimized\n")
    monkeypatch.setattr(agent, "query", fake_query([], tamper))
    assert asyncio.run(agent.run(cfg, root)) == 3
    assert "source was modified" in capsys.readouterr().err


def test_run_reports_setup_errors(cfg, tmp_path, fake_perf, monkeypatch, capsys):
    monkeypatch.setattr(agent, "query", lambda **kw: pytest.fail("agent should not start"))
    bad = TargetConfig(**{**cfg.__dict__, "build": ["exit 1"]})
    assert asyncio.run(agent.run(bad, tmp_path)) == 1
    assert "build command failed" in capsys.readouterr().err


def test_run_requires_perf(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent, "query", lambda **kw: pytest.fail("agent should not start"))
    assert asyncio.run(agent.run(cfg, tmp_path)) == 1
    assert not (tmp_path / ".work").exists()


# --- CLI ----------------------------------------------------------------------

@pytest.fixture
def captured_run(monkeypatch):
    seen = {}

    async def fake_run(cfg, root, model, max_turns, goal):
        seen.update(cfg=cfg, root=root, model=model, max_turns=max_turns, goal=goal)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    return seen


@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_cli_analyze_loads_config(target, captured_run):
    argv = ["analyze", "--target", target, "--config", str(CONFIGS / f"{target}.yaml"), "--max-turns", "5"]
    assert agent.main(argv) == 0
    assert captured_run["cfg"].target == target
    assert captured_run["max_turns"] == 5 and captured_run["root"] == Path.cwd()


def cli_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        agent.main(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_cli_invalid_target(captured_run, capsys):
    err = cli_error(["analyze", "--target", "hpcg", "--config", str(CONFIGS / "lulesh.yaml")], capsys)
    assert "invalid target 'hpcg' (choose from comd, lulesh)" in err
    assert not captured_run


def test_cli_config_errors(captured_run, tmp_path, capsys):
    err = cli_error(["analyze", "--target", "comd", "--config", str(CONFIGS / "lulesh.yaml")], capsys)
    assert "config is for target 'lulesh' but --target is 'comd'" in err
    err = cli_error(["analyze", "--target", "comd", "--config", str(tmp_path / "missing.yaml")], capsys)
    assert "config file not found" in err
    (tmp_path / "bad.yaml").write_text("target: comd\ncommit: abc\n")
    err = cli_error(["analyze", "--target", "comd", "--config", str(tmp_path / "bad.yaml")], capsys)
    assert "commit must be a full 40-character" in err and "missing required key 'build'" in err
    assert not captured_run


def test_cli_requires_arguments(capsys):
    assert "required" in cli_error([], capsys)
    assert "--config" in cli_error(["analyze", "--target", "lulesh"], capsys)
