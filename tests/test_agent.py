import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from sage import agent
from sage.config import TargetConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
COMMIT = "a" * 40
PREFLIGHT = {
    "ok": True, "host": {"cpu_model": "Fake CPU", "cpus": 4}, "perf": {"version": "perf version 6.9"},
    "perf_event_paranoid": 1, "events": {"cycles": False, "instructions": False, "software": True, "topdown": False},
    "warnings": ["only software events are available"], "errors": [],
}


@pytest.fixture
def cfg():
    return TargetConfig(target="lulesh", repo_url="https://example.invalid/up.git", commit=COMMIT,
                        compiler="g++", flags="-O3 -g -fno-omit-frame-pointer",
                        build=['make CXX="$SAGE_COMPILER" CXXFLAGS="$SAGE_FLAGS"'], binary="build/app",
                        run="{binary} -s 5", env={"GREETING": "hi"})


@pytest.fixture
def scripts(monkeypatch):
    """Stand-in for scripts/*.sh: writes what the real scripts write, returns the given exit codes."""
    calls, codes = [], {"preflight.sh": 0, "build_target.sh": 0}

    def fake(name, *args, cwd):
        calls.append((name, args))
        if name == "preflight.sh":
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(PREFLIGHT))
        else:
            work = Path(args[args.index("--work") + 1]) / args[0]
            src = work / "src"
            src.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q"], cwd=src, check=True)
            (src / "main.c").write_text("int main(void) { return 0; }\n")
            subprocess.run(["git", "add", "."], cwd=src, check=True)
            (work / "build-info.json").write_text(json.dumps({
                "repo_url": "https://example.invalid/up.git", "commit": COMMIT, "binary": str(work / "build/app"),
                "compiler_version": "g++ (Fake) 13.0", "flags": "-O3 -g -fno-omit-frame-pointer",
                "build_commands": ["make"]}))
        return codes[name]

    monkeypatch.setattr(agent, "run_script", fake)
    return calls, codes


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


def test_run_preflight_build_then_agent(cfg, tmp_path, scripts, monkeypatch, capsys):
    script_calls, _ = scripts
    calls = []
    monkeypatch.setattr(agent, "query", fake_query(calls))
    root = tmp_path / "project"
    root.mkdir()
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), root, goal="memory")) == 0

    work = root / ".work" / "lulesh"
    assert [c[0] for c in script_calls] == ["preflight.sh", "build_target.sh"]
    assert script_calls[0][1] == ("--output", str(root / ".work" / "preflight.json"))
    assert script_calls[1][1] == ("lulesh", "--config", str(Path("cfg.yaml").resolve()), "--work", str(root / ".work"))
    [run_dir] = (root / "runs").iterdir()
    assert run_dir.name.startswith("lulesh-")
    assert "Hotspots" in (run_dir / "report.md").read_text()

    [(prompt, options)] = calls
    assert f"Default run command: {work}/build/app -s 5" in prompt
    assert COMMIT in prompt and "g++ (Fake) 13.0" in prompt and "-O3 -g -fno-omit-frame-pointer" in prompt
    assert "cycles=no" in prompt and "warning: only software events are available" in prompt
    assert "memory" in prompt
    assert options.cwd == str(run_dir)
    out = capsys.readouterr()
    assert "Hotspots" in out.out and "[Bash] perf --version" in out.err


@pytest.mark.parametrize("failing,message", [("preflight.sh", "preflight failed"), ("build_target.sh", "build failed")])
def test_run_stops_when_a_script_fails(cfg, tmp_path, scripts, monkeypatch, capsys, failing, message):
    _, codes = scripts
    codes[failing] = 1
    monkeypatch.setattr(agent, "query", lambda **kw: pytest.fail("agent should not start"))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path)) == 1
    assert message in capsys.readouterr().err


def test_run_allows_generated_files_in_checkout(cfg, tmp_path, scripts, monkeypatch):
    gen = lambda: (tmp_path / ".work/lulesh/src/generated.h").write_text("// generated")
    monkeypatch.setattr(agent, "query", fake_query([], gen))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path)) == 0


def test_run_fails_if_tracked_source_modified(cfg, tmp_path, scripts, monkeypatch, capsys):
    tamper = lambda: (tmp_path / ".work/lulesh/src/main.c").write_text("// optimized\n")
    monkeypatch.setattr(agent, "query", fake_query([], tamper))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path)) == 3
    assert "source was modified" in capsys.readouterr().err


def test_real_scripts_exist_and_are_executable():
    for name in ("preflight.sh", "build_target.sh"):
        path = agent.SCRIPTS / name
        assert path.is_file() and path.stat().st_mode & 0o111


# --- CLI ----------------------------------------------------------------------

@pytest.fixture
def captured_run(monkeypatch):
    seen = {}

    async def fake_run(cfg, config_path, root, model, max_turns, goal):
        seen.update(cfg=cfg, config_path=config_path, root=root, model=model, max_turns=max_turns, goal=goal)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    return seen


@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_cli_analyze_loads_config(target, captured_run):
    path = CONFIGS / f"{target}.yaml"
    assert agent.main(["analyze", "--target", target, "--config", str(path), "--max-turns", "5"]) == 0
    assert captured_run["cfg"].target == target and captured_run["config_path"] == path
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
