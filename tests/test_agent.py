import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from sage import agent
from sage.artifacts import build_metadata
from sage.config import MpiConfig, TargetConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
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
                        run="{binary} -s 5", env={"GREETING": "hi"},
                        mpi=MpiConfig(compiler="mpicxx", build=['make CXX="$SAGE_COMPILER" CXXFLAGS="$SAGE_FLAGS"'],
                                      binary="build-mpi/app", launcher="mpirun -np {ranks}", ranks=8,
                                      run="{binary} -s 3", env={"OMP_NUM_THREADS": "1"}))


@pytest.fixture
def scripts(monkeypatch):
    """Stand-in for scripts/*.sh: writes what the real scripts write, returns the given exit codes."""
    calls, codes = [], {"preflight.sh": 0, "build_target.sh": 0, "profile_target.sh": 0}

    def fake(name, *args, cwd):
        calls.append((name, args))
        if name == "preflight.sh":
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(PREFLIGHT))
        elif name == "profile_target.sh":
            run_dir = Path(args[args.index("--runs") + 1]) / args[1]
            ranks = args[args.index("--ranks") + 1] if "--ranks" in args else ""
            run_dir.mkdir(parents=True)
            (run_dir / "perf-stat.txt").write_text((FIXTURES / "perf-stat-software.txt").read_text())
            for d in [run_dir / f"rank-{r}" for r in range(int(ranks))] if ranks else [run_dir]:
                d.mkdir(exist_ok=True)
                (d / "perf-report.txt").write_text((FIXTURES / "perf-report.txt").read_text())
            work = Path(args[args.index("--work") + 1]) / args[0]
            fields = {"run_id": args[1], "status": "ok", "run_command": f"{work}/build/app -s 5",
                      "ranks": ranks, "launcher": f"mpirun -np {ranks}" if ranks else "",
                      "build_info": str(work / "build-info.json")}
            (run_dir / "metadata.json").write_text(json.dumps(build_metadata(run_dir, fields, {"OMP_NUM_THREADS": "4"})))
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
                "build_commands": ["make"], "variant": "mpi" if "--mpi" in args else "serial",
                "compiler": "g++"}))
        return codes[name]

    monkeypatch.setattr(agent, "run_script", fake)
    return calls, codes


def findings_for(options, status="bottlenecks_found"):
    """What a well-behaved agent writes: a valid findings.json citing the baseline's perf-report.txt."""
    run_dir = Path(options.cwd)
    finding = {"rank": 1, "function": "ljForce", "source_location": "src-mpi/ljForce.c:139-188",
               "sample_percent": 85.04, "bottleneck_type": "compute_bound",
               "supporting_evidence": ["85% of samples in ljForce"], "likely_cause": "scalar inner loop",
               "experiment": "size scan", "experiment_result": "flat per-atom cost", "confidence": "high",
               "artifact_paths": ["perf-report.txt" if (run_dir / "perf-report.txt").exists() else "rank-0/perf-report.txt"]}
    return {"schema_version": 1, "run_id": run_dir.name, "target": "lulesh", "status": status,
            "summary": "ljForce dominates.", "findings": [finding] if status == "bottlenecks_found" else []}


def fake_query(calls, side_effect=None, findings=findings_for):
    async def query(*, prompt, options):
        calls.append((prompt, options))
        if side_effect:
            side_effect()
        data = findings(options) if findings else None
        if data is not None:
            (Path(options.cwd) / "findings.json").write_text(data if isinstance(data, str) else json.dumps(data))
        (Path(options.cwd) / "report.md").write_text("agent-written report, must be replaced")
        yield AssistantMessage(content=[TextBlock(text="checking perf"),
                                        ToolUseBlock(id="t1", name="Bash", input={"command": "perf --version"})],
                               model="test")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                            num_turns=2, session_id="s", result="findings.json is complete")
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


def test_run_preflight_build_profile_then_agent(cfg, tmp_path, scripts, monkeypatch, capsys):
    script_calls, _ = scripts
    calls = []
    monkeypatch.setattr(agent, "query", fake_query(calls))
    root = tmp_path / "project"
    root.mkdir()
    config = Path("cfg.yaml").resolve()
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), root, "r1", label="base", goal="memory")) == 0

    work, run_dir = root / ".work" / "lulesh", root / "runs" / "r1"
    assert [c[0] for c in script_calls] == ["preflight.sh", "build_target.sh", "profile_target.sh"]
    assert script_calls[0][1] == ("--output", str(root / ".work" / "preflight.json"))
    assert script_calls[1][1] == ("lulesh", "--config", str(config), "--work", str(root / ".work"))
    assert script_calls[2][1] == ("lulesh", "r1", "--config", str(config), "--work", str(root / ".work"),
                                  "--runs", str(root / "runs"), "--label", "base")
    report = (run_dir / "report.md").read_text()
    assert report.startswith("# LULESH performance bottlenecks: run r1") and "agent-written" not in report
    assert "| 1 | `ljForce` |" in report and "## Run" in report

    [(prompt, options)] = calls
    assert options.cwd == str(run_dir)
    assert f"Baseline artifact set (run id r1): {run_dir}" in prompt
    assert "app wall-clock: 0.0931 s mean over 3 runs (+- 34.29%)" in prompt
    assert "perf stat counted: task-clock, context-switches, cpu-migrations, page-faults" in prompt
    assert "not available: cycles, instructions" in prompt
    assert "top self time [448 samples of task-clock:ppp]: ljForce (CoMD-serial) 85.04%" in prompt
    assert f"{agent.SCRIPTS}/annotate_symbol.sh r1 <symbol> --runs {root / 'runs'}" in prompt
    assert f"{agent.SCRIPTS}/profile_target.sh lulesh <name> --runs {run_dir}/experiments --work {root / '.work'}" in prompt
    assert "--rank <r>" not in prompt
    assert COMMIT in prompt and "g++ (Fake) 13.0" in prompt and "-O3 -g -fno-omit-frame-pointer" in prompt
    assert "cycles=no" in prompt and "warning: only software events are available" in prompt
    assert "memory" in prompt
    out = capsys.readouterr()
    assert "**Status:** Bottlenecks found" in out.out and "[Bash] perf --version" in out.err
    assert f"{agent.SCHEMA_PATH}" in prompt and f"-m sage.findings validate {run_dir}" in prompt


def test_run_mpi_builds_and_profiles_mpi_variant(cfg, tmp_path, scripts, monkeypatch):
    script_calls, _ = scripts
    calls = []
    monkeypatch.setattr(agent, "query", fake_query(calls))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "m1", mpi=True)) == 0
    assert "--mpi" in script_calls[1][1]
    assert script_calls[2][1][-2:] == ("--ranks", "8")
    [(prompt, options)] = calls
    assert "rank 0 [448 samples" in prompt and "rank 7 [448 samples" in prompt
    assert "annotate_symbol.sh m1 <symbol> --rank <r>" in prompt and "--ranks 8" in prompt
    assert options.env == {"GREETING": "hi", "OMP_NUM_THREADS": "1"}


@pytest.mark.parametrize("failing,message", [("preflight.sh", "preflight failed"), ("build_target.sh", "build failed"),
                                             ("profile_target.sh", "baseline profiling failed")])
def test_run_stops_when_a_script_fails(cfg, tmp_path, scripts, monkeypatch, capsys, failing, message):
    script_calls, codes = scripts
    codes[failing] = 1
    monkeypatch.setattr(agent, "query", lambda **kw: pytest.fail("agent should not start"))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 1
    assert message in capsys.readouterr().err
    assert script_calls[-1][0] == failing


def test_run_allows_generated_files_in_checkout(cfg, tmp_path, scripts, monkeypatch):
    gen = lambda: (tmp_path / ".work/lulesh/src/generated.h").write_text("// generated")
    monkeypatch.setattr(agent, "query", fake_query([], gen))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 0


def test_run_fails_if_tracked_source_modified(cfg, tmp_path, scripts, monkeypatch, capsys):
    tamper = lambda: (tmp_path / ".work/lulesh/src/main.c").write_text("// optimized\n")
    monkeypatch.setattr(agent, "query", fake_query([], tamper))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 3
    assert "source was modified" in capsys.readouterr().err


def test_real_scripts_exist_and_are_executable():
    for name in ("preflight.sh", "build_target.sh", "profile_target.sh", "annotate_symbol.sh"):
        path = agent.SCRIPTS / name
        assert path.is_file() and path.stat().st_mode & 0o111


# --- CLI ----------------------------------------------------------------------

@pytest.fixture
def captured_run(monkeypatch):
    seen = {}

    async def fake_run(cfg, config_path, root, run_id, **kw):
        seen.update(cfg=cfg, config_path=config_path, root=root, run_id=run_id, **kw)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    return seen


@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_cli_analyze_loads_config(target, captured_run):
    path = CONFIGS / f"{target}.yaml"
    assert agent.main(["analyze", "--target", target, "--config", str(path), "--max-turns", "5"]) == 0
    assert captured_run["cfg"].target == target and captured_run["config_path"] == path
    assert captured_run["max_turns"] == 5 and captured_run["root"] == Path.cwd()
    assert captured_run["run_id"].startswith(f"{target}-") and captured_run["mpi"] is False


def test_cli_run_options(captured_run):
    argv = ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"),
            "--run-id", "trial.1", "--label", "first try", "--mpi"]
    assert agent.main(argv) == 0
    assert captured_run["run_id"] == "trial.1" and captured_run["label"] == "first try" and captured_run["mpi"] is True


def test_cli_run_option_errors(captured_run, tmp_path, monkeypatch, capsys):
    base = ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml")]
    assert "invalid run id 'a/b'" in cli_error(base + ["--run-id", "a/b"], capsys)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    assert "runs/taken already exists" in cli_error(base + ["--run-id", "taken"], capsys)
    no_mpi = tmp_path / "no-mpi.yaml"
    no_mpi.write_text("\n".join(l for l in (CONFIGS / "comd.yaml").read_text().split("\nmpi:")[0].splitlines()))
    err = cli_error(["analyze", "--target", "comd", "--config", str(no_mpi), "--mpi"], capsys)
    assert "--mpi needs an mpi section" in err
    assert not captured_run


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


@pytest.mark.parametrize("findings,message", [
    (None, "findings.json not found"),
    ("{broken", "is not valid JSON"),
    (lambda o: {**findings_for(o), "findings": []}, "status is bottlenecks_found but there are no findings"),
    (lambda o: {**findings_for(o), "findings": [{**findings_for(o)["findings"][0], "artifact_paths": ["nope.txt"]}]},
     "'nope.txt' does not exist in the run directory"),
])
def test_run_fails_without_valid_findings(cfg, tmp_path, scripts, monkeypatch, capsys, findings, message):
    produce = findings if callable(findings) else (lambda o: findings)
    monkeypatch.setattr(agent, "query", fake_query([], findings=produce if findings is not None else None))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 4
    err = capsys.readouterr().err
    assert "findings.json is missing or invalid" in err and message in err
    assert not (tmp_path / "runs/r1/report.md").exists()  # the agent's own report.md was discarded


def test_inconclusive_run_succeeds(cfg, tmp_path, scripts, monkeypatch, capsys):
    monkeypatch.setattr(agent, "query", fake_query([], findings=lambda o: findings_for(o, "inconclusive")))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 0
    assert "**Status:** Inconclusive" in (tmp_path / "runs/r1/report.md").read_text()


def test_source_modification_outranks_invalid_findings(cfg, tmp_path, scripts, monkeypatch):
    tamper = lambda: (tmp_path / ".work/lulesh/src/main.c").write_text("// optimized\n")
    monkeypatch.setattr(agent, "query", fake_query([], tamper, findings=None))
    assert asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1")) == 3


def test_system_prompt_asks_for_findings_json():
    assert "findings.json" in agent.SYSTEM_PROMPT and "do not write report.md yourself" in agent.SYSTEM_PROMPT
