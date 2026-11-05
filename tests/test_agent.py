"""The agent wiring: prompts, limits, options, the retry and the transcript, with a fake SDK client.

No API is called: ClaudeSDKClient is replaced by FakeClient, which plays scripted turns (each turn
can write files into the run directory, as the real agent would, sleep, or end with an error).
"""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultError, ResultMessage, TextBlock, ToolUseBlock

from sage import agent
from sage.config import MpiConfig, TargetConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
COMMIT = "a" * 40
PREFLIGHT = {
    "ok": True, "host": {"cpu_model": "Fake CPU", "cpus": 4, "arch": "x86_64", "kernel": "6.8"},
    "perf": {"version": "perf version 6.9"}, "perf_event_paranoid": 1,
    "events": {"cycles": False, "instructions": False, "software": True, "topdown": False},
    "warnings": ["only software events are available"], "errors": [],
}


def make_cfg(commit=COMMIT):
    return TargetConfig(target="lulesh", repo_url="https://example.invalid/up.git", commit=commit,
                        compiler="g++", flags="-O3 -g -fno-omit-frame-pointer",
                        build=['make CXX="$SAGE_COMPILER" CXXFLAGS="$SAGE_FLAGS"'], binary="build/lulesh2.0",
                        run="{binary} -s 5", env={"OMP_NUM_THREADS": "4"},
                        mpi=MpiConfig(compiler="mpicxx", build=['make CXX="$SAGE_COMPILER" CXXFLAGS="$SAGE_FLAGS"'],
                                      binary="build-mpi/lulesh2.0", launcher="mpirun -np {ranks}", ranks=8,
                                      run="{binary} -s 3", env={"OMP_NUM_THREADS": "1"}))


@pytest.fixture
def cfg():
    return make_cfg()


def make_run(tmp_path, cfg, mpi=False, label=None):
    return agent.Run(cfg=cfg, config_path=Path("cfg.yaml"), root=tmp_path, run_id="r1", mpi=mpi, label=label)


# --- prompt loading ---------------------------------------------------------------------

def test_prompt_files_exist_and_use_only_known_placeholders():
    for name in ("system", "task", "retry"):
        text = (agent.PROMPTS / f"{name}.md").read_text()
        assert agent.PLACEHOLDER.findall(text), name


def test_load_prompt_requires_every_value_and_rejects_unknown_ones():
    with pytest.raises(agent.PromptError, match="needs values for: max_turns, timeout_minutes"):
        agent.load_prompt("system", max_experiments=1)
    with pytest.raises(agent.PromptError, match="has no placeholders for: bogus"):
        agent.load_prompt("system", max_experiments=1, max_turns=2, timeout_minutes=3, bogus=4)
    with pytest.raises(FileNotFoundError):
        agent.load_prompt("nonexistent")


def test_system_prompt_rules_and_limits():
    text = agent.system_prompt(agent.Limits(max_turns=33, timeout_s=1800, max_experiments=2))
    assert "{{" not in text and "}}" not in text
    assert "Do not rely on prior knowledge of this application" in text
    assert "Separate measured facts from inferences" in text and "`supporting_evidence` holds only measured facts" in text
    assert "at most 2 experiment(s)" in text and "33 turns and 30 minutes" in text
    for step in ("1. **Preflight and build.**", "2. **Baseline.**", "3. **Inspect.**", "4. **Annotate and read source.**",
                 "5. **Hypothesis.**", "6. **Experiment.**", "7. **Findings.**"):
        assert step in text
    assert "Change exactly one" in text and "factor" in text
    assert "validate command and fix every problem it reports" in " ".join(text.split())
    assert "perf is the only profiler" in text and "source checkout is read-only" in text
    # no application-specific hints any more
    for leaked in ("LULESH", "CoMD", "Lennard-Jones", "ljForce", "hydrodynamics"):
        assert leaked not in text.replace("ljForce has 91.8%", "")


@pytest.mark.parametrize("mpi", [False, True])
@pytest.mark.parametrize("goal", [None, "memory"])
def test_task_prompt_is_complete(tmp_path, cfg, mpi, goal):
    r = make_run(tmp_path, cfg, mpi=mpi, label="first try")
    text = agent.task_prompt(r, agent.Limits(max_experiments=1), PREFLIGHT, goal)
    assert "{{" not in text
    run_dir = tmp_path / "runs" / "r1"
    assert f"Run directory (your working directory): {run_dir}" in text
    assert f"profile_target.sh lulesh baseline --runs {run_dir} --work {tmp_path / '.work'}" in text
    assert f"--runs {run_dir / 'experiments'}" in text and "--cmd '{binary} <arguments>'" in text
    assert f"annotate_symbol.sh baseline <symbol>{' --rank <r>' if mpi else ''} --runs {run_dir}" in text
    assert f"-m sage.findings validate {run_dir}" in text
    assert "--label 'first try'" in text
    assert "events: cycles=no, instructions=no, software=yes, topdown=no" in text
    assert "warning: only software events are available" in text
    assert str(agent.SCHEMA_PATH) in text and "1 experiment run(s)" in text
    if mpi:
        assert "build_target.sh lulesh --mpi" in text and "--ranks 8" in text and "MPI analysis with 8 ranks" in text
        assert "build-info-mpi.json" in text and "build-mpi/lulesh2.0 -s 3" in text and "OMP_NUM_THREADS=1" in text
    else:
        assert "--ranks" not in text and "build-info.json" in text and "build/lulesh2.0 -s 5" in text
    assert ("Additional focus: memory" in text) == bool(goal)


def test_retry_prompt():
    text = agent.load_prompt("retry", errors="- run_id: bad", findings="/r/findings.json", validate_command="v")
    assert "- run_id: bad" in text and "/r/findings.json" in text and "This is the only retry" in text


# --- limits -----------------------------------------------------------------------------

def test_limit_defaults():
    limits = agent.Limits()
    assert (limits.model, limits.max_turns, limits.timeout_s, limits.max_experiments) == (None, 80, 3600, 1)


@pytest.mark.parametrize("kwargs,message", [
    ({"max_turns": 0}, "max turns must be a positive integer"),
    ({"max_turns": True}, "max turns must be a positive integer"),
    ({"max_turns": 2.5}, "max turns must be a positive integer"),
    ({"timeout_s": 0}, "timeout must be positive"),
    ({"timeout_s": -5}, "timeout must be positive"),
    ({"max_experiments": -1}, "max experiments must be 0 or more"),
    ({"model": " "}, "model must not be empty"),
])
def test_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        agent.Limits(**kwargs)


@pytest.mark.parametrize("text,seconds", [("90", 90), ("90s", 90), ("45m", 2700), ("1.5h", 5400), (" 2m ", 120)])
def test_parse_duration(text, seconds):
    assert agent.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "10x", "-5m", "0", "1h30m"])
def test_parse_duration_rejects(text):
    with pytest.raises(ValueError):
        agent.parse_duration(text)


def test_options_carry_limits_and_restrictions(tmp_path, cfg):
    limits = agent.Limits(model="claude-sonnet-5-5", max_turns=12, max_experiments=2)
    opts = agent.build_options(make_run(tmp_path, cfg), limits)
    assert opts.tools == ["Bash", "Read", "Glob", "Grep", "Write", "Edit"] and opts.allowed_tools == opts.tools
    assert not opts.agents and not opts.mcp_servers
    assert opts.permission_mode == "dontAsk" and opts.setting_sources == []
    assert opts.model == "claude-sonnet-5-5" and opts.max_turns == 12
    assert opts.cwd == str(tmp_path / "runs" / "r1") and opts.add_dirs == [str(tmp_path / ".work" / "lulesh")]
    assert opts.env["OMP_NUM_THREADS"] == "4" and opts.env["PYTHON"]
    assert "at most 2 experiment(s)" in opts.system_prompt
    assert agent.build_options(make_run(tmp_path, cfg, mpi=True), limits).env["OMP_NUM_THREADS"] == "1"


def test_experiment_limit_reaches_the_hook(tmp_path, cfg):
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "experiments" / "first").mkdir(parents=True)
    [matcher] = agent.build_options(make_run(tmp_path, cfg), agent.Limits(max_experiments=1)).hooks["PreToolUse"]
    command = f"/x/profile_target.sh lulesh second --runs {run_dir / 'experiments'} --cmd '{{binary}}'"
    out = asyncio.run(matcher.hooks[0]({"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(run_dir)},
                                       "id", None))
    assert "Experiment limit reached: 1 of 1" in out["hookSpecificOutput"]["permissionDecisionReason"]


# --- the session: fake client ---------------------------------------------------------------

def findings(run_dir, status="bottlenecks_found", artifact="baseline/perf-report.txt"):
    finding = {"rank": 1, "function": "f", "source_location": "a.c:10", "sample_percent": 80.0,
               "bottleneck_type": "compute_bound", "supporting_evidence": ["80% of samples in f (baseline/perf-report.txt)"],
               "likely_cause": "scalar loop", "experiment": "size scan", "experiment_result": "flat",
               "confidence": "medium", "artifact_paths": [artifact]}
    return {"schema_version": 1, "run_id": run_dir.name, "target": "lulesh", "status": status,
            "summary": "f dominates.", "findings": [finding] if status == "bottlenecks_found" else []}


def write_valid(run_dir):
    (run_dir / "baseline").mkdir(exist_ok=True)
    (run_dir / "baseline" / "perf-report.txt").write_text((FIXTURES / "perf-report.txt").read_text())
    (run_dir / "findings.json").write_text(json.dumps(findings(run_dir)))
    (run_dir / "report.md").write_text("agent-written, must be replaced")


def write_invalid(run_dir):
    (run_dir / "findings.json").write_text(json.dumps(findings(run_dir, artifact="missing.txt")))


def do_nothing(run_dir):
    pass


class FakeClient:
    """Plays one scripted turn per query(); each turn is a callable(run_dir), optionally async."""
    instances: list["FakeClient"] = []
    turns: list = []

    def __init__(self, options):
        self.options, self.prompts, self.closed = options, [], False
        FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        turn = FakeClient.turns[len(self.prompts) - 1]
        result = turn(Path(self.options.cwd))
        if asyncio.iscoroutine(result):
            await result
        yield AssistantMessage(content=[TextBlock(text="working"),
                                        ToolUseBlock(id="t1", name="Bash", input={"command": "perf --version"})],
                               model="test")
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=3,
                            session_id="s", result="findings.json is complete")


@pytest.fixture
def fake(monkeypatch):
    calls = {"scripts": [], "preflight_rc": 0}

    def run_script(name, *args, cwd):
        calls["scripts"].append((name, args))
        out = Path(args[args.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(PREFLIGHT))
        return calls["preflight_rc"]

    FakeClient.instances, FakeClient.turns = [], []
    monkeypatch.setattr(agent, "run_script", run_script)
    monkeypatch.setattr(agent, "ClaudeSDKClient", FakeClient)

    def play(*turns):
        FakeClient.turns = list(turns)
    calls["play"] = play
    return calls


def go(cfg, tmp_path, limits=agent.Limits(), **kw):
    return asyncio.run(agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1", limits=limits, **kw))


def transcript(tmp_path):
    return [json.loads(l) for l in (tmp_path / "runs/r1/transcript.jsonl").read_text().splitlines()]


def test_successful_analysis(cfg, tmp_path, fake, capsys):
    fake["play"](write_valid)
    assert go(cfg, tmp_path, goal="memory") == agent.OK
    [client] = FakeClient.instances
    assert len(client.prompts) == 1 and "Additional focus: memory" in client.prompts[0] and client.closed
    assert [s[0] for s in fake["scripts"]] == ["preflight.sh"]  # build and baseline are the agent's job
    report = (tmp_path / "runs/r1/report.md").read_text()
    assert report.startswith("# LULESH performance bottlenecks: run r1") and "agent-written" not in report
    out = capsys.readouterr()
    assert "**Status:** Bottlenecks found" in out.out and "[Bash] perf --version" in out.err

    records = transcript(tmp_path)
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "start" and kinds[1] == "prompt" and kinds[-1] == "end"
    assert records[0]["limits"] == {"model": None, "max_turns": 80, "timeout_s": 3600, "max_experiments": 1}
    assert "Separate measured facts from inferences" in records[0]["system_prompt"]
    assert records[0]["tools"] == agent.TOOLS
    messages = [r for r in records if r["kind"] == "message"]
    assert [m["type"] for m in messages] == ["AssistantMessage", "ResultMessage"]
    assert messages[0]["message"]["content"][1]["input"] == {"command": "perf --version"}
    assert [r["errors"] for r in records if r["kind"] == "validation"] == [[]]
    assert records[-1]["status"] == 0


def test_inconclusive_without_findings_succeeds(cfg, tmp_path, fake):
    fake["play"](lambda d: (d / "findings.json").write_text(json.dumps(findings(d, status="inconclusive"))))
    assert go(cfg, tmp_path) == agent.OK
    assert "**Status:** Inconclusive" in (tmp_path / "runs/r1/report.md").read_text()


def test_invalid_findings_get_one_retry_that_succeeds(cfg, tmp_path, fake, capsys):
    fake["play"](write_invalid, write_valid)
    assert go(cfg, tmp_path) == agent.OK
    [client] = FakeClient.instances
    assert len(client.prompts) == 2
    assert "'missing.txt' does not exist in the run directory" in client.prompts[1]
    assert "This is the only retry" in client.prompts[1]
    assert "asking the agent to fix it once" in capsys.readouterr().err
    validations = [r for r in transcript(tmp_path) if r["kind"] == "validation"]
    assert [v["attempt"] for v in validations] == ["analysis", "retry"]
    assert validations[0]["errors"] and validations[1]["errors"] == []
    assert sum(r["kind"] == "prompt" for r in transcript(tmp_path)) == 2


@pytest.mark.parametrize("first,second,message", [
    (write_invalid, write_invalid, "'missing.txt' does not exist"),
    (do_nothing, do_nothing, "findings.json not found"),
    (write_invalid, do_nothing, "'missing.txt' does not exist"),
])
def test_invalid_after_retry_fails_clearly(cfg, tmp_path, fake, capsys, first, second, message):
    fake["play"](first, second, pytest.fail)  # a third turn must never be requested
    assert go(cfg, tmp_path) == agent.INVALID_FINDINGS
    [client] = FakeClient.instances
    assert len(client.prompts) == 2
    err = capsys.readouterr().err
    assert "findings.json is still invalid after one retry:" in err and message in err
    assert not (tmp_path / "runs/r1/report.md").exists()
    assert transcript(tmp_path)[-1]["status"] == agent.INVALID_FINDINGS


def test_turn_ending_in_error_still_gets_validated_and_retried(cfg, tmp_path, fake, capsys):
    def hits_max_turns(run_dir):
        raise ResultError("Reached maximum number of turns",
                          data={"subtype": "error_max_turns", "terminal_reason": "max_turns", "errors": []})
    fake["play"](hits_max_turns, write_valid)
    assert go(cfg, tmp_path) == agent.OK
    assert len(FakeClient.instances[0].prompts) == 2
    [error] = [r for r in transcript(tmp_path) if r["kind"] == "error"]
    assert error["attempt"] == "analysis" and "maximum number of turns" in error["error"]
    assert "analysis ended with an error (max_turns)" in capsys.readouterr().err


def test_wall_clock_limit_stops_the_agent(cfg, tmp_path, fake, capsys):
    async def slow(run_dir):
        await asyncio.sleep(30)
    fake["play"](slow)
    assert go(cfg, tmp_path, limits=agent.Limits(timeout_s=0.2)) == agent.TIMED_OUT
    err = capsys.readouterr().err
    assert "wall-clock limit of 0.00333333 minutes reached" in err and "findings.json is missing or invalid" in err
    assert FakeClient.instances[0].closed  # the session was torn down
    assert any(r["kind"] == "timeout" for r in transcript(tmp_path))
    assert transcript(tmp_path)[-1]["status"] == agent.TIMED_OUT


def test_timeout_keeps_valid_findings(cfg, tmp_path, fake):
    async def write_then_hang(run_dir):
        write_valid(run_dir)
        await asyncio.sleep(30)
    fake["play"](write_then_hang)
    assert go(cfg, tmp_path, limits=agent.Limits(timeout_s=0.2)) == agent.TIMED_OUT
    assert (tmp_path / "runs/r1/report.md").is_file()


def test_preflight_failure_stops_before_the_agent(cfg, tmp_path, fake, capsys):
    fake["preflight_rc"] = 1
    assert go(cfg, tmp_path) == agent.FAILED
    assert FakeClient.instances == [] and not (tmp_path / "runs/r1").exists()
    assert "preflight failed" in capsys.readouterr().err


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def test_source_modification_fails_the_run(tmp_path, fake, capsys):
    src = tmp_path / ".work/lulesh/src"
    src.mkdir(parents=True)
    git("init", "-q", cwd=src)
    (src / "main.c").write_text("int main(void) { return 0; }\n")
    git("add", ".", cwd=src)
    git("commit", "-qm", "v1", cwd=src)
    cfg = make_cfg(commit=git("rev-parse", "HEAD", cwd=src))

    def tamper(run_dir):
        write_valid(run_dir)
        (src / "main.c").write_text("// optimized\n")
    fake["play"](tamper)
    assert go(cfg, tmp_path) == agent.SOURCE_MODIFIED
    assert f"{src / 'main.c'} was modified" in capsys.readouterr().err
    assert transcript(tmp_path)[-1]["source_problems"] == [f"{src / 'main.c'} was modified"]


def test_untouched_checkout_passes(tmp_path, fake):
    src = tmp_path / ".work/lulesh/src"
    src.mkdir(parents=True)
    git("init", "-q", cwd=src)
    (src / "main.c").write_text("int main(void) { return 0; }\n")
    git("add", ".", cwd=src)
    git("commit", "-qm", "v1", cwd=src)
    fake["play"](lambda d: (write_valid(d), (src / "main.o").write_text("generated")))
    assert go(make_cfg(commit=git("rev-parse", "HEAD", cwd=src)), tmp_path) == agent.OK


# --- CLI ----------------------------------------------------------------------------------

@pytest.fixture
def captured_run(monkeypatch):
    seen = {}

    async def fake_run(cfg, config_path, root, run_id, **kw):
        seen.update(cfg=cfg, config_path=config_path, root=root, run_id=run_id, **kw)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    return seen


def cli_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        agent.main(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_cli_analyze_defaults(target, captured_run):
    path = CONFIGS / f"{target}.yaml"
    assert agent.main(["analyze", "--target", target, "--config", str(path)]) == 0
    assert captured_run["cfg"].target == target and captured_run["config_path"] == path
    assert captured_run["root"] == Path.cwd() and captured_run["run_id"].startswith(f"{target}-")
    assert captured_run["limits"] == agent.Limits() and captured_run["mpi"] is False


def test_cli_limits(captured_run):
    argv = ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"), "--model", "claude-opus-5-5",
            "--max-turns", "40", "--timeout", "90m", "--max-experiments", "2", "--run-id", "t.1", "--label", "x",
            "--mpi", "--goal", "memory"]
    assert agent.main(argv) == 0
    assert captured_run["limits"] == agent.Limits(model="claude-opus-5-5", max_turns=40, timeout_s=5400,
                                                  max_experiments=2)
    assert (captured_run["run_id"], captured_run["label"], captured_run["mpi"], captured_run["goal"]) == \
        ("t.1", "x", True, "memory")


@pytest.mark.parametrize("extra,message", [
    (["--max-turns", "0"], "max turns must be a positive integer"),
    (["--timeout", "soon"], "invalid duration 'soon'"),
    (["--timeout", "0m"], "duration must be positive"),
    (["--max-experiments", "-1"], "max experiments must be 0 or more"),
    (["--max-turns", "many"], "invalid int value"),
    (["--run-id", "a/b"], "invalid run id 'a/b'"),
])
def test_cli_limit_errors(captured_run, capsys, extra, message):
    assert message in cli_error(["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"), *extra], capsys)
    assert not captured_run


def test_cli_target_and_config_errors(captured_run, tmp_path, capsys, monkeypatch):
    assert "invalid target 'hpcg' (choose from comd, lulesh)" in cli_error(
        ["analyze", "--target", "hpcg", "--config", str(CONFIGS / "lulesh.yaml")], capsys)
    assert "config is for target 'lulesh' but --target is 'comd'" in cli_error(
        ["analyze", "--target", "comd", "--config", str(CONFIGS / "lulesh.yaml")], capsys)
    assert "config file not found" in cli_error(
        ["analyze", "--target", "comd", "--config", str(tmp_path / "missing.yaml")], capsys)
    no_mpi = tmp_path / "no-mpi.yaml"
    no_mpi.write_text((CONFIGS / "comd.yaml").read_text().split("\nmpi:")[0])
    assert "--mpi needs an mpi section" in cli_error(
        ["analyze", "--target", "comd", "--config", str(no_mpi), "--mpi"], capsys)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    assert "runs/taken already exists" in cli_error(
        ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"), "--run-id", "taken"], capsys)
    assert "required" in cli_error([], capsys)
    assert not captured_run
