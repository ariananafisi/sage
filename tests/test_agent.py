"""The agent wiring: prompts, limits, the HokieAI client, the tool loop, the retry and the transcript.

No API is called: FakeClient stands in for openai.OpenAI and returns scripted Chat Completions
responses. Tool calls in those responses run for real (in a temporary run directory).
"""

import json
import subprocess
import time
from pathlib import Path

import openai
import openai._exceptions
import pytest
from openai.types.chat import ChatCompletion

from sage import agent
from sage.config import MpiConfig, TargetConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HTTPX = getattr(openai._exceptions, "httpx2", None) or getattr(openai._exceptions, "httpx")
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

def test_prompt_files_exist_and_use_placeholders():
    for name in ("system", "task", "retry"):
        assert agent.PLACEHOLDER.findall((agent.PROMPTS / f"{name}.md").read_text()), name


def test_load_prompt_requires_every_value_and_rejects_unknown_ones():
    with pytest.raises(agent.PromptError, match="needs values for: max_turns, timeout_minutes"):
        agent.load_prompt("system", max_experiments=1)
    with pytest.raises(agent.PromptError, match="has no placeholders for: bogus"):
        agent.load_prompt("system", max_experiments=1, max_turns=2, timeout_minutes=3, bogus=4)
    with pytest.raises(FileNotFoundError):
        agent.load_prompt("nonexistent")


def test_system_prompt_rules_tools_and_limits():
    text = agent.system_prompt(agent.Limits(max_turns=33, timeout_s=1800, max_experiments=2))
    flat = " ".join(text.split())
    assert "{{" not in text and "}}" not in text
    assert "Do not rely on prior knowledge of this application" in text
    assert "Separate measured facts from inferences" in text and "`supporting_evidence` holds only measured facts" in flat
    # the diagnosis method: general measurements and decision rules, nothing application-specific
    for rule in ("# Diagnosis method", "about 1.1 or more is uneven", "the least runtime time", "working-set experiment",
                 "per unit B/A", "0.6 or less means memory-bound", "about 0.9 to 1.1", "Between about 0.6 and 0.9", "not from the options you passed", "samples skid",
                 "one per application function with at least 3% of self time", "Read baseline/summary.md first"):
        assert rule in flat, rule
    assert "at most 2 experiment(s)" in text and "33 model calls and 30 minutes" in flat
    for step in ("1. **Preflight and build.**", "2. **Baseline.**", "3. **Inspect.**", "4. **Annotate and read source.**",
                 "5. **Hypothesis.**", "6. **Experiment.**", "7. **Findings.**"):
        assert step in text
    assert "changing exactly one factor" in flat
    for tool in ("`bash`", "`read_file`", "`search`", "`write_findings`"):
        assert tool in text
    for stale in ("Write tool", "Grep/Read", "Edit or Write"):
        assert stale not in text
    assert "validate command and fix every problem it reports" in flat
    for leaked in ("LULESH", "CoMD", "Lennard-Jones", "hydrodynamics"):
        assert leaked not in text


@pytest.mark.parametrize("mpi", [False, True])
@pytest.mark.parametrize("goal", [None, "memory"])
def test_task_prompt_is_complete(tmp_path, cfg, mpi, goal):
    text = agent.task_prompt(make_run(tmp_path, cfg, mpi=mpi, label="first try"), agent.Limits(), PREFLIGHT, goal)
    run_dir = tmp_path / "runs" / "r1"
    assert "{{" not in text
    assert f"profile_target.sh lulesh baseline --runs {run_dir} --work {tmp_path / '.work'}" in text
    assert f"--runs {run_dir / 'experiments'}" in text and "[--env NAME=VALUE]" in text
    assert f"-m sage.findings validate {run_dir}" in text and "--label 'first try'" in text
    assert "events: cycles=no, instructions=no, software=yes, topdown=no" in text
    assert str(agent.SCHEMA_PATH) in text
    assert ("--ranks 8" in text) == mpi and ("build-mpi/lulesh2.0 -s 3" in text) == mpi
    assert ("Additional focus: memory" in text) == bool(goal)


def test_retry_prompt_names_the_write_tool():
    text = agent.load_prompt("retry", errors="- run_id: bad", findings="/r/findings.json", validate_command="v")
    assert "- run_id: bad" in text and "write_findings" in text and "This is the only retry" in text


# --- limits and HokieAI settings ------------------------------------------------------------------

def test_limit_defaults():
    assert agent.Limits() == agent.Limits(model="gpt-5", max_turns=80, timeout_s=3600, max_experiments=1)


@pytest.mark.parametrize("kwargs,message", [
    ({"max_turns": 0}, "max turns must be a positive integer"),
    ({"max_turns": True}, "max turns must be a positive integer"),
    ({"timeout_s": 0}, "timeout must be positive"),
    ({"max_experiments": -1}, "max experiments must be 0 or more"),
    ({"model": " "}, "model must not be empty"),
])
def test_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        agent.Limits(**kwargs)


@pytest.mark.parametrize("text,seconds", [("90", 90), ("90s", 90), ("45m", 2700), ("1.5h", 5400)])
def test_parse_duration(text, seconds):
    assert agent.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "10x", "-5m", "0", "1h30m"])
def test_parse_duration_rejects(text):
    with pytest.raises(ValueError):
        agent.parse_duration(text)


def test_read_dotenv(tmp_path):
    (tmp_path / ".env").write_text('# comment\n\nexport HOKIEAI_API="abc 123"\nHOKIEAI_BASE_URL=https://x/v1\n'
                                   "QUOTED='q'\nnot a line\n1BAD=x\n")
    assert agent.read_dotenv(tmp_path / ".env") == {"HOKIEAI_API": "abc 123", "HOKIEAI_BASE_URL": "https://x/v1",
                                                     "QUOTED": "q"}
    assert agent.read_dotenv(tmp_path / "missing") == {}


def test_hokieai_settings_environment_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("HOKIEAI_API", raising=False)
    monkeypatch.delenv("HOKIEAI_BASE_URL", raising=False)
    assert agent.hokieai_settings(tmp_path) == (None, agent.DEFAULT_BASE_URL)
    (tmp_path / ".env").write_text("HOKIEAI_API=from-file\nHOKIEAI_BASE_URL=https://file/v1\n")
    assert agent.hokieai_settings(tmp_path) == ("from-file", "https://file/v1")
    monkeypatch.setenv("HOKIEAI_API", "from-env")
    assert agent.hokieai_settings(tmp_path)[0] == "from-env"


def test_client_sends_key_only_in_api_key_header():
    client = agent.make_client("secret-key", "https://gateway.example/openai/v1")
    assert str(client.base_url).rstrip("/") == "https://gateway.example/openai/v1"
    assert client.default_headers["api-key"] == "secret-key"
    assert client.api_key == "hokieai" and "secret-key" not in client.default_headers.get("Authorization", "")


# --- the tool loop with a fake client ------------------------------------------------------------

def reply(text=None, calls=()):
    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-5-2025-08-07",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": {
            "role": "assistant", "content": text,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)] or None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})


def findings(run_dir, status="bottlenecks_found", artifact="baseline/perf-report.txt"):
    finding = {"rank": 1, "function": "f", "source_location": "a.c:10", "sample_percent": 80.0,
               "bottleneck_type": "compute_bound", "supporting_evidence": ["80% of samples in f (baseline/perf-report.txt)"],
               "likely_cause": "scalar loop", "experiment": "size scan", "experiment_result": "flat",
               "confidence": "medium", "artifact_paths": [artifact]}
    return json.dumps({"schema_version": 1, "run_id": run_dir.name, "target": "lulesh", "status": status,
                       "summary": "f dominates.", "findings": [finding] if status == "bottlenecks_found" else []})


def with_baseline(run_dir):
    (run_dir / "baseline").mkdir(exist_ok=True)
    (run_dir / "baseline" / "perf-report.txt").write_text((FIXTURES / "perf-report.txt").read_text())


def write_valid(run_dir):
    with_baseline(run_dir)
    return reply(calls=[("write_findings", {"content": findings(run_dir)}),
                        ("bash", {"command": "echo agent-report > report.md"})])


def write_invalid(run_dir):
    return reply(calls=[("write_findings", {"content": findings(run_dir, artifact="missing.txt")})])


def done(run_dir):
    return reply("findings.json is complete.")


class FakeClient:
    """Each create() call returns the next scripted step: step(run_dir) -> ChatCompletion (or raises)."""

    def __init__(self, steps, run_dir):
        self.steps, self.run_dir, self.requests, self.timeouts = list(steps), run_dir, [], []
        self.chat = self
        self.completions = self

    def with_options(self, timeout):
        self.timeouts.append(timeout)
        return self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": json.loads(json.dumps(kwargs["messages"]))})
        if not self.steps:
            pytest.fail("the agent made more model calls than scripted")
        return self.steps.pop(0)(self.run_dir)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = {"preflight_rc": 0, "scripts": []}

    def run_script(name, *args, cwd):
        state["scripts"].append(name)
        out = Path(args[args.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(PREFLIGHT))
        return state["preflight_rc"]

    monkeypatch.setattr(agent, "run_script", run_script)
    return state


def go(cfg, tmp_path, steps, limits=agent.Limits(), **kw):
    client = FakeClient(steps, tmp_path / "runs" / "r1")
    status = agent.run(cfg, Path("cfg.yaml"), tmp_path, "r1", client=client, limits=limits, **kw)
    return status, client


def transcript(tmp_path):
    return [json.loads(line) for line in (tmp_path / "runs/r1/transcript.jsonl").read_text().splitlines()]


def test_successful_analysis(cfg, tmp_path, fake, capsys):
    status, client = go(cfg, tmp_path, [write_valid, done], goal="memory",
                        limits=agent.Limits(model="gpt-5-mini"))
    assert status == agent.OK and fake["scripts"] == ["preflight.sh"]
    first, second = client.requests
    assert first["model"] == "gpt-5-mini" and [t["function"]["name"] for t in first["tools"]] == \
        ["bash", "read_file", "search", "write_findings"]
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert "Additional focus: memory" in first["messages"][1]["content"]
    # the tool results go back to the model, matched to their calls
    assistant, tool1, tool2 = second["messages"][2:]
    assert assistant["role"] == "assistant" and [c["id"] for c in assistant["tool_calls"]] == ["call_0", "call_1"]
    assert tool1 == {"role": "tool", "tool_call_id": "call_0", "content": tool1["content"]}
    assert tool1["content"].startswith("saved") and tool2["content"].startswith("exit code 0")
    assert all(0 < t <= 3600 for t in client.timeouts)

    report = (tmp_path / "runs/r1/report.md").read_text()
    assert report.startswith("# LULESH performance bottlenecks: run r1") and "agent-report" not in report
    out = capsys.readouterr()
    assert "**Status:** Bottlenecks found" in out.out
    assert "[write_findings] findings.json (" in out.err and "[bash] echo agent-report > report.md" in out.err

    records = transcript(tmp_path)
    assert [r["kind"] for r in records] == ["start", "prompt", "response", "tool", "tool", "response", "validation", "end"]
    start = records[0]
    assert start["limits"]["model"] == "gpt-5-mini" and start["base_url"] == agent.DEFAULT_BASE_URL
    assert start["tools"] == ["bash", "read_file", "search", "write_findings"]
    assert "Separate measured facts from inferences" in start["system_prompt"]
    assert records[2]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "write_findings"
    assert records[3]["name"] == "write_findings" and records[3]["result"].startswith("saved")
    assert records[-1]["status"] == 0


def test_guard_denial_is_returned_to_the_model(cfg, tmp_path, fake):
    steps = [lambda d: reply(calls=[("bash", {"command": "perf stat -e task-clock ls"})]), write_valid, done]
    status, client = go(cfg, tmp_path, steps)
    assert status == agent.OK
    assert client.requests[1]["messages"][-1]["content"].startswith("DENIED: 'perf stat' runs the application")


def test_inconclusive_without_findings_succeeds(cfg, tmp_path, fake):
    steps = [lambda d: reply(calls=[("write_findings", {"content": findings(d, status="inconclusive")})]), done]
    assert go(cfg, tmp_path, steps)[0] == agent.OK
    assert "**Status:** Inconclusive" in (tmp_path / "runs/r1/report.md").read_text()


def test_invalid_findings_get_one_retry_that_succeeds(cfg, tmp_path, fake, capsys):
    status, client = go(cfg, tmp_path, [write_invalid, done, write_valid, done])
    assert status == agent.OK
    retry = client.requests[2]["messages"][-1]
    assert retry["role"] == "user" and "This is the only retry" in retry["content"]
    assert "'missing.txt' does not exist in the run directory" in retry["content"]
    assert "asking the agent to fix it once" in capsys.readouterr().err
    validations = [r for r in transcript(tmp_path) if r["kind"] == "validation"]
    assert [v["attempt"] for v in validations] == ["analysis", "retry"] and validations[1]["errors"] == []


@pytest.mark.parametrize("steps,message", [
    ([write_invalid, done, write_invalid, done], "'missing.txt' does not exist"),
    ([done, done], "findings.json not found"),
])
def test_invalid_after_retry_fails_clearly(cfg, tmp_path, fake, capsys, steps, message):
    status, client = go(cfg, tmp_path, steps)  # FakeClient fails the test if a third attempt is made
    assert status == agent.INVALID_FINDINGS
    err = capsys.readouterr().err
    assert "findings.json is still invalid after one retry:" in err and message in err
    assert not (tmp_path / "runs/r1/report.md").exists()
    assert transcript(tmp_path)[-1]["status"] == agent.INVALID_FINDINGS


def test_turn_limit_then_retry(cfg, tmp_path, fake, capsys):
    busy = lambda d: reply(calls=[("bash", {"command": "true"})])
    status, client = go(cfg, tmp_path, [busy, busy, write_valid, done], limits=agent.Limits(max_turns=2))
    assert status == agent.OK and len(client.requests) == 4
    assert "stopped after the turn limit (2)" in capsys.readouterr().err
    assert any(r["kind"] == "max_turns" for r in transcript(tmp_path))


def test_turn_limit_on_both_attempts_fails(cfg, tmp_path, fake):
    busy = lambda d: reply(calls=[("bash", {"command": "true"})])
    status, client = go(cfg, tmp_path, [busy] * 4, limits=agent.Limits(max_turns=2))
    assert status == agent.INVALID_FINDINGS and len(client.requests) == 4


def test_wall_clock_limit_stops_a_long_command(cfg, tmp_path, fake, capsys):
    slow = lambda d: reply(calls=[("bash", {"command": "sleep 30"})])
    start = time.monotonic()
    status, client = go(cfg, tmp_path, [slow], limits=agent.Limits(timeout_s=0.5))
    assert status == agent.TIMED_OUT and time.monotonic() - start < 10
    err = capsys.readouterr().err
    assert "wall-clock limit of 0.00833333 minutes reached" in err and "findings.json is missing or invalid" in err
    assert transcript(tmp_path)[-1]["status"] == agent.TIMED_OUT


def test_timeout_keeps_valid_findings(cfg, tmp_path, fake):
    def write_then_wait(d):
        with_baseline(d)
        return reply(calls=[("write_findings", {"content": findings(d)}), ("bash", {"command": "sleep 30"})])
    status, _ = go(cfg, tmp_path, [write_then_wait], limits=agent.Limits(timeout_s=1))
    assert status == agent.TIMED_OUT and (tmp_path / "runs/r1/report.md").is_file()


def api_error(status_code, body):
    request = HTTPX.Request("POST", "https://gateway.example/openai/v1/chat/completions")
    response = HTTPX.Response(status_code, request=request)
    cls = {403: openai.PermissionDeniedError, 429: openai.RateLimitError}[status_code]
    return cls("error", response=response, body=body)


def test_api_refusal_fails_clearly(cfg, tmp_path, fake, capsys):
    def refuse(d):
        raise api_error(403, {"error": "ModelNotAllowed", "message": "The specified model is not allowed"})
    status, _ = go(cfg, tmp_path, [refuse])
    assert status == agent.FAILED
    err = capsys.readouterr().err
    assert "the model API refused the request (HTTP 403)" in err and "ModelNotAllowed" in err
    assert transcript(tmp_path)[-1]["status"] == agent.FAILED


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(agent, "sleep", waits.append)
    return waits


def rate_limited(d):
    raise api_error(429, {"statusCode": 429, "message": "OpenAI token limit is already exceeded"})


def test_rate_limit_is_waited_out(cfg, tmp_path, fake, no_sleep, capsys):
    status, client = go(cfg, tmp_path, [rate_limited, rate_limited, write_valid, done])
    assert status == agent.OK and no_sleep == [15, 30] and len(client.requests) == 4
    assert "HokieAI rate limit (HTTP 429" in capsys.readouterr().err
    assert [r["wait_s"] for r in transcript(tmp_path) if r["kind"] == "rate_limited"] == [15, 30]


def test_rate_limit_honours_retry_after(cfg, tmp_path, fake, no_sleep):
    def with_header(d):
        request = HTTPX.Request("POST", "https://gateway.example/openai/v1/chat/completions")
        response = HTTPX.Response(429, request=request, headers={"retry-after": "7"})
        raise openai.RateLimitError("limit", response=response, body={"message": "limit"})
    assert go(cfg, tmp_path, [with_header, write_valid, done])[0] == agent.OK and no_sleep == [7]


def test_persistent_rate_limit_fails_clearly(cfg, tmp_path, fake, no_sleep, capsys):
    status, client = go(cfg, tmp_path, [rate_limited] * 5)
    assert status == agent.FAILED and no_sleep == list(agent.RATE_LIMIT_WAITS)
    assert "HokieAI kept refusing with HTTP 429 after 4 waits" in capsys.readouterr().err


def test_rate_limit_wait_never_passes_the_deadline(cfg, tmp_path, fake, no_sleep):
    status, _ = go(cfg, tmp_path, [rate_limited], limits=agent.Limits(timeout_s=5))
    assert status == agent.TIMED_OUT and no_sleep == []


def test_api_connection_error_fails_clearly(cfg, tmp_path, fake, capsys):
    def unreachable(d):
        raise openai.APIConnectionError(request=HTTPX.Request("POST", "https://gateway.example/x"))
    assert go(cfg, tmp_path, [unreachable])[0] == agent.FAILED
    assert "could not reach the model API" in capsys.readouterr().err


def test_preflight_failure_stops_before_the_agent(cfg, tmp_path, fake, capsys):
    fake["preflight_rc"] = 1
    status, client = go(cfg, tmp_path, [])
    assert status == agent.FAILED and client.requests == [] and not (tmp_path / "runs/r1").exists()
    assert "preflight failed" in capsys.readouterr().err


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def checkout(tmp_path):
    src = tmp_path / ".work/lulesh/src"
    src.mkdir(parents=True)
    git("init", "-q", cwd=src)
    (src / "main.c").write_text("int main(void) { return 0; }\n")
    git("add", ".", cwd=src)
    git("commit", "-qm", "v1", cwd=src)
    return src, git("rev-parse", "HEAD", cwd=src)


def test_source_modification_fails_the_run(tmp_path, fake, capsys):
    src, commit = checkout(tmp_path)

    def tamper(d):  # e.g. a write from inside python -c, which the command check cannot see
        (src / "main.c").write_text("// optimized\n")
        return write_valid(d)
    status, _ = go(make_cfg(commit), tmp_path, [tamper, done])
    assert status == agent.SOURCE_MODIFIED
    assert f"{src / 'main.c'} was modified" in capsys.readouterr().err


def test_shell_edit_of_source_is_denied(tmp_path, fake):
    src, commit = checkout(tmp_path)
    steps = [lambda d: reply(calls=[("bash", {"command": f"echo x > {src}/main.c"})]), write_valid, done]
    status, client = go(make_cfg(commit), tmp_path, steps)
    assert status == agent.OK and "Source files are read-only" in client.requests[1]["messages"][-1]["content"]
    assert (src / "main.c").read_text().startswith("int main")


# --- CLI ----------------------------------------------------------------------------------

@pytest.fixture
def captured_run(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cfg, config_path, root, run_id, **kw):
        seen.update(cfg=cfg, config_path=config_path, root=root, run_id=run_id, **kw)
        return 0
    monkeypatch.setattr(agent, "run", fake_run)
    monkeypatch.setenv("HOKIEAI_API", "test-key")
    monkeypatch.delenv("HOKIEAI_BASE_URL", raising=False)
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
    assert captured_run["run_id"].startswith(f"{target}-") and captured_run["limits"] == agent.Limits()
    assert captured_run["client"].default_headers["api-key"] == "test-key"
    assert captured_run["base_url"] == agent.DEFAULT_BASE_URL


def test_cli_limits_and_options(captured_run, monkeypatch):
    monkeypatch.setenv("HOKIEAI_BASE_URL", "https://other.example/openai/v1")
    argv = ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"), "--model", "gpt-5-mini",
            "--max-turns", "40", "--timeout", "90m", "--max-experiments", "2", "--run-id", "t.1", "--label", "x",
            "--mpi", "--goal", "memory"]
    assert agent.main(argv) == 0
    assert captured_run["limits"] == agent.Limits(model="gpt-5-mini", max_turns=40, timeout_s=5400, max_experiments=2)
    assert (captured_run["run_id"], captured_run["label"], captured_run["mpi"], captured_run["goal"]) == \
        ("t.1", "x", True, "memory")
    assert captured_run["base_url"] == "https://other.example/openai/v1"


def test_cli_reads_key_from_dotenv(captured_run, monkeypatch, tmp_path):
    monkeypatch.delenv("HOKIEAI_API")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HOKIEAI_API=dotenv-key\n")
    assert agent.main(["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml")]) == 0
    assert captured_run["client"].default_headers["api-key"] == "dotenv-key"


def test_cli_without_key_fails_before_running(captured_run, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HOKIEAI_API")
    monkeypatch.chdir(tmp_path)
    err = cli_error(["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml")], capsys)
    assert "HOKIEAI_API is not set: put your HokieAI key in .env" in err and not captured_run


@pytest.mark.parametrize("extra,message", [
    (["--max-turns", "0"], "max turns must be a positive integer"),
    (["--timeout", "soon"], "invalid duration 'soon'"),
    (["--max-experiments", "-1"], "max experiments must be 0 or more"),
    (["--model", ""], "model must not be empty"),
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
    no_mpi = tmp_path / "no-mpi.yaml"
    no_mpi.write_text((CONFIGS / "comd.yaml").read_text().split("\nmpi:")[0])
    assert "--mpi needs an mpi section" in cli_error(
        ["analyze", "--target", "comd", "--config", str(no_mpi), "--mpi"], capsys)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    assert "runs/taken already exists" in cli_error(
        ["analyze", "--target", "comd", "--config", str(CONFIGS / "comd.yaml"), "--run-id", "taken"], capsys)
    assert not captured_run
