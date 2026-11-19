"""bottleneck-agent: a single LLM agent that finds performance bottlenecks with Linux perf.

The model is served by HokieAI (Virginia Tech's Azure API Management gateway to OpenAI models),
through the OpenAI Python SDK's Chat Completions API with function calling.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import openai

from .config import TARGETS, ConfigError, TargetConfig, load_config
from .findings import SCHEMA_PATH, FindingsError, load as load_findings, write_report
from .guard import Policy, source_changes
from .tools import SCHEMAS, DeadlineExceeded, Tools

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PROMPTS = Path(__file__).resolve().parent / "prompts"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # same rule as scripts/profile_target.sh
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

KEY_VAR = "HOKIEAI_API"  # the HokieAI subscription key, from the environment or .env
BASE_URL_VAR = "HOKIEAI_BASE_URL"
DEFAULT_BASE_URL = "https://apim-n1ai-usc-e1f7a372a.azure-api.net/openai/v1"
DEFAULT_MODEL = "gpt-5"

# Exit codes (also in the README).
OK, FAILED, USAGE, SOURCE_MODIFIED, INVALID_FINDINGS, TIMED_OUT, BUDGET_EXHAUSTED = 0, 1, 2, 3, 4, 5, 6


class PromptError(Exception):
    pass


def load_prompt(name: str, **values: object) -> str:
    """Fill sage/prompts/<name>.md. Every {{placeholder}} must be given, and every value used."""
    text = (PROMPTS / f"{name}.md").read_text()
    used = set(PLACEHOLDER.findall(text))
    if missing := used - set(values):
        raise PromptError(f"prompt {name}.md needs values for: {', '.join(sorted(missing))}")
    if unused := set(values) - used:
        raise PromptError(f"prompt {name}.md has no placeholders for: {', '.join(sorted(unused))}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


@dataclass(frozen=True)
class Limits:
    model: str = DEFAULT_MODEL
    max_turns: int = 80  # model calls, for the analysis and again for the retry
    timeout_s: float = 3600  # wall-clock for the whole agent session, retry included
    max_experiments: int = 1  # experiment artifact sets under <run>/experiments

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns < 1:
            raise ValueError(f"max turns must be a positive integer, got {self.max_turns!r}")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or not self.timeout_s > 0:
            raise ValueError(f"timeout must be positive, got {self.timeout_s!r}")
        if isinstance(self.max_experiments, bool) or not isinstance(self.max_experiments, int) \
                or self.max_experiments < 0:
            raise ValueError(f"max experiments must be 0 or more, got {self.max_experiments!r}")


def parse_duration(text: str) -> float:
    """'90', '90s', '45m', '1.5h' -> seconds."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text)
    if not m:
        raise ValueError(f"invalid duration {text!r}: use seconds, or a number followed by s, m or h")
    value = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    if value <= 0:
        raise ValueError(f"duration must be positive, got {text!r}")
    return value


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a .env file (comments, blank lines and an `export ` prefix allowed)."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if sep and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def hokieai_settings(root: Path) -> tuple[str | None, str]:
    """The HokieAI key and base URL: the process environment wins over <root>/.env."""
    dotenv = read_dotenv(root / ".env")
    key = os.environ.get(KEY_VAR) or dotenv.get(KEY_VAR) or None
    base_url = os.environ.get(BASE_URL_VAR) or dotenv.get(BASE_URL_VAR) or DEFAULT_BASE_URL
    return key, base_url


def make_client(key: str, base_url: str) -> openai.OpenAI:
    # HokieAI authenticates with the APIM subscription key in the `api-key` header and ignores
    # Authorization, so the SDK's required api_key gets a placeholder and the key is sent once.
    return openai.OpenAI(base_url=base_url, api_key="hokieai", default_headers={"api-key": key}, max_retries=3)


def run_script(name: str, *args: str, cwd: Path) -> int:
    """Run scripts/<name>; its progress and errors go straight to the user's terminal."""
    env = {**os.environ, "PYTHON": sys.executable}
    return subprocess.run(["bash", str(SCRIPTS / name), *args], cwd=cwd, env=env).returncode


@dataclass(frozen=True)
class Run:
    """Paths and commands for one analysis."""
    cfg: TargetConfig
    config_path: Path
    root: Path
    run_id: str
    mpi: bool = False
    label: str | None = None
    hidden: tuple[Path, ...] = ()  # paths the agent may not read (used by eval)

    @property
    def work_root(self) -> Path: return self.root / ".work"
    @property
    def work(self) -> Path: return self.work_root / self.cfg.target
    @property
    def src(self) -> Path: return self.work / "src"
    @property
    def run_dir(self) -> Path: return self.root / "runs" / self.run_id
    @property
    def preflight(self) -> Path: return self.work_root / "preflight.json"
    @property
    def variant(self): return self.cfg.mpi if self.mpi else self.cfg

    def commands(self) -> dict[str, str]:
        config, work = str(Path(self.config_path).resolve()), str(self.work_root)
        ranks = f" --ranks {self.cfg.mpi.ranks}" if self.mpi else ""
        rank = " --rank <r>" if self.mpi else ""
        profile = f"{SCRIPTS}/profile_target.sh {self.cfg.target}"
        return {
            "build_command": f"{SCRIPTS}/build_target.sh {self.cfg.target}{' --mpi' if self.mpi else ''} "
                             f"--config {config} --work {work}",
            "baseline_command": f"{profile} baseline --runs {self.run_dir} --work {work} --config {config}"
                                f"{ranks} --label {_quote(self.label or 'baseline')}",
            "annotate_command": f"{SCRIPTS}/annotate_symbol.sh baseline <symbol>{rank} --runs {self.run_dir}",
            "experiment_command": f"{profile} <name> --runs {self.run_dir / 'experiments'} --work {work} "
                                  f"--config {config}{ranks} --label '<the one factor changed>' "
                                  f"[--cmd '{{binary}} <arguments>'] [--env NAME=VALUE]",
            "annotate_experiment_command": f"{SCRIPTS}/annotate_symbol.sh <name> <symbol>{rank} "
                                           f"--runs {self.run_dir / 'experiments'}",
            "compare_command": f"{sys.executable} -m sage.analysis compare baseline experiments/<name> "
                               f"[--work <units in baseline> <units in experiment>] | tee compare-<name>.md",
            "validate_command": f"{sys.executable} -m sage.findings validate {self.run_dir}",
        }

    def env(self) -> dict[str, str]:
        return {**self.cfg.env, **(self.cfg.mpi.env if self.mpi else {}), "PYTHON": sys.executable}

    def policy(self, limits: Limits) -> Policy:
        binaries = [self.cfg.binary] + ([self.cfg.mpi.binary] if self.cfg.mpi else [])
        return Policy(src=self.src, run_dir=self.run_dir, max_experiments=limits.max_experiments,
                      binaries=frozenset(Path(b).name for b in binaries),
                      hidden=tuple(Path(h).resolve() for h in self.hidden))


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def system_prompt(limits: Limits) -> str:
    return load_prompt("system", max_experiments=limits.max_experiments, max_turns=limits.max_turns,
                       timeout_minutes=f"{limits.timeout_s / 60:g}")


def task_prompt(run: Run, limits: Limits, preflight: dict, goal: str | None) -> str:
    cfg, variant = run.cfg, run.variant
    host = preflight.get("host", {})
    pre = [f"  CPU: {host.get('cpu_model')} ({host.get('cpus')} logical CPUs), {host.get('arch')}, "
           f"kernel {host.get('kernel')}",
           f"  perf: {preflight.get('perf', {}).get('version')}; "
           f"perf_event_paranoid={preflight.get('perf_event_paranoid')}",
           "  events: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in preflight.get("events", {}).items()),
           *[f"  warning: {w}" for w in preflight.get("warnings", [])]]
    mpi_note = (f"This is an MPI analysis with {cfg.mpi.ranks} ranks via '{cfg.mpi.launcher}'. The "
                "default run command is written for that rank count. Each rank has its own perf.data "
                "and perf-report.txt under rank-<r>/." if run.mpi else "")
    return load_prompt(
        "task", target=cfg.target, variant="MPI" if run.mpi else "serial", run_dir=run.run_dir, src=run.src,
        repo_url=cfg.repo_url, commit=cfg.commit,
        build_info=run.work / f"build-info{'-mpi' if run.mpi else ''}.json",
        run_command=variant.run.format(binary=run.work / variant.binary),
        environment=" ".join(f"{k}={v}" for k, v in run.env().items() if k != "PYTHON") or "(none)",
        preflight_path=run.preflight, preflight="\n".join(pre), mpi_note=mpi_note,
        schema=SCHEMA_PATH, max_experiments=limits.max_experiments, max_turns=limits.max_turns,
        timeout_minutes=f"{limits.timeout_s / 60:g}", goal=f"\nAdditional focus: {goal}" if goal else "",
        **run.commands(),
    )


class Transcript:
    """Append-only JSON Lines record of everything sent to and received from the model."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("a")

    def write(self, kind: str, **data):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **data}
        self.file.write(json.dumps(record, default=str) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class AgentError(Exception):
    """The model API failed in a way retrying does not fix."""


class BudgetExhausted(AgentError):
    """The API account's spending budget is used up: no call will succeed until it is renewed."""


RATE_LIMIT_WAITS = (15, 30, 60, 60)  # seconds; HokieAI's token limits renew every 60 s
sleep = time.sleep  # replaced in tests


def complete(client, limits: Limits, messages: list, tools: Tools, transcript: Transcript):
    """One model call. A 429 (HokieAI token or request limit) is waited out, within the wall clock."""
    for attempt, wait in enumerate((*RATE_LIMIT_WAITS, None)):
        remaining = tools.remaining()
        if remaining <= 0:
            raise DeadlineExceeded()
        try:
            return client.with_options(timeout=remaining).chat.completions.create(
                model=limits.model, messages=messages, tools=SCHEMAS)
        except openai.RateLimitError as e:
            if "UserBudgetExceeded" in str(e.body or e.message):
                raise BudgetExhausted(f"the HokieAI budget for this account is used up (HTTP 429: "
                                      f"{e.body or e.message}); waiting does not help until it is renewed") from None
            if wait is None:
                raise AgentError(f"HokieAI kept refusing with HTTP 429 after {len(RATE_LIMIT_WAITS)} waits: "
                                 f"{e.body or e.message}") from None
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            if retry_after and retry_after.replace(".", "", 1).isdigit():
                wait = max(float(retry_after), 1)
            if tools.remaining() <= wait:
                raise DeadlineExceeded() from None
            print(f"agent: HokieAI rate limit (HTTP 429: {e.body or e.message}); waiting {wait:g} s", file=sys.stderr)
            transcript.write("rate_limited", wait_s=wait, error=str(e.body or e.message))
            sleep(wait)
        except openai.APITimeoutError:
            if tools.remaining() <= 0:
                raise DeadlineExceeded() from None
            raise AgentError("the model API timed out") from None
        except openai.APIStatusError as e:
            raise AgentError(f"the model API refused the request (HTTP {e.status_code}): {e.body or e.message}") from None
        except openai.APIConnectionError as e:
            raise AgentError(f"could not reach the model API: {e}") from None


def agent_loop(client, limits: Limits, messages: list, tools: Tools, transcript: Transcript, attempt: str) -> str:
    """Call the model and run its tool calls until it answers without tools ("done") or runs out of turns."""
    for turn in range(1, limits.max_turns + 1):
        response = complete(client, limits, messages, tools, transcript)
        transcript.write("response", attempt=attempt, turn=turn, response=response.model_dump(exclude_none=True))
        message = response.choices[0].message
        calls = message.tool_calls or []
        entry = {"role": "assistant", "content": message.content or ""}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in calls]
        messages.append(entry)
        if message.content:
            print(message.content, file=sys.stderr)
        if not calls:
            return "done"
        for call in calls:
            print(f"[{call.function.name}] {_summary(call.function.arguments)}", file=sys.stderr)
            result = tools.call(call.function.name, call.function.arguments)
            transcript.write("tool", attempt=attempt, turn=turn, id=call.id, name=call.function.name,
                             arguments=call.function.arguments, result=result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    transcript.write("max_turns", attempt=attempt, turns=limits.max_turns)
    print(f"agent: {attempt} stopped after the turn limit ({limits.max_turns})", file=sys.stderr)
    return "max_turns"


def _summary(arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments[:200]
    if isinstance(args, dict):
        text = args.get("command") or args.get("path") or args.get("pattern") or ""
        if "content" in args:
            text = f"findings.json ({len(args['content'])} characters)"
        return str(text)[:300]
    return arguments[:200]


def findings_errors(run_dir: Path) -> list[str]:
    try:
        load_findings(run_dir)
        return []
    except FindingsError as e:
        return e.errors


def session(client, run: Run, limits: Limits, task: str, transcript: Transcript, deadline: float) -> list[str]:
    """One agent conversation: the analysis, then at most one retry if findings.json is invalid."""
    tools = Tools(policy=run.policy(limits), env=run.env(), deadline=deadline)
    messages = [{"role": "system", "content": system_prompt(limits)}, {"role": "user", "content": task}]
    transcript.write("prompt", attempt="analysis", text=task)
    errors: list[str] = []
    for attempt in ("analysis", "retry"):
        if attempt == "retry":
            retry = load_prompt("retry", errors="\n".join(f"- {e}" for e in errors),
                                findings=run.run_dir / "findings.json",
                                validate_command=run.commands()["validate_command"])
            messages.append({"role": "user", "content": retry})
            transcript.write("prompt", attempt="retry", text=retry)
        agent_loop(client, limits, messages, tools, transcript, attempt)
        errors = findings_errors(run.run_dir)
        transcript.write("validation", attempt=attempt, errors=errors)
        if not errors:
            return []
        if attempt == "analysis":
            print(f"findings.json is invalid ({len(errors)} problem(s)); asking the agent to fix it once.",
                  file=sys.stderr)
    return errors


def run(cfg: TargetConfig, config_path: Path, root: Path, run_id: str, *, client, label: str | None = None,
        mpi: bool = False, limits: Limits = Limits(), goal: str | None = None, base_url: str = DEFAULT_BASE_URL,
        hidden: tuple[Path, ...] = ()) -> int:
    r = Run(cfg=cfg, config_path=Path(config_path), root=Path(root).resolve(), run_id=run_id, mpi=mpi, label=label,
            hidden=hidden)
    # Preflight is the only step before the agent: it is cheap and stops a run that cannot profile.
    if run_script("preflight.sh", "--output", str(r.preflight), cwd=r.root):
        print(f"error: preflight failed; see {r.preflight}", file=sys.stderr)
        return FAILED
    preflight = json.loads(r.preflight.read_text())
    r.run_dir.mkdir(parents=True)
    r.work.mkdir(parents=True, exist_ok=True)

    transcript = Transcript(r.run_dir / "transcript.jsonl")
    task = task_prompt(r, limits, preflight, goal)
    transcript.write("start", run_id=run_id, target=cfg.target, mpi=mpi, limits=dataclasses.asdict(limits),
                     base_url=base_url, tools=[t["function"]["name"] for t in SCHEMAS],
                     system_prompt=system_prompt(limits))
    deadline = time.monotonic() + limits.timeout_s
    status = OK
    try:
        errors = session(client, r, limits, task, transcript, deadline)
    except DeadlineExceeded:
        print(f"error: wall-clock limit of {limits.timeout_s / 60:g} minutes reached; the agent was stopped",
              file=sys.stderr)
        transcript.write("timeout", seconds=limits.timeout_s)
        errors, status = findings_errors(r.run_dir), TIMED_OUT
    except AgentError as e:
        print(f"error: {e}", file=sys.stderr)
        transcript.write("error", error=str(e))
        errors, status = findings_errors(r.run_dir), BUDGET_EXHAUSTED if isinstance(e, BudgetExhausted) else FAILED

    (r.run_dir / "report.md").unlink(missing_ok=True)  # the report is always generated, never the agent's
    if errors:
        print(f"error: {r.run_dir / 'findings.json'} is "
              f"{'still invalid after one retry' if status == OK else 'missing or invalid'}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        status = status or INVALID_FINDINGS
    else:
        print(write_report(r.run_dir).read_text())

    if problems := source_changes(r.src, cfg.commit):
        print("error: application source was modified:\n  " + "\n  ".join(problems), file=sys.stderr)
        status = SOURCE_MODIFIED
    transcript.write("end", status=status, findings_errors=errors, source_problems=problems)
    transcript.close()
    print(f"run directory: {r.run_dir} (transcript: transcript.jsonl)", file=sys.stderr)
    return status


def add_limit_options(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("limits")
    group.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"HokieAI model (default {DEFAULT_MODEL}; your subscription decides which are allowed)")
    group.add_argument("--max-turns", type=int, default=Limits.max_turns,
                       help=f"model calls for the analysis and for the retry (default {Limits.max_turns})")
    group.add_argument("--timeout", default="60m", metavar="DURATION",
                       help="wall-clock limit for the agent, e.g. 45m or 2h (default 60m; per case for eval)")
    group.add_argument("--max-experiments", type=int, default=Limits.max_experiments,
                       help=f"experiment runs the agent may make (default {Limits.max_experiments})")


def limits_from(args, parser) -> Limits:
    try:
        return Limits(model=args.model, max_turns=args.max_turns, timeout_s=parse_duration(args.timeout),
                      max_experiments=args.max_experiments)
    except ValueError as e:
        parser.error(str(e))


def client_from(parser) -> tuple[openai.OpenAI, str]:
    key, base_url = hokieai_settings(Path.cwd())
    if not key:
        parser.error(f"{KEY_VAR} is not set: put your HokieAI key in .env ({KEY_VAR}=...) or the environment")
    return make_client(key, base_url), base_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bottleneck-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    analyze = commands.add_parser("analyze", help="profile a target with perf and report its bottlenecks",
                                  description="Run preflight checks, then let the agent build, profile and "
                                              "analyze the target and write findings.json.")
    analyze.add_argument("--target", required=True, metavar="{" + ",".join(TARGETS) + "}",
                         help="proxy application to analyze")
    analyze.add_argument("--config", required=True, type=Path, metavar="PATH",
                         help="target config YAML, e.g. configs/lulesh.yaml")
    analyze.add_argument("--run-id", help="name of the run directory under runs/ (default: <target>-<timestamp>)")
    analyze.add_argument("--label", help="free-form note stored in the baseline's metadata.json")
    analyze.add_argument("--mpi", action="store_true",
                         help="build and profile the config's MPI variant with its default rank count")
    analyze.add_argument("--goal", help="extra focus for the analysis, e.g. 'memory behaviour'")
    add_limit_options(analyze)
    evaluate = commands.add_parser("eval", help="measure diagnosis accuracy on a suite of injected bottlenecks",
                                   description="Run the agent on every case of a suite (fresh source copy each) "
                                               "and score its findings; writes results.json and results.md.")
    evaluate.add_argument("--suite", required=True, type=Path, metavar="PATH", help="suite YAML, e.g. suites/comd.yaml")
    evaluate.add_argument("--out", type=Path, metavar="DIR",
                          help="output directory (default: evals/<suite>-<timestamp>)")
    evaluate.add_argument("--cases", metavar="ID,ID", help="run only these case ids")
    evaluate.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                          help="set an environment variable for every case, e.g. OMP_NUM_THREADS=18 (repeatable)")
    add_limit_options(evaluate)
    args = parser.parse_args(argv)

    if args.command == "eval":
        from .evaluate import SuiteError, load_suite, run_suite
        try:
            suite = load_suite(args.suite)
        except SuiteError as e:
            evaluate.error(str(e))
        only = [c.strip() for c in args.cases.split(",")] if args.cases else None
        if only and (unknown := set(only) - {c.id for c in suite.cases}):
            evaluate.error(f"unknown case id(s): {', '.join(sorted(unknown))} "
                           f"(suite has {', '.join(c.id for c in suite.cases)})")
        limits = limits_from(args, evaluate)
        env = {}
        for item in args.env:
            name, sep, value = item.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_]\w*", name) or name.startswith("SAGE_"):
                evaluate.error(f"--env needs NAME=VALUE with a valid name (not SAGE_*), got {item!r}")
            env[name] = value
        out = args.out or Path("evals") / f"{suite.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        if out.exists():
            evaluate.error(f"{out} already exists; choose another --out")
        client, base_url = client_from(evaluate)
        results = run_suite(suite, out.resolve(), Path.cwd(), client, limits, base_url, only, env)
        s = results["summary"]
        print(f"\n{s['correct']} of {s['cases']} cases correct; results: {out / 'results.md'}")
        if results.get("not_run"):
            return BUDGET_EXHAUSTED
        return OK if s["errors"] == 0 else FAILED

    # Validate everything up front, before running anything or calling the API.
    if args.target not in TARGETS:
        analyze.error(f"invalid target {args.target!r} (choose from {', '.join(TARGETS)})")
    try:
        cfg = load_config(args.config, args.target)
    except ConfigError as e:
        analyze.error(str(e))
    limits = limits_from(args, analyze)
    if args.mpi and cfg.mpi is None:
        analyze.error(f"--mpi needs an mpi section in {args.config}")
    run_id = args.run_id or f"{args.target}{'-mpi' if args.mpi else ''}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not RUN_ID.fullmatch(run_id):
        analyze.error(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    if (Path.cwd() / "runs" / run_id).exists():
        analyze.error(f"runs/{run_id} already exists; choose another --run-id")
    client, base_url = client_from(analyze)
    return run(cfg, args.config, Path.cwd(), run_id, client=client, label=args.label,
               mpi=args.mpi, limits=limits, goal=args.goal, base_url=base_url)


if __name__ == "__main__":
    sys.exit(main())
