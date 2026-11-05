"""bottleneck-agent: a single Claude agent that finds performance bottlenecks with Linux perf."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultError,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from .config import TARGETS, ConfigError, TargetConfig, load_config
from .findings import SCHEMA_PATH, FindingsError, load as load_findings, write_report
from .guard import Policy, make_guard, source_changes

TOOLS = ["Bash", "Read", "Glob", "Grep", "Write", "Edit"]  # Write/Edit are limited to findings.json by the hook
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PROMPTS = Path(__file__).resolve().parent / "prompts"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # same rule as scripts/profile_target.sh
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# Exit codes (also in the README).
OK, FAILED, USAGE, SOURCE_MODIFIED, INVALID_FINDINGS, TIMED_OUT = 0, 1, 2, 3, 4, 5


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
    model: str | None = None  # None: the SDK's default model
    max_turns: int = 80  # per agent turn (the analysis, and the retry)
    timeout_s: float = 3600  # wall-clock for the whole agent session, retry included
    max_experiments: int = 1  # experiment artifact sets under <run>/experiments

    def __post_init__(self):
        if self.model is not None and not self.model.strip():
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
                                  f"--cmd '{{binary}} <arguments>'",
            "annotate_experiment_command": f"{SCRIPTS}/annotate_symbol.sh <name> <symbol>{rank} "
                                           f"--runs {self.run_dir / 'experiments'}",
            "validate_command": f"{sys.executable} -m sage.findings validate {self.run_dir}",
        }

    def env(self) -> dict[str, str]:
        return {**self.cfg.env, **(self.cfg.mpi.env if self.mpi else {}), "PYTHON": sys.executable}


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


def build_options(run: Run, limits: Limits) -> ClaudeAgentOptions:
    binaries = [run.cfg.binary] + ([run.cfg.mpi.binary] if run.cfg.mpi else [])
    policy = Policy(src=run.src, run_dir=run.run_dir, max_experiments=limits.max_experiments,
                    binaries=frozenset(Path(b).name for b in binaries))
    return ClaudeAgentOptions(
        tools=TOOLS,  # the agent's whole toolset: no subagents, no web, no MCP servers
        allowed_tools=TOOLS,
        permission_mode="dontAsk",  # anything not pre-approved is denied, never prompted
        system_prompt=system_prompt(limits),
        cwd=str(run.run_dir),
        add_dirs=[str(run.work)],
        env=run.env(),
        setting_sources=[],  # ignore user/project settings, hooks and CLAUDE.md files
        hooks={"PreToolUse": [HookMatcher(hooks=[make_guard(policy)])]},
        model=limits.model,
        max_turns=limits.max_turns,
    )


class Transcript:
    """Append-only JSON Lines record of everything sent to and received from the agent."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("a")

    def write(self, kind: str, **data):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **data}
        self.file.write(json.dumps(record, default=str) + "\n")
        self.file.flush()

    def message(self, message):
        data = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else repr(message)
        self.write("message", type=type(message).__name__, message=data)

    def close(self):
        self.file.close()


async def agent_turn(client, prompt: str, transcript: Transcript) -> ResultMessage | None:
    transcript.write("prompt", text=prompt)
    await client.query(prompt)
    result = None
    async for message in client.receive_response():
        transcript.message(message)
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, file=sys.stderr)
                elif isinstance(block, ToolUseBlock):
                    detail = block.input.get("command") or block.input.get("file_path") or block.input.get("pattern", "")
                    print(f"[{block.name}] {detail}", file=sys.stderr)
        elif isinstance(message, ResultMessage):
            result = message
            if message.is_error:
                print(f"agent: turn ended with an error ({message.subtype}): {message.errors or message.result}",
                      file=sys.stderr)
    return result


def findings_errors(run_dir: Path) -> list[str]:
    try:
        load_findings(run_dir)
        return []
    except FindingsError as e:
        return e.errors


async def session(run: Run, limits: Limits, task: str, transcript: Transcript) -> list[str]:
    """One agent session: the analysis, then at most one retry if findings.json is invalid."""
    errors: list[str] = []
    async with ClaudeSDKClient(options=build_options(run, limits)) as client:
        for attempt in ("analysis", "retry"):
            prompt = task if attempt == "analysis" else load_prompt(
                "retry", errors="\n".join(f"- {e}" for e in errors),
                findings=run.run_dir / "findings.json", validate_command=run.commands()["validate_command"])
            try:
                await agent_turn(client, prompt, transcript)
            except ResultError as e:  # the turn ended with an error result, e.g. max turns
                transcript.write("error", attempt=attempt, error=str(e), subtype=e.subtype)
                print(f"agent: {attempt} ended with an error ({e.terminal_reason or e.subtype}): {e}", file=sys.stderr)
            errors = findings_errors(run.run_dir)
            transcript.write("validation", attempt=attempt, errors=errors)
            if not errors:
                return []
            if attempt == "analysis":
                print(f"findings.json is invalid ({len(errors)} problem(s)); asking the agent to fix it once.",
                      file=sys.stderr)
    return errors


async def run(cfg: TargetConfig, config_path: Path, root: Path, run_id: str, *, label: str | None = None,
              mpi: bool = False, limits: Limits = Limits(), goal: str | None = None) -> int:
    r = Run(cfg=cfg, config_path=Path(config_path), root=Path(root).resolve(), run_id=run_id, mpi=mpi, label=label)
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
                     tools=TOOLS, system_prompt=system_prompt(limits))
    status = OK
    try:
        errors = await asyncio.wait_for(session(r, limits, task, transcript), timeout=limits.timeout_s)
    except asyncio.TimeoutError:
        print(f"error: wall-clock limit of {limits.timeout_s / 60:g} minutes reached; the agent was stopped",
              file=sys.stderr)
        transcript.write("timeout", seconds=limits.timeout_s)
        errors, status = findings_errors(r.run_dir), TIMED_OUT

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
    group = analyze.add_argument_group("limits")
    group.add_argument("--model", help="Claude model (default: the SDK's default)")
    group.add_argument("--max-turns", type=int, default=Limits.max_turns,
                       help=f"agent turns for the analysis and for the retry (default {Limits.max_turns})")
    group.add_argument("--timeout", default="60m", metavar="DURATION",
                       help="wall-clock limit for the agent, e.g. 45m or 2h (default 60m)")
    group.add_argument("--max-experiments", type=int, default=Limits.max_experiments,
                       help=f"experiment runs the agent may make (default {Limits.max_experiments})")
    args = parser.parse_args(argv)

    # Validate everything up front, before running anything or calling the API.
    if args.target not in TARGETS:
        analyze.error(f"invalid target {args.target!r} (choose from {', '.join(TARGETS)})")
    try:
        cfg = load_config(args.config, args.target)
    except ConfigError as e:
        analyze.error(str(e))
    try:
        limits = Limits(model=args.model, max_turns=args.max_turns, timeout_s=parse_duration(args.timeout),
                        max_experiments=args.max_experiments)
    except ValueError as e:
        analyze.error(str(e))
    if args.mpi and cfg.mpi is None:
        analyze.error(f"--mpi needs an mpi section in {args.config}")
    run_id = args.run_id or f"{args.target}{'-mpi' if args.mpi else ''}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not RUN_ID.fullmatch(run_id):
        analyze.error(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    if (Path.cwd() / "runs" / run_id).exists():
        analyze.error(f"runs/{run_id} already exists; choose another --run-id")
    return asyncio.run(run(cfg, args.config, Path.cwd(), run_id, label=args.label, mpi=args.mpi,
                           limits=limits, goal=args.goal))


if __name__ == "__main__":
    sys.exit(main())
