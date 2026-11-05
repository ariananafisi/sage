"""bottleneck-agent: a single Claude agent that finds performance bottlenecks with Linux perf."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultError,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from .config import TARGETS, ConfigError, TargetConfig, load_config
from .guard import changed, make_guard, snapshot

TOOLS = ["Bash", "Read", "Glob", "Grep"]

SYSTEM_PROMPT = """\
You are SAGE, a performance engineer. Your job is to find and explain the
performance bottlenecks of an HPC proxy application using Linux perf.

Rules (enforced; violating commands are denied):
- perf is the only profiler. Do not use gprof, valgrind, VTune, strace, -pg, etc.
- Never modify the application's source checkout. If you need a different build,
  configure it only through command-line variables (e.g. `make VAR=value`,
  `cmake -DVAR=value`) into a new build directory under the work directory.
- Write perf data, logs and notes in the current directory (the run directory).

Method:
1. Preflight checks have already run; their results are in the task. If hardware
   counters are unavailable, use software events (task-clock, cpu-clock) and say
   so; if topdown metrics are unsupported, do not attempt them.
2. The binary is already built (optimized, -g, frame pointers), and a baseline
   artifact set already exists in the current directory: metadata.json (command,
   timings, CPU, threads, parsed perf summary), perf-stat.txt (perf stat -d, 3
   repeats), perf.data and perf-report.txt (self time by DSO and symbol; one
   rank-<r>/ directory per rank for MPI), stdout.log and stderr.log. Start from
   these; do not re-run the baseline.
3. For each top hotspot, save its annotation with the annotate_symbol.sh command
   given in the task, then read it next to the source.
4. To test a hypothesis, make a new artifact set with the profile_target.sh
   command given in the task (it writes under experiments/), e.g. a different
   problem size or thread count via --cmd, or run targeted `perf stat -e` /
   `perf record` commands writing into the current directory.
5. For each hotspot, classify the bottleneck (compute-bound, memory-bandwidth/
   latency-bound, branch mispredictions, poor vectorization, load imbalance /
   serial sections, MPI communication, etc.), citing the perf evidence. For MPI
   runs, compare ranks and time spent in MPI libraries.

Finish with a Markdown report and nothing else, containing:
## Environment
## Build and run (exact commands)
## Baseline metrics (perf stat table)
## Hotspots (table: function, file:line, % of samples)
## Bottleneck analysis (one subsection per hotspot: evidence -> diagnosis)
## Optimization opportunities (suggestions only; nothing was changed)
## Limitations
"""

APP_HINTS = {
    "lulesh": (
        "The application is LULESH 2.0 (C++, Lagrangian shock hydrodynamics, OpenMP). "
        "Run the binary with -h for options: -s elements per edge, -i iterations. "
        "Threads are controlled by OMP_NUM_THREADS."
    ),
    "comd": (
        "The application is CoMD (C, classical molecular dynamics). Run the binary with "
        "--help for options: -x/-y/-z unit cells, -N steps. The default potential is "
        "Lennard-Jones; -e selects EAM and needs --potDir <checkout>/pots."
    ),
}


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # same rule as scripts/profile_target.sh


def run_script(name: str, *args: str, cwd: Path) -> int:
    """Run scripts/<name>; its progress and errors go straight to the user's terminal."""
    env = {**os.environ, "PYTHON": sys.executable}
    return subprocess.run(["bash", str(SCRIPTS / name), *args], cwd=cwd, env=env).returncode


def build_prompt(cfg: TargetConfig, work: Path, run_dir: Path, config_path: Path,
                 metadata: dict, preflight: dict, goal: str | None) -> str:
    build, perf, wall = metadata["build"] or {}, metadata["perf"], metadata["wall_clock"]
    mpi = metadata["mpi"]
    events = preflight.get("events", {})
    stat = perf.get("stat") or {}
    counted = [e["event"] for e in stat.get("events", []) if e["counted"]]
    missing = [e["event"] for e in stat.get("events", []) if not e["counted"]]
    lines = [
        f"Profile {cfg.target.upper()} with perf and report its performance bottlenecks.",
        "",
        f"Source checkout (read-only): {work / 'src'}  ({build.get('repo_url')} @ {build.get('commit')})",
        f"Built binary: {build.get('binary')} ({build.get('variant')} build; compiler {build.get('compiler')}: "
        f"{build.get('compiler_version')}; flags: {build.get('flags')})",
        f"Run command: {metadata['run_command']}" + (f"  (launched with: {mpi['launcher']})" if mpi else ""),
        f"Environment (already set for your commands): "
        + (" ".join(f"{k}={v}" for k, v in metadata["threads"]["variables"].items()) or "(no thread variables)"),
        "",
        "Preflight results (from scripts/preflight.sh; no need to repeat these checks):",
        f"  CPU: {preflight.get('host', {}).get('cpu_model')} ({preflight.get('host', {}).get('cpus')} CPUs), "
        f"{preflight.get('perf', {}).get('version')}, perf_event_paranoid={preflight.get('perf_event_paranoid')}",
        "  events: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in events.items()),
        *[f"  warning: {w}" for w in preflight.get("warnings", [])],
        "",
        f"Baseline artifact set (run id {metadata['run_id']}): {run_dir}",
        f"  app wall-clock: {wall['app_mean_s']} s mean over {wall['repeats']} runs (+- {wall['app_variance_pct']}%)",
        f"  perf stat counted: {', '.join(counted) or 'nothing'}; not available: {', '.join(missing) or 'none'}",
    ]
    reports = perf["rank_reports"] or {"(all)": perf["report"] or {}}
    for rank, report in sorted(reports.items(), key=lambda kv: kv[0]):
        top = ", ".join(f"{r['symbol']} ({r['dso']}) {r['self_pct']}%" for r in report.get("top", [])[:5])
        lines.append(f"  top self time{'' if rank == '(all)' else f' rank {rank}'} "
                     f"[{report.get('samples')} samples of {report.get('event')}]: {top}")
    scripts = SCRIPTS
    rank_opt = " --rank <r>" if mpi else ""
    lines += [
        "",
        "Commands available to you:",
        f"  annotate a symbol:  {scripts}/annotate_symbol.sh {metadata['run_id']} <symbol>{rank_opt} --runs {run_dir.parent}",
        f"  new artifact set:   {scripts}/profile_target.sh {cfg.target} <name> --runs {run_dir}/experiments "
        f"--work {work.parent} --config {Path(config_path).resolve()}"
        + (f" --ranks {mpi['ranks']}" if mpi else "") + " [--label TEXT] [--cmd '{binary} <args>']",
    ]
    if goal:
        lines += ["", f"Additional focus: {goal}"]
    return "\n".join(lines)


def build_options(cfg: TargetConfig, work: Path, run_dir: Path, protected: set[str],
                  model: str | None, max_turns: int, mpi: bool = False) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=TOOLS,  # the only built-in tools the agent can see: no Write/Edit, no subagents
        allowed_tools=TOOLS,
        permission_mode="dontAsk",  # anything not pre-approved is denied, never prompted
        system_prompt=SYSTEM_PROMPT + "\n" + APP_HINTS[cfg.target],
        cwd=str(run_dir),
        add_dirs=[str(work)],
        env={**cfg.env, **(cfg.mpi.env if mpi and cfg.mpi else {})},
        setting_sources=[],  # ignore user/project settings, hooks and CLAUDE.md files
        hooks={"PreToolUse": [HookMatcher(hooks=[make_guard(protected)])]},
        model=model,
        max_turns=max_turns,
    )


async def run(cfg: TargetConfig, config_path: Path, root: Path, run_id: str, *, label: str | None = None,
              mpi: bool = False, model: str | None = None, max_turns: int = 80, goal: str | None = None) -> int:
    root = root.resolve()
    work_root = root / ".work"
    work = work_root / cfg.target
    run_dir = root / "runs" / run_id
    config_arg = str(Path(config_path).resolve())
    preflight_path = work_root / "preflight.json"
    if run_script("preflight.sh", "--output", str(preflight_path), cwd=root):
        print(f"error: preflight failed; see {preflight_path}", file=sys.stderr)
        return 1
    if run_script("build_target.sh", cfg.target, *(["--mpi"] if mpi else []), "--config", config_arg,
                  "--work", str(work_root), cwd=root):
        print("error: build failed (see the log path above)", file=sys.stderr)
        return 1
    source = snapshot(work / "src")
    profile_args = [cfg.target, run_id, "--config", config_arg, "--work", str(work_root),
                    "--runs", str(root / "runs")]
    if label:
        profile_args += ["--label", label]
    if mpi:
        profile_args += ["--ranks", str(cfg.mpi.ranks)]
    if run_script("profile_target.sh", *profile_args, cwd=root):
        print(f"error: baseline profiling failed; see {run_dir}", file=sys.stderr)
        return 1
    preflight = json.loads(preflight_path.read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    options = build_options(cfg, work, run_dir, set(source), model, max_turns, mpi)
    prompt = build_prompt(cfg, work, run_dir, config_path, metadata, preflight, goal)

    report, status = None, 0
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, file=sys.stderr)
                    elif isinstance(block, ToolUseBlock):
                        detail = block.input.get("command") or block.input.get("file_path") or block.input.get("pattern", "")
                        print(f"[{block.name}] {detail}", file=sys.stderr)
            elif isinstance(message, ResultMessage):
                report = message.result
                if message.is_error:
                    status = 1
    except ResultError as e:
        print(f"error: agent run failed ({e.terminal_reason or e.subtype}): {e}", file=sys.stderr)
        status = 1

    if report:
        (run_dir / "report.md").write_text(report)
        print(report)
    print(f"run directory: {run_dir}", file=sys.stderr)

    violations = changed(source)
    if violations:
        print("error: application source was modified:\n  " + "\n  ".join(violations), file=sys.stderr)
        return 3
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bottleneck-agent", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    analyze = commands.add_parser("analyze", help="build a target and profile it with perf",
                                  description="Run preflight checks, check out and build a target, "
                                              "then let the agent profile it with perf.")
    analyze.add_argument("--target", required=True, metavar="{" + ",".join(TARGETS) + "}",
                         help="proxy application to analyze")
    analyze.add_argument("--config", required=True, type=Path, metavar="PATH",
                         help="target config YAML, e.g. configs/lulesh.yaml")
    analyze.add_argument("--run-id", help="name of the run directory under runs/ (default: <target>-<timestamp>)")
    analyze.add_argument("--label", help="free-form note stored in the run's metadata.json")
    analyze.add_argument("--mpi", action="store_true",
                         help="build and profile the config's MPI variant with its default rank count")
    analyze.add_argument("--model", help="Claude model to use (default: SDK default)")
    analyze.add_argument("--max-turns", type=int, default=80)
    analyze.add_argument("--goal", help="extra focus for the analysis, e.g. 'memory behaviour'")
    args = parser.parse_args(argv)

    # Validate everything up front, before cloning, building or calling the API.
    if args.target not in TARGETS:
        analyze.error(f"invalid target {args.target!r} (choose from {', '.join(TARGETS)})")
    try:
        cfg = load_config(args.config, args.target)
    except ConfigError as e:
        analyze.error(str(e))
    if args.mpi and cfg.mpi is None:
        analyze.error(f"--mpi needs an mpi section in {args.config}")
    run_id = args.run_id or f"{args.target}{'-mpi' if args.mpi else ''}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not RUN_ID.fullmatch(run_id):
        analyze.error(f"invalid run id {run_id!r}: use letters, digits, '.', '_' or '-'")
    if (Path.cwd() / "runs" / run_id).exists():
        analyze.error(f"runs/{run_id} already exists; choose another --run-id")
    return asyncio.run(run(cfg, args.config, Path.cwd(), run_id, label=args.label, mpi=args.mpi,
                           model=args.model, max_turns=args.max_turns, goal=args.goal))


if __name__ == "__main__":
    sys.exit(main())
