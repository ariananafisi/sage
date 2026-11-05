"""bottleneck-agent: a single Claude agent that finds performance bottlenecks with Linux perf."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
- MPI may be unavailable: use serial / single-rank runs.

Method:
1. Preflight checks have already run; their results are in the task. If hardware
   counters are unavailable, use software events (task-clock, cpu-clock) and say
   so; if topdown metrics are unsupported, do not attempt them.
2. The binary is already built (optimized, -g, frame pointers). Start from the
   default run command; adjust the problem size only if a run is far outside
   roughly 5-30 seconds. Record the exact command and the app's own timing output.
3. Baseline with `perf stat` (cycles, instructions, IPC, cache and branch
   misses, as available).
4. Sample with `perf record -g` (try `--call-graph dwarf` if frame-pointer
   stacks look broken) and inspect `perf report --stdio --no-children` and
   `perf annotate --stdio` for the top functions.
5. For each hotspot, read the relevant source and classify the bottleneck
   (compute-bound, memory-bandwidth/latency-bound, branch mispredictions,
   poor vectorization, load imbalance / serial sections, etc.), citing the perf
   evidence. Use targeted `perf stat -e` runs to confirm or reject hypotheses.

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


def run_script(name: str, *args: str, cwd: Path) -> int:
    """Run scripts/<name>; its progress and errors go straight to the user's terminal."""
    env = {**os.environ, "PYTHON": sys.executable}
    return subprocess.run(["bash", str(SCRIPTS / name), *args], cwd=cwd, env=env).returncode


def build_prompt(cfg: TargetConfig, work: Path, build_info: dict, preflight: dict, goal: str | None) -> str:
    env = " ".join(f"{k}={v}" for k, v in cfg.env.items()) or "(none)"
    events = preflight.get("events", {})
    lines = [
        f"Profile {cfg.target.upper()} with perf and report its performance bottlenecks.",
        "",
        f"Source checkout (read-only): {work / 'src'}  ({build_info['repo_url']} @ {build_info['commit']})",
        f"Work directory (builds): {work}",
        f"Built binary: {build_info['binary']}",
        f"Compiler: {build_info['compiler_version']}; flags: {build_info['flags']}",
        "Build commands used (run from the work directory, build log in build.log):",
        *[f"  $ {c}" for c in build_info["build_commands"]],
        f"Default run command: {cfg.run_command(work)}",
        f"Environment (already set for your commands): {env}",
        "",
        "Preflight results (from scripts/preflight.sh; no need to repeat these checks):",
        f"  CPU: {preflight.get('host', {}).get('cpu_model')} ({preflight.get('host', {}).get('cpus')} CPUs), "
        f"{preflight.get('perf', {}).get('version')}, perf_event_paranoid={preflight.get('perf_event_paranoid')}",
        "  events: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in events.items()),
        *[f"  warning: {w}" for w in preflight.get("warnings", [])],
    ]
    if goal:
        lines += ["", f"Additional focus: {goal}"]
    return "\n".join(lines)


def build_options(cfg: TargetConfig, work: Path, run_dir: Path, protected: set[str],
                  model: str | None, max_turns: int) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=TOOLS,  # the only built-in tools the agent can see: no Write/Edit, no subagents
        allowed_tools=TOOLS,
        permission_mode="dontAsk",  # anything not pre-approved is denied, never prompted
        system_prompt=SYSTEM_PROMPT + "\n" + APP_HINTS[cfg.target],
        cwd=str(run_dir),
        add_dirs=[str(work)],
        env=dict(cfg.env),
        setting_sources=[],  # ignore user/project settings, hooks and CLAUDE.md files
        hooks={"PreToolUse": [HookMatcher(hooks=[make_guard(protected)])]},
        model=model,
        max_turns=max_turns,
    )


async def run(cfg: TargetConfig, config_path: Path, root: Path, model: str | None = None,
              max_turns: int = 80, goal: str | None = None) -> int:
    root = root.resolve()
    work_root = root / ".work"
    work = work_root / cfg.target
    preflight_path = work_root / "preflight.json"
    if run_script("preflight.sh", "--output", str(preflight_path), cwd=root):
        print(f"error: preflight failed; see {preflight_path}", file=sys.stderr)
        return 1
    if run_script("build_target.sh", cfg.target, "--config", str(Path(config_path).resolve()),
                  "--work", str(work_root), cwd=root):
        print("error: build failed (see the log path above)", file=sys.stderr)
        return 1
    preflight = json.loads(preflight_path.read_text())
    build_info = json.loads((work / "build-info.json").read_text())
    source = snapshot(work / "src")

    run_dir = root / "runs" / f"{cfg.target}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    options = build_options(cfg, work, run_dir, set(source), model, max_turns)

    report, status = None, 0
    try:
        async for message in query(prompt=build_prompt(cfg, work, build_info, preflight, goal), options=options):
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
    return asyncio.run(run(cfg, args.config, Path.cwd(), args.model, args.max_turns, args.goal))


if __name__ == "__main__":
    sys.exit(main())
