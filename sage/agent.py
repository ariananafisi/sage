"""bottleneck-agent: a single Claude agent that finds performance bottlenecks with Linux perf."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
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
1. Check the environment: `perf --version`, /proc/sys/kernel/perf_event_paranoid,
   CPU model, core count, and whether hardware events work
   (`perf stat -e cycles,instructions true`). If hardware counters are
   unavailable, fall back to software events (task-clock, cpu-clock) and say so.
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


class SetupError(Exception):
    """Checkout or build failed; the agent was not started."""


def git(*args: str, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise SetupError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def checkout(cfg: TargetConfig, src: Path) -> None:
    """Clone ``cfg.repo_url`` into ``src`` (or reuse it) and check out the pinned commit."""
    if not (src / ".git").exists():
        if src.exists() and any(src.iterdir()):
            raise SetupError(f"{src} exists but is not a git checkout; remove it and retry")
        print(f"cloning {cfg.repo_url} into {src}", file=sys.stderr)
        git("clone", "--quiet", cfg.repo_url, str(src))
    origin = git("remote", "get-url", "origin", cwd=src)
    if origin != cfg.repo_url:
        raise SetupError(f"{src} was cloned from {origin}, but the config says {cfg.repo_url}; remove it and retry")
    if git("status", "--porcelain", "--untracked-files=no", cwd=src):
        raise SetupError(f"{src} has local changes to tracked files; restore or remove it and retry")
    if git("rev-parse", "HEAD", cwd=src) != cfg.commit:
        if subprocess.run(["git", "cat-file", "-e", f"{cfg.commit}^{{commit}}"], cwd=src,
                          capture_output=True).returncode:
            git("fetch", "--quiet", "origin", cwd=src)
        git("checkout", "--quiet", "--detach", cfg.commit, cwd=src)
    if git("rev-parse", "HEAD", cwd=src) != cfg.commit:
        raise SetupError(f"could not check out {cfg.commit} in {src}")


def build(cfg: TargetConfig, work: Path) -> Path:
    """Run the configured build commands from ``work``; return the binary's path."""
    log = work / "build.log"
    env = {**os.environ, **cfg.env}
    with log.open("w") as f:
        for cmd in cfg.build:
            print(f"build: {cmd}", file=sys.stderr)
            f.write(f"$ {cmd}\n")
            f.flush()
            r = subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=work, env=env,
                               stdout=f, stderr=subprocess.STDOUT)
            if r.returncode:
                raise SetupError(f"build command failed (exit {r.returncode}): {cmd}\n  see {log}")
    binary = work / cfg.binary
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        raise SetupError(f"build finished but {binary} is not an executable file; see {log}")
    return binary


def build_prompt(cfg: TargetConfig, work: Path, goal: str | None) -> str:
    env = " ".join(f"{k}={v}" for k, v in cfg.env.items()) or "(none)"
    lines = [
        f"Profile {cfg.target.upper()} with perf and report its performance bottlenecks.",
        "",
        f"Source checkout (read-only): {work / 'src'}  ({cfg.repo_url} @ {cfg.commit})",
        f"Work directory (builds): {work}",
        f"Built binary: {work / cfg.binary}",
        "Build commands used (run from the work directory):",
        *[f"  $ {c}" for c in cfg.build],
        f"Default run command: {cfg.run_command(work)}",
        f"Environment (already set for your commands): {env}",
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


async def run(cfg: TargetConfig, root: Path, model: str | None = None,
              max_turns: int = 80, goal: str | None = None) -> int:
    if shutil.which("perf") is None:
        print("error: perf not found on PATH", file=sys.stderr)
        return 1
    root = root.resolve()
    work = root / ".work" / cfg.target
    work.mkdir(parents=True, exist_ok=True)
    try:
        checkout(cfg, work / "src")
        source = snapshot(work / "src")
        build(cfg, work)
    except SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    run_dir = root / "runs" / f"{cfg.target}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    options = build_options(cfg, work, run_dir, set(source), model, max_turns)

    report, status = None, 0
    try:
        async for message in query(prompt=build_prompt(cfg, work, goal), options=options):
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
                                  description="Check out and build a target, then let the agent profile it with perf.")
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
    return asyncio.run(run(cfg, Path.cwd(), args.model, args.max_turns, args.goal))


if __name__ == "__main__":
    sys.exit(main())
