"""SAGE: a single Claude agent that finds performance bottlenecks with Linux perf."""

from __future__ import annotations

import argparse
import asyncio
import shutil
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

from .guard import changed, make_guard, snapshot

TOOLS = ["Bash", "Read", "Glob", "Grep"]

SYSTEM_PROMPT = """\
You are SAGE, a performance engineer. Your job is to find and explain the
performance bottlenecks of an HPC proxy application using Linux perf.

Rules (enforced; violating commands are denied):
- perf is the only profiler. Do not use gprof, valgrind, VTune, strace, -pg, etc.
- Never modify application source code, Makefiles or CMake files. Configure the
  build only through command-line variables (e.g. `make CC=gcc VAR=value`,
  `cmake -DVAR=value`). Creating new files (build directories, a Makefile copied
  from a shipped template, perf.data, logs) is fine.
- Work inside the current directory. The application is in ./src.
- MPI may be unavailable: build and run serial / single-rank variants.

Method:
1. Check the environment: `perf --version`, /proc/sys/kernel/perf_event_paranoid,
   CPU model, core count, and whether hardware events work
   (`perf stat -e cycles,instructions true`). If hardware counters are
   unavailable, fall back to software events (task-clock, cpu-clock) and say so.
2. Read the build files and README, then build an optimized binary with symbols
   and frame pointers (-O2 or -O3, -g, -fno-omit-frame-pointer).
3. Choose a problem size that runs for roughly 5-30 seconds. Record the exact
   command line and the application's own timing output.
4. Baseline with `perf stat` (cycles, instructions, IPC, cache and branch
   misses, as available).
5. Sample with `perf record -g` (try `--call-graph dwarf` if frame-pointer
   stacks look broken) and inspect `perf report --stdio --no-children` and
   `perf annotate --stdio` for the top functions.
6. For each hotspot, read the relevant source and classify the bottleneck
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
        "The application is LULESH 2.0 (C++, hydrodynamics). It ships a CMakeLists.txt "
        "(options WITH_MPI, WITH_OPENMP) and a Makefile. Prefer an out-of-tree CMake "
        "build with -DWITH_MPI=OFF. Run `lulesh2.0 -h` for options such as -s (size) "
        "and -i (iterations). If built with OpenMP, set OMP_NUM_THREADS explicitly."
    ),
    "comd": (
        "The application is CoMD (C, classical molecular dynamics). Use src/src-mpi: "
        "copy Makefile.vanilla to Makefile (a new file) and set variables on the make "
        "command line, e.g. DO_MPI=OFF CC=gcc OPTFLAGS='-O3 -g -fno-omit-frame-pointer'. "
        "Run the binary in src/bin with --help for options such as -x/-y/-z (lattice) "
        "and -N (steps); -e selects the EAM potential (needs --potDir src/pots)."
    ),
}


def build_options(app: str, workdir: Path, protected: set[str], model: str | None, max_turns: int) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        tools=TOOLS,  # the only built-in tools the agent can see: no Write/Edit, no subagents
        allowed_tools=TOOLS,
        permission_mode="dontAsk",  # anything not pre-approved is denied, never prompted
        system_prompt=SYSTEM_PROMPT + "\n" + APP_HINTS[app],
        cwd=str(workdir),
        setting_sources=[],  # ignore user/project settings, hooks and CLAUDE.md files
        hooks={"PreToolUse": [HookMatcher(hooks=[make_guard(protected)])]},
        model=model,
        max_turns=max_turns,
    )


async def run(app: str, app_dir: Path, workdir: Path, model: str | None = None,
              max_turns: int = 80, goal: str | None = None) -> int:
    app_dir = app_dir.resolve()
    workdir = workdir.resolve()
    if not app_dir.is_dir():
        print(f"error: {app_dir} is not a directory", file=sys.stderr)
        return 1
    if workdir == app_dir or app_dir in workdir.parents:
        print("error: --workdir must be outside the application directory", file=sys.stderr)
        return 1
    if shutil.which("perf") is None:
        print("error: perf not found on PATH", file=sys.stderr)
        return 1

    # The agent builds and profiles a pristine copy so the user's checkout stays clean.
    src = workdir / "src"
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(app_dir, src, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
    original, copy = snapshot(app_dir), snapshot(src)

    options = build_options(app, workdir, set(original) | set(copy), model, max_turns)
    prompt = f"Profile {app.upper()} (source in ./src) with perf and report its performance bottlenecks."
    if goal:
        prompt += f"\nAdditional focus: {goal}"

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
        (workdir / "report.md").write_text(report)
        print(report)

    violations = changed(original) + changed(copy)
    if violations:
        print("error: application source was modified:\n  " + "\n  ".join(violations), file=sys.stderr)
        return 2
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sage", description=__doc__)
    parser.add_argument("app", choices=sorted(APP_HINTS), help="proxy application to profile")
    parser.add_argument("app_dir", type=Path, help="path to the application's source checkout")
    parser.add_argument("--workdir", type=Path, help="where builds, perf data and report.md go "
                        "(default: ./sage-runs/<app>-<timestamp>)")
    parser.add_argument("--model", help="Claude model to use (default: SDK default)")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--goal", help="extra focus for the analysis, e.g. 'memory behaviour'")
    args = parser.parse_args(argv)
    workdir = args.workdir or Path("sage-runs") / f"{args.app}-{time.strftime('%Y%m%d-%H%M%S')}"
    return asyncio.run(run(args.app, args.app_dir, workdir, args.model, args.max_turns, args.goal))


if __name__ == "__main__":
    sys.exit(main())
