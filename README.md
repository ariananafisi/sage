# SAGE

SAGE is a command-line tool that runs a single Claude agent (built on the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python)). The agent
finds performance bottlenecks in the **LULESH** and **CoMD** proxy apps, using
**Linux perf** as its only profiler.

The agent builds the app, picks a problem size, runs `perf stat`,
`perf record` and `perf annotate`, reads the hot source code, and writes a
Markdown report. It never changes application code.

## Usage

```bash
pip install -e .
```

```bash
sage lulesh ~/src/LULESH
```

```bash
sage comd ~/src/CoMD --goal "memory behaviour of the force kernels" --workdir runs/comd1
```

Requirements: `perf` on `PATH`, a C/C++ compiler and CMake, and Claude credentials
that the Agent SDK can use (for example, `ANTHROPIC_API_KEY`). MPI is not needed:
the agent builds serial variants. If hardware counters are not available (for
example, in a VM, or when `perf_event_paranoid` is restrictive), it uses
software events and says so in the report.

Output goes to the workdir (default `./sage-runs/<app>-<timestamp>/`):
`src/` holds a pristine copy of the app, plus the build and perf data the agent
creates. `report.md` holds the final report, which is also printed to stdout.
Agent progress is printed to stderr.

Exit codes: `0` success, `1` setup or agent failure, `2` application source was modified.

## Design

The code is in two small modules:

- `sage/agent.py`: the system prompt, the `ClaudeAgentOptions`, the `query()` loop and the CLI.
- `sage/guard.py`: the safety rails.

The agent is set up like this:

- **One agent.** It runs one `query()` call. `tools=["Bash", "Read", "Glob", "Grep"]` is the only
  toolset the model sees, so there is no Write or Edit tool and no subagent tool.
  No custom tools or MCP servers are added.
- **No prompts, no inherited config.** `permission_mode="dontAsk"` denies anything that is not
  pre-approved. `setting_sources=[]` means user and project settings, hooks and CLAUDE.md files
  are not loaded.
- **perf only.** A `PreToolUse` hook denies shell commands that call other profilers
  (gprof, valgrind, VTune, likwid, HPCToolkit, Nsight, strace, `-pg`, and others).
- **Source is read-only.** The same hook denies commands that would edit or remove a source file
  (`*.c`, `*.cc`, `*.h`, `Makefile*`, `CMakeLists.txt`, and so on) that existed before the run.
  This covers in-place editors (`sed -i`, `patch`, editors), redirections, `cp`/`mv`/`rm`/`tee`/`dd`,
  `find -exec`, `bash -c`, and git commands that change the working tree. Creating *new* files is
  allowed. CoMD's build needs this: it copies `Makefile.vanilla` to `Makefile` and generates
  `CoMD_info.h`.
- **Verified afterwards.** The agent works on a copy of the app, so your checkout stays clean.
  SAGE hashes the source files in both the original and the copy before the run, and checks them
  after. This catches writes the command check cannot see, such as a write from inside
  `python -c`. If anything changed, SAGE exits with code 2.

## Tests

The tests replace the SDK's `query()` with a fake. They need no API access, MPI,
perf binary or hardware counters.

```bash
pip install -e '.[test]'
```

```bash
pytest
```
