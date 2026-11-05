# SAGE

SAGE is a command-line tool, `bottleneck-agent`, that runs a single Claude agent
built on the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python).
The agent finds performance bottlenecks in the **LULESH** and **CoMD** proxy apps,
using **Linux perf** as its only profiler.

For a given target, SAGE:

1. runs preflight checks for perf, compilers, the CPU and perf event support (`scripts/preflight.sh`),
2. checks out the official upstream repository at a pinned commit and builds it with optimization,
   `-g` and `-fno-omit-frame-pointer` (`scripts/build_target.sh`),
3. hands the binary and the preflight results to the agent. The agent runs `perf stat`, `perf record` and
   `perf annotate`, reads the hot source code, and writes a Markdown report.

The application's source code is never modified.

## Setup

### 1. Prerequisites

- Linux, with Python 3.10 or newer and `git`
- A C/C++ toolchain with OpenMP (`gcc`/`g++`), plus `make` and `cmake`. MPI is **not** needed:
  both targets are built serial.
- `perf`, for example:

  ```bash
  sudo apt install linux-tools-common linux-tools-$(uname -r)
  ```

  ```bash
  sudo dnf install perf
  ```

- Permission to profile as a normal user. A value of 1 or lower is enough for user-space
  sampling with hardware events:

  ```bash
  cat /proc/sys/kernel/perf_event_paranoid
  ```

  ```bash
  sudo sysctl kernel.perf_event_paranoid=1
  ```

  Hardware counters are often missing in VMs and containers. In that case the agent uses
  software events (`task-clock`, `cpu-clock`) and says so in the report.

- On Ubuntu, `/usr/bin/perf` is a wrapper that refuses to run when no `linux-tools` package
  matches the running kernel, which is common in VMs and containers. The preflight reports this
  and names a real perf binary under `/usr/lib/linux-tools/*/perf`. Put that binary first on
  `PATH`, for example with a symlink in `~/.local/bin`.

### 2. Install

```bash
git clone https://github.com/ariananafisi/sage.git && cd sage
```

```bash
python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install -e '.[test]'
```

### 3. Credentials

The Agent SDK needs Claude credentials. The simplest option is an API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Check the machine (optional)

`analyze` runs this itself, but you can run it alone to see what the machine supports:

```bash
scripts/preflight.sh
```

It writes `.work/preflight.json` and prints a summary. The checks:

- `perf` is installed and runs
- C and C++ compilers (gcc/g++ or clang/clang++), with their versions
- CPU vendor, model and count, architecture and kernel
- `perf_event_paranoid`
- whether `cycles` and `instructions` can be counted, plus a software event (`task-clock`)
  and topdown metrics (`perf stat -M TopdownL1`)

It exits 1 if perf or a compiler is missing, or if no events can be measured. It **warns**,
but still exits 0, when only software events are available, when topdown metrics are
unsupported, or when `perf_event_paranoid` limits an unprivileged user to user-space samples.

### 5. Build a target (optional)

`analyze` also runs this itself:

```bash
scripts/build_target.sh lulesh
```

```bash
scripts/build_target.sh comd --clean
```

The script validates the config, clones the upstream (or reuses `.work/<target>/src`), checks
out the pinned commit, and runs the build commands. `--clean` deletes `.work/<target>/` first.
It writes:

- `.work/<target>/build.log`: the full output of every build command, after a header with the
  commit, compiler version and flags.
- `.work/<target>/build-info.json`: the commit, repo URL, compiler and compiler version, flags,
  build commands, binary path, cmake and make versions, and the time of the build.
  It is written only when the build succeeds.

A failed build exits 1, prints the last lines of the log, and gives the log's path. A bad
config exits 2.

Upstream source files are never edited. The script refuses to build if a tracked file in the
checkout differs from the pinned commit, and fails if the build itself changes one. Both
targets currently build unmodified, using only command-line variables. CoMD's
`Makefile.vanilla` is copied to a new, untracked `Makefile`, so nothing upstream changes. If a
future build needs a source change, it should be stored as a patch file in this repository and
applied explicitly. No patch is needed today, so there is no patch mechanism yet.

### 6. Run

Run from the repository root, since `.work/` and `runs/` are created in the current directory:

```bash
bottleneck-agent analyze --target lulesh --config configs/lulesh.yaml
```

```bash
bottleneck-agent analyze --target comd --config configs/comd.yaml --goal "memory behaviour of the force loop"
```

Options: `--model` sets the Claude model (default: the SDK's default). `--max-turns` sets the
agent's turn limit (default 80). `--goal` adds a focus for the analysis.

## Layout

```
configs/lulesh.yaml, configs/comd.yaml   target configs
.work/preflight.json                     results of the latest preflight checks
.work/<target>/src/                      upstream checkout at the pinned commit (read-only to the agent)
.work/<target>/build.log                 output of the build commands
.work/<target>/build-info.json           commit, compiler version, flags and tools of the last good build
.work/<target>/...                       build outputs (e.g. .work/lulesh/build/)
runs/<target>-<timestamp>/               agent working directory: perf.data, logs, report.md
```

`.work/` and `runs/` are git-ignored. A later run reuses the checkout and rebuilds incrementally.
If the checkout has local changes to tracked files, or was cloned from a different URL,
SAGE stops. Delete `.work/<target>/` to start fresh.

## Target configs

The config is validated when the CLI starts, before anything is cloned, built or sent to the
API. All problems are reported together.

| Key | Required | Meaning |
| --- | --- | --- |
| `target` | yes | `lulesh` or `comd`; must match `--target` |
| `repo_url` | yes | `https://` URL of the upstream repository |
| `commit` | yes | full 40-character commit SHA to check out |
| `compiler` | yes | compiler command, available to the build commands as `$SAGE_COMPILER` |
| `flags` | yes | compile flags, available as `$SAGE_FLAGS`; must include an `-O1`/`-O2`/`-O3`/`-Os`/`-Ofast` level, `-g` and `-fno-omit-frame-pointer` |
| `build` | yes | list of shell commands, run in order from `.work/<target>/`; they must use both `$SAGE_COMPILER` and `$SAGE_FLAGS` |
| `binary` | yes | path to the built binary, relative to `.work/<target>/` |
| `run` | yes | default run command; `{binary}` expands to the binary's absolute path |
| `env` | no | environment variables for the build, the agent and its runs (names may not start with `SAGE_`) |

Pinned upstreams (both URLs checked with `git ls-remote` when they were added):

| Target | Repository | Commit |
| --- | --- | --- |
| LULESH 2.0 | https://github.com/LLNL/LULESH | `3e01c40b` (master, 2020-11-18) |
| CoMD | https://github.com/ECP-copa/CoMD | `3d48396b` (master, 2016-03-02) |

CoMD's original home, `github.com/exmatex/CoMD`, now resolves to `ECP-copa/CoMD`.

Both targets use `-O3 -g -fno-omit-frame-pointer`, so perf can resolve symbols and call graphs.
The LULESH config sets no `CMAKE_BUILD_TYPE`, so these are the exact compile flags. The build
log shows the full compile lines. The default run sizes are chosen to take about 10–15 seconds
on a 4-core machine. The agent adjusts the size if a run is far outside 5–30 seconds.

## How the agent is constrained

- **One agent.** It runs as a single `query()` call. `tools=["Bash", "Read", "Glob", "Grep"]` is
  the agent's whole toolset, so there is no Write or Edit tool and no subagent tool. No custom
  tools or MCP servers are added. `permission_mode="dontAsk"` refuses anything that is not
  pre-approved. `setting_sources=[]` means user and project settings, hooks and CLAUDE.md files
  are not loaded.
- **perf only.** A `PreToolUse` hook denies shell commands that call other profilers
  (gprof, valgrind, VTune, likwid, HPCToolkit, Nsight, strace, `-pg`, and others).
- **Source is read-only.** The same hook denies commands that would edit, move or delete files
  in the checkout: in-place editors (`sed -i`, `patch`, editors), redirections,
  `cp`/`mv`/`rm`/`tee`/`dd`, `find -exec`, `bash -c`, and git commands that change the working
  tree. New files, such as build outputs, are allowed.
- **Verified afterwards.** SAGE hashes every git-tracked file in the checkout before the build,
  and checks them again after the agent finishes. This catches writes the command check cannot
  see, such as a write from inside `python -c`.

Code: `scripts/preflight.sh` and `scripts/build_target.sh` (machine checks, checkout and build),
`sage/config.py` (config loading and validation, also used by the build script),
`sage/agent.py` (the agent and the CLI), `sage/guard.py` (the safety rails).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | report written |
| 1 | preflight failed (no usable perf, compiler or events), checkout or build failed, or the agent run failed |
| 2 | invalid command line, target or config |
| 3 | application source was modified |

## Tests

The tests need no Anthropic API access, MPI, perf binary, hardware counters, compiler or
network. The SDK's `query()` is replaced with a fake. perf and the compiler are replaced with
small fake executables, which simulate machines with full hardware counters, with software
events only, or with a broken perf. A local git repository stands in for the upstream.

```bash
pytest
```
