# SAGE

SAGE is a command-line tool, `bottleneck-agent`, that runs a single Claude agent
built on the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python).
The agent finds performance bottlenecks in the **LULESH** and **CoMD** proxy apps,
using **Linux perf** as its only profiler.

For a given target, SAGE:

1. checks out the official upstream repository at a pinned commit,
2. builds it with the target's configured commands,
3. hands the binary to the agent. The agent runs `perf stat`, `perf record` and
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

### 4. Run

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
.work/<target>/src/                      upstream checkout at the pinned commit (read-only to the agent)
.work/<target>/build.log                 output of the build commands
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
| `build` | yes | list of shell commands, run in order from `.work/<target>/` |
| `binary` | yes | path to the built binary, relative to `.work/<target>/` |
| `run` | yes | default run command; `{binary}` expands to the binary's absolute path |
| `env` | no | environment variables for the build, the agent and its runs |

Pinned upstreams (both URLs checked with `git ls-remote` when they were added):

| Target | Repository | Commit |
| --- | --- | --- |
| LULESH 2.0 | https://github.com/LLNL/LULESH | `3e01c40b` (master, 2020-11-18) |
| CoMD | https://github.com/ECP-copa/CoMD | `3d48396b` (master, 2016-03-02) |

CoMD's original home, `github.com/exmatex/CoMD`, now resolves to `ECP-copa/CoMD`.

Both builds produce optimized binaries with `-g -fno-omit-frame-pointer`, so perf can
resolve symbols and call graphs. The default run sizes are chosen to take about 10–15 seconds
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

Code: `sage/config.py` (config loading and validation), `sage/agent.py` (checkout, build,
the agent and the CLI), `sage/guard.py` (the safety rails).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | report written |
| 1 | setup failed (perf missing, checkout or build error) or the agent run failed |
| 2 | invalid command line, target or config |
| 3 | application source was modified |

## Tests

The tests need no Anthropic API access, MPI, perf binary, hardware counters or network.
The SDK's `query()` is replaced with a fake, and a local git repository stands in for the
upstream.

```bash
pytest
```
