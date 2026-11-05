# SAGE

SAGE is a command-line tool, `bottleneck-agent`, that runs a single Claude agent
built on the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python).
The agent finds performance bottlenecks in the **LULESH** and **CoMD** proxy apps,
using **Linux perf** as its only profiler.

For a given target, SAGE runs preflight checks for perf, compilers, the CPU and perf event
support (`scripts/preflight.sh`). If the machine can profile, it starts one agent session. The
agent follows a fixed workflow:

1. Check the preflight output, and build the target if needed: `scripts/build_target.sh` checks
   out the official upstream at a pinned commit and builds it with optimization, `-g` and
   `-fno-omit-frame-pointer`.
2. Run the baseline profile with `scripts/profile_target.sh`: `perf stat` with 3 repeats,
   `perf record`, a self-time report, and metadata.
3. Inspect the `perf stat` and `perf report` output.
4. Annotate the top functions (`scripts/annotate_symbol.sh`) and read the matching source.
5. Form a testable hypothesis.
6. Run one controlled experiment, changing only one factor.
7. Write `findings.json` and validate it.

SAGE then validates `findings.json` itself (one retry is allowed), generates `report.md` from it,
saves the full transcript, and checks that the application's source is unchanged.

Each step is also a standalone script you can run yourself.

## Setup

### 1. Prerequisites

- Linux, with Python 3.10 or newer and `git`
- A C/C++ toolchain with OpenMP (`gcc`/`g++`), plus `make` and `cmake`.
- MPI is optional. The default builds and runs are serial. To build and profile the MPI
  variants (`--mpi`), you need an MPI implementation with `mpicc`, `mpicxx` and `mpirun`:

  ```bash
  sudo apt install openmpi-bin libopenmpi-dev
  ```

  The configs' launchers use Open MPI syntax; see [MPI runs](#mpi-runs). Open MPI refuses to
  run as root unless `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` are set,
  as in some containers.
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

During `analyze`, the agent runs this itself when the build is missing or stale:

```bash
scripts/build_target.sh lulesh
```

```bash
scripts/build_target.sh comd --clean
```

```bash
scripts/build_target.sh comd --mpi
```

The script validates the config, clones the upstream (or reuses `.work/<target>/src`), checks
out the pinned commit, and runs the build commands. `--clean` deletes `.work/<target>/` first.
It writes:

- `.work/<target>/build.log`: the full output of every build command, after a header with the
  commit, compiler version and flags.
- `.work/<target>/build-info.json`: the commit, repo URL, compiler and compiler version, flags,
  build commands, binary path, cmake and make versions, and the time of the build.
  It is written only when the build succeeds.
- `--mpi` builds the config's MPI variant and writes `build-mpi.log` and `build-info-mpi.json`
  instead, so the serial and MPI records never overwrite each other.

A failed build exits 1, prints the last lines of the log, and gives the log's path. A bad
config exits 2.

Upstream source files are never edited. The script refuses to build if a tracked file in the
checkout differs from the pinned commit, and fails if the build itself changes one. Both
targets currently build unmodified, using only command-line variables. CoMD's
`Makefile.vanilla` is copied to a new, untracked `Makefile`, so nothing upstream changes. If a
future build needs a source change, it should be stored as a patch file in this repository and
applied explicitly. No patch is needed today, so there is no patch mechanism yet.

### 6. Profile a run (optional)

During `analyze`, the agent creates its baseline and experiments with this script:

```bash
scripts/profile_target.sh comd baseline --label "default size"
```

```bash
scripts/profile_target.sh lulesh small --cmd "{binary} -s 20 -i 50"
```

```bash
scripts/annotate_symbol.sh baseline ljForce
```

The target must already be built. `profile_target.sh <target> <run-id>` creates
`runs/<run-id>/`. An existing run directory is never overwritten.

| File | Contents |
| --- | --- |
| `metadata.json` | run id, label, status (and error if a step failed), start/end times, the script's command line, the run command, wall-clock times (whole script, each perf step, and the app's mean ± variance over the 3 repeats), perf version and the exact perf commands, the parsed perf stat and report summaries, the build record, CPU details (`/proc/cpuinfo`, `lscpu`), thread settings (CPU affinity, `OMP_*`/`MKL_*`/... variables, MPI ranks), the environment, and a list of artifacts |
| `stdout.log`, `stderr.log` | output of every execution, with a header line per step. `stderr.log` also holds perf's own messages |
| `perf-stat.txt` | `perf stat -r 3 -d`: default and cache events, mean and variance over 3 repeats |
| `perf.data` | `perf record -g` samples with frame-pointer call graphs |
| `perf-report.txt` | `perf report --no-children --sort overhead,dso,sym`: self time per DSO and symbol, highest first |

Options:

- `--label TEXT`: a note stored in the metadata.
- `--cmd COMMAND`: replaces the config's run command; `{binary}` expands to the binary.
- `--ranks N`: an MPI run.
- `--runs DIR`: where the run directory goes.
- `--config`, `--work`: as for `build_target.sh`.

Environment variables whose names look like secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASS*`,
`*PROXY*`, and similar) are recorded as `<redacted>`.

Each step must succeed before the next one runs. If perf, or the application under perf, fails,
the script stops and exits 1. It prints which step failed, with its exit code and the last lines
of `stderr.log`, and records `"status": "failed"` and the error in `metadata.json`. A report with
no samples also counts as a failure. Bad arguments, an existing run id or a bad config exit 2,
and nothing is written.

`annotate_symbol.sh <run-id> <symbol>` saves `perf annotate --stdio` (source interleaved with
disassembly and sample percentages) to `runs/<run-id>/annotate/<symbol>.txt`. If the symbol has
no samples, it exits 1 and points to `perf-report.txt` for valid names. For MPI runs, pass
`--rank R`.

#### MPI runs

`--ranks N` uses the config's `mpi` section. The application runs under the configured launcher,
for example `mpirun -np N`.

- `perf stat` measures the whole job: the launcher plus all ranks, aggregated.
- `perf record` runs once per rank, through a small wrapper (`perf-rank-wrapper.sh` in the run
  directory). The wrapper reads the rank from the MPI runtime (`OMPI_COMM_WORLD_RANK`,
  `PMIX_RANK`, `PMI_RANK`, `MV2_COMM_WORLD_RANK` or `SLURM_PROCID`).
- Each rank gets `rank-<r>/perf.data` and `rank-<r>/perf-report.txt` instead of the top-level pair.
- `metadata.json` holds a parsed report per rank.

The config's MPI run command is written for `mpi.ranks` ranks, because LULESH needs a cube number
of ranks and CoMD's `-i/-j/-k` decomposition must multiply to the rank count. Any other `--ranks`
needs a matching `--cmd`:

```bash
scripts/profile_target.sh comd mpi8 --ranks 8 --cmd "{binary} -i 2 -j 2 -k 2 -x 40 -y 40 -z 40 -N 100"
```

LULESH's default is 8 ranks with `--oversubscribe`. On a machine with fewer than 8 cores, the
ranks share cores, and much of the profile will be MPI busy-waiting rather than computation.

### 7. Run the agent

Run from the repository root, since `.work/` and `runs/` are created in the current directory:

```bash
bottleneck-agent analyze --target lulesh --config configs/lulesh.yaml
```

```bash
bottleneck-agent analyze --target comd --config configs/comd.yaml --goal "memory behaviour of the force loop"
```

```bash
bottleneck-agent analyze --target comd --config configs/comd.yaml --mpi --run-id comd-mpi-1
```

Options:

Options:

- `--run-id`: the run directory name (default `<target>-<timestamp>`). The CLI rejects an
  existing one before doing anything.
- `--label`: stored in the baseline's `metadata.json`.
- `--mpi`: build and profile the MPI variant with the config's default rank count.
- `--goal`: an extra focus for the analysis.

Limits, all validated before anything runs:

| Option | Default | Limits |
| --- | --- | --- |
| `--model` | the SDK's default | the Claude model |
| `--max-turns` | 80 | agent turns, for the analysis and again for the retry |
| `--timeout` | `60m` | wall-clock time for the whole agent session, retry included (`90s`, `45m`, `1.5h`, or plain seconds) |
| `--max-experiments` | 1 | experiment artifact sets the agent may create; later attempts are refused |

A run directory looks like this:

```
runs/<run-id>/
  baseline/            the baseline artifact set (metadata.json, perf-stat.txt, perf.data,
                       perf-report.txt, stdout.log, stderr.log, annotate/)
  experiments/<name>/  one artifact set per experiment
  findings.json        the agent's result
  report.md            generated from findings.json
  transcript.jsonl     the full agent transcript
```

`transcript.jsonl` has one JSON record per line:

- the start (limits, tools and the system prompt)
- every prompt sent to the agent
- every SDK message received (assistant text, tool calls and their results, the final result)
- each findings.json validation, any errors, a timeout
- the end status

It is written as the run goes, so it survives a crash or a timeout.

### Prompts

The prompts are in `sage/prompts/`. They are filled in at run time, and a missing or unused value
is an error.

- `system.md`: the ground rules and the seven-step workflow above.
- `task.md`: this run's paths, the preflight summary, the exact commands, and the limits.
- `retry.md`: the follow-up sent when `findings.json` fails validation.

The system prompt tells the agent:

- not to rely on prior knowledge of the application. It must learn options from `--help` and the
  source, and back every claim with something measured in the run. For the same reason, the prompts
  contain no application-specific hints.
- to keep measured facts and inferences apart. `supporting_evidence` holds only facts, each naming
  its artifact. `likely_cause` and the interpretation in `experiment_result` are inferences and
  must be worded as such.

### Retry and failure

When the agent finishes, SAGE validates `findings.json` itself. If the file is missing or invalid,
SAGE sends the problems back to the same agent session once, and the agent fixes the file. If it is
still invalid, the run fails with exit code 4, lists the remaining problems, and writes no report.
Hitting the turn limit is handled the same way: the findings are validated, and if they are not
valid, the retry is used.

Reaching the wall-clock limit stops the agent immediately and exits 5. If a valid `findings.json`
already exists, the report is still generated.

## Findings and report

The agent's final result is `findings.json` in the run directory. It is defined by the JSON schema
in [`sage/findings.schema.json`](sage/findings.schema.json) (draft-07). The agent writes the file
through the shell. It has no file-writing tool, and SAGE adds none. The agent then runs the
validator itself and fixes any problems.

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `run_id`, `target` | the run directory's name and `lulesh` or `comd` |
| `status` | `bottlenecks_found` (needs at least one finding) or `inconclusive` (findings may be empty) |
| `summary` | a few sentences on what limits performance, or why it is unclear |
| `findings` | the findings, see below |
| `limitations` | optional: what could not be measured or checked |

Every finding has all of these fields:

| Field | Meaning |
| --- | --- |
| `rank` | priority: 1 is the most important; ranks must be 1..n with no gaps or repeats |
| `function` | symbol as perf reports it |
| `source_location` | `path:line` or `path:first-last` in the checkout |
| `sample_percent` | self-time share, 0–100 |
| `bottleneck_type` | `compute_bound`, `memory_bandwidth_bound`, `memory_latency_bound`, `branch_misprediction`, `poor_vectorization`, `serial_section`, `load_imbalance`, `mpi_communication`, `synchronization`, `kernel_or_system` or `other` |
| `supporting_evidence` | at least one observation, with measured numbers |
| `likely_cause` | the explanation |
| `experiment`, `experiment_result` | what was run to test it, and what it showed |
| `confidence` | `high`, `medium` or `low` |
| `artifact_paths` | at least one file or directory holding the evidence, relative to the run directory |

The validator checks the schema, then:

- `run_id` must match the run directory's name.
- Ranks must be 1..n with no gaps or repeats.
- Every artifact path must exist inside the run directory. Paths that escape it, including through
  `..` or a symlink, are rejected.
- `findings.json` and `report.md` themselves cannot be cited as evidence.

A finding without artifact evidence is invalid. All problems are reported at once, with their
location (for example `findings[1].artifact_paths[0]: ...`).

```bash
python -m sage.findings validate runs/comd-baseline
```

```bash
python -m sage.findings render runs/comd-baseline
```

`render` validates first, then writes `report.md`. The report contains:

- the status and summary
- run details from `metadata.json`: command, build, wall-clock, host, unavailable perf events
- a findings table ordered by rank
- one section per finding: evidence, likely cause, experiment and result, and links to its artifacts
- the limitations

`analyze` deletes any `report.md` the agent wrote. If `findings.json` is missing or invalid, it
prints the problems, writes no report, and exits 4.

## Layout

```
configs/lulesh.yaml, configs/comd.yaml   target configs
.work/preflight.json                     results of the latest preflight checks
.work/<target>/src/                      upstream checkout at the pinned commit (read-only to the agent)
.work/<target>/build.log                 output of the build commands
.work/<target>/build-info.json           commit, compiler version, flags and tools of the last good build
.work/<target>/build-mpi.log, build-info-mpi.json   the same for the MPI variant
.work/<target>/...                       build outputs (e.g. .work/lulesh/build/, build-mpi/)
runs/<run-id>/                           one artifact set (see above); for analyze also the agent's
                                         working directory, with annotate/, experiments/,
                                         findings.json (agent result) and report.md (generated)
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
| `mpi` | no | MPI variant: `compiler` (e.g. `mpicxx`), `build`, `binary` and `run` (same rules as above), `launcher` (must use `{ranks}`, e.g. `mpirun -np {ranks}`), `ranks` (the rank count `run` is written for), and optional `env` added on top of the top-level `env`. It shares `flags` with the serial build |

Pinned upstreams (both URLs checked with `git ls-remote` when they were added):

| Target | Repository | Commit |
| --- | --- | --- |
| LULESH 2.0 | https://github.com/LLNL/LULESH | `3e01c40b` (master, 2020-11-18) |
| CoMD | https://github.com/ECP-copa/CoMD | `3d48396b` (master, 2016-03-02) |

CoMD's original home, `github.com/exmatex/CoMD`, now resolves to `ECP-copa/CoMD`.

Both targets use `-O3 -g -fno-omit-frame-pointer`, so perf can resolve symbols and call graphs.
The LULESH config sets no `CMAKE_BUILD_TYPE`, so these are the exact compile flags. The build
log shows the full compile lines. The default run sizes are chosen to take about 10–15 seconds
on a 4-core machine. The agent notes it in its findings if a run is very short or very long.

## How the agent is constrained

- **One agent.** It runs as one `ClaudeSDKClient` session, used for the analysis and the retry.
  `tools=["Bash", "Read", "Glob", "Grep", "Write", "Edit"]` is the agent's whole toolset. There is
  no subagent tool, no web tool, and no custom tools or MCP servers. `permission_mode="dontAsk"`
  refuses anything that is not pre-approved. `setting_sources=[]` means user and project settings,
  hooks and CLAUDE.md files are not loaded.
- **Writing.** Write and Edit work only on `runs/<run-id>/findings.json`. Any other path, or a
  symlink in its place, is refused. `report.md` is always generated by SAGE: one written by the
  agent is deleted. The shell can still create perf output and notes, because experiments need it.
- **Every run is recorded.** The application may only be run through `profile_target.sh`.
  - Direct `perf stat`/`perf record`/`perf top`/`perf trace` and MPI launchers are denied.
  - Running the target binary directly is denied, except `--help`/`-h`.
  - `perf report`, `perf annotate` and `perf script` on existing data are allowed.
  - Artifact sets may only go to `runs/<run-id>/baseline` or `runs/<run-id>/experiments/`.
    Experiments beyond `--max-experiments` are refused.
- **perf only.** A `PreToolUse` hook denies shell commands that call other profilers
  (gprof, valgrind, VTune, likwid, HPCToolkit, Nsight, strace, `-pg`, and others).
- **Source is read-only.** The same hook denies commands that would edit, move or delete a file
  git tracks in the checkout. That covers in-place editors (`sed -i`, `patch`, editors),
  redirections, `cp`/`mv`/`rm`/`tee`/`dd`, `find -exec` and `bash -c`. Only read-only git
  subcommands are allowed. The tracked files are looked up on every command, because the agent
  may clone the checkout during the run. New files, such as build outputs, are allowed.
- **Verified afterwards.** When the session ends, SAGE compares every file of the pinned commit
  with the checkout by content (`git hash-object` against the commit's tree), and checks HEAD.
  This catches what the command check cannot see, such as a write from inside `python -c`. It
  cannot be fooled by git index tricks such as `--assume-unchanged`.

These command checks are best-effort pattern matching on shell text. The content comparison
after the run is the guarantee for source files.

Code: `scripts/preflight.sh` and `scripts/build_target.sh` (machine checks, checkout and build),
`scripts/profile_target.sh` and `scripts/annotate_symbol.sh` (perf artifact sets),
`sage/config.py` (config loading and validation, also used by the scripts),
`sage/artifacts.py` (perf stat and report parsers and the `metadata.json` writer),
`sage/findings.py` and `sage/findings.schema.json` (the result format, its validator and the report generator),
`sage/agent.py` and `sage/prompts/` (the agent, its prompts, limits and the CLI), `sage/guard.py` (the safety rails).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | report written |
| 1 | preflight failed (no usable perf, compiler or events) |
| 2 | invalid command line, target or config |
| 3 | application source was modified |
| 4 | `findings.json` is missing or still invalid after one retry; no report was written |
| 5 | the wall-clock limit (`--timeout`) was reached |

## Tests

The tests need no Anthropic API access, MPI, perf binary, hardware counters, compiler or
network. `ClaudeSDKClient` is replaced with a fake that plays scripted agent turns. These cover
prompt loading, limits, the retry, timeouts and the transcript. perf, `mpirun` and the compiler are
replaced with small fake executables, which simulate machines with full hardware counters, with
software events only, or with a broken perf, and can make any perf step fail. A local git
repository stands in for the upstream.

The findings validator is tested against valid and invalid `findings.json` files in
`tests/fixtures/findings/`, and against many single-field changes to them.

The parsers are tested against saved perf output in `tests/fixtures/`. The software-only
`perf stat`, the report and the annotate output are real output from a VM. The hardware-counter
and single-run `perf stat` samples follow the same perf 6.x format.

```bash
pytest
```
