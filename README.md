# SAGE

SAGE is a command-line tool, `bottleneck-agent`, that runs a single LLM agent. The agent finds
performance bottlenecks in the **LULESH** and **CoMD** proxy apps, using **Linux perf** as its only
profiler.

The model is an OpenAI model (`gpt-5` by default) served by **HokieAI**, Virginia Tech's Azure
API Management gateway. SAGE talks to it with the
[OpenAI Python SDK](https://github.com/openai/openai-python): the Chat Completions API, with
function calling for the agent's four tools.

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

### 3. HokieAI key

Put your HokieAI subscription key in `.env` at the repository root. `.env` is git-ignored, so the
key is never committed:

```bash
cp .env.example .env
```

Then edit `.env` and set `HOKIEAI_API=<your key>`. A `HOKIEAI_API` environment variable, if set,
takes precedence over `.env`.

- The default endpoint is `https://apim-n1ai-usc-e1f7a372a.azure-api.net/openai/v1`. Set
  `HOKIEAI_BASE_URL` in `.env` or the environment to use another.
- SAGE sends the key in the `api-key` header, which is the one the gateway checks.
- Which models you may use depends on your HokieAI subscription. With the subscription SAGE was
  tested on, `gpt-5` and `gpt-5-mini` were allowed, and `gpt-4o`, `gpt-4.1`, `o3`, `o4-mini` and
  `gpt-5.1` were refused with `ModelNotAllowed`.

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
| `functions.json` | each function's self time split over the threads (or, for MPI, the ranks), with an imbalance factor (max / mean over workers), the share of time in parallel runtime libraries (OpenMP/MPI waiting) and in the kernel, and the peak memory of the largest process |
| `summary.md` | a readable digest: wall-clock and variance, peak memory next to the last-level cache size, counted events, and the per-function, per-thread (or per-rank) table. The agent starts here |

Two analysis commands work on artifact sets:

```bash
python -m sage.analysis summary runs/baseline
```

```bash
python -m sage.analysis compare runs/large runs/small --work 120^3*40 30^3*640
```

- `summary` (re)writes `summary.md` and `functions.json`.
- `compare` shows, per function, the time in A and B, the ratio, and with `--work` the ratio of
  cost per unit of work (for example cells × iterations), plus each side's imbalance. Each `--work`
  value is a number or a product of numbers (`*`, with optional `^` powers); the tool multiplies,
  and prints the product next to its result. For example,
  it shows whether a function gets cheaper per element when the data fits in cache.

Options:

- `--label TEXT`: a note stored in the metadata.
- `--cmd COMMAND`: replaces the config's run command; `{binary}` expands to the binary. It may not
  set environment variables (`env X=1 ...` or `X=1 ...`), because those would not be recorded.
- `--env NAME=VALUE` (repeatable): sets an environment variable for the application, overriding
  the config's `env`, e.g. `--env OMP_NUM_THREADS=1`. The run uses it, and `metadata.json` records
  it, both in its environment and thread settings and as `env_overrides`.
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
| `--model` | `gpt-5` | the HokieAI model; it must support function calling and be allowed for your subscription |
| `--max-turns` | 80 | model calls, for the analysis and again for the retry |
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
- every model response, in full: assistant text, tool calls and token usage
- every tool call's arguments and result
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
- a general diagnosis method, applied to every application function with at least 3% of self time.
  It contains nothing specific to LULESH or CoMD:
  1. **Load imbalance:** from the per-thread or per-rank split in `summary.md`. An imbalance factor
     of about 1.1 or more (1.0 is even) means uneven work. When the busiest workers in that function
     spend the least time in the parallel runtime, the others are waiting for them. It is reported
     for the function with the uneven work, not the runtime's spin loop.
  2. **Memory-bound or not:** from a working-set experiment. Rerun with a problem that fits well
     inside the last-level cache (a tenth of its reported size, since it is often shared), then compare cost per unit of work with
     `compare --work`. A per-unit ratio of about 0.6 or less means memory-bound; 0.6–0.9 means partly memory-sensitive, which is classified as not memory-bound; about 0.9–1.1 means
     not memory-bound (compute-bound, or poor vectorization if the annotation shows scalar loops).
  3. **Corroboration:** where the hot instructions sit in the annotation (loads versus arithmetic,
     allowing for sample skid), and hardware counters if the machine has them.

  The thresholds come from the injected cases on the 4-vCPU development VM, so treat them as
  calibrated on one machine. Once their data fits in cache, a memory-bound injection's cost per
  unit falls to 0.28–0.46 (LULESH) or 0.25–0.49 (CoMD). Unmodified code falls only to 0.75–1.06
  (LULESH, depending on how small the experiment is) or stays at 1.07–1.13 (CoMD). Work must be
  counted from what the run reports: LULESH stops at its end time, so `-s 30 -i 2000` runs 932
  iterations, and counting 2000 makes every kernel look memory-bound. An imbalance
  injection gives a factor of 2.9 (LULESH, threads) or 1.19 (CoMD, ranks); unmodified code gives
  1.01–1.04. The reported LLC (260 MiB) overstates the usable cache: CoMD's injected table behaves
  as cache-resident at 32 MiB but not at 64 MiB.

### Retry and failure

When the agent finishes, SAGE validates `findings.json` itself. If the file is missing or invalid,
SAGE sends the problems back to the same conversation once, and the agent fixes the file. If it is
still invalid, the run fails with exit code 4, lists the remaining problems, and writes no report.
Hitting the turn limit is handled the same way: the findings are validated, and if they are not
valid, the retry is used.

If HokieAI answers HTTP 429 (its token or request limits, which renew every 60 seconds), SAGE waits
and retries the call. It waits for the `Retry-After` time if the gateway sends one, otherwise 15,
30, 60 and 60 seconds, but never past the wall-clock limit. If the limit persists, the run fails
with exit code 1. One analysis sends roughly 1M input tokens, most of them served from the
gateway's prompt cache, so starting runs back to back can hit a per-minute limit.

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

## Evaluation

`bottleneck-agent eval` measures how well the agent diagnoses bottlenecks whose cause is known. Run
one suite per command:

```bash
bottleneck-agent eval --suite suites/lulesh.yaml
```

```bash
bottleneck-agent eval --suite suites/comd.yaml --timeout 45m
```

Options:

- `--out DIR`: the output directory (default `evals/<suite>-<timestamp>/`).
- `--cases id,id`: run only some cases.
- `--env NAME=VALUE`: set an environment variable for every case, for example the thread count
  on a large node (repeatable; overrides the suite's `env`).
- `--model`, `--max-turns`, `--timeout`, `--max-experiments`: the same limits as `analyze`, applied
  to each case.

Each case runs on a fresh copy of the pinned source, through the same pipeline as `analyze`:
preflight, then the agent builds, profiles, experiments and writes `findings.json`. The copy is
the pinned tree with the case's patch applied, committed as a single commit with no upstream
history. The source check after the run compares against that commit.

### Suites

`suites/lulesh.yaml` and `suites/comd.yaml` each have three cases. The code changes are patch files
in `suites/patches/`, and each case's expected result is in the suite YAML.

**LULESH's ground truth is a published roofline study.** The unmodified LULESH case is scored against
the roofline analysis by A. Afzal, G. Hager and G. Wellein, ["Analytic Roofline Modeling and Energy
Analysis of LULESH Proxy Application on Multi-Core Clusters"](https://arxiv.org/abs/2412.08792)
(arXiv:2412.08792, 2024). In the OpenMP version, it finds five of the six hot functions
memory-bound: `CalcHourglassControlForElems`, `CalcFBHourglassForceForElems`,
`IntegrateStressForElems`, `CalcMonotonicQGradientsForElems` and `EvalEOSForElems`.
`CalcKinematicsForElems` is not. The classification is recorded, with each function's measured
computational intensity, in
[`suites/references/lulesh-afzal2024.yaml`](suites/references/lulesh-afzal2024.yaml).

Every LULESH case runs with the study's settings, `-s 120 -i 40 -b 0 -c 0`:

- LULESH's built-in load imbalance is off (`-b 0 -c 0`).
- The working set is about 1.5 GB.
- 40 iterations instead of the study's 300, to keep a 4-core run near 25 s. The classification is
  per iteration, so the count doesn't affect it.

**The reference is only scored on hardware like the study's.** The study measured Intel Ice Lake
and Sapphire Rapids nodes with 13–18 cores per memory domain, using hardware counters and AVX-512
builds. The classification depends on that concurrency: a few cores cannot draw enough bandwidth to
hit the memory ceiling.

On the 4-vCPU development VM (260 MiB L3), the main LULESH kernels kept the same cost per element
when the problem shrank from 1.5 GB to 28 MB. That means they are not memory-bound there, so a
correct agent would disagree with the study.

The reference file therefore records the study's conditions:

- at least 13 threads or ranks per memory domain
- a working set at least 10× the last-level cache

Each case checks them against its own baseline: thread or rank count, peak memory, and the L3
size from `lscpu`. If they are not met, the comparison is still shown in the results, marked "not
scored on this host" with the reasons, and it does not affect the case's score.

To score against the study, run the suite on a node like the study's, with enough threads and a
problem at least 10× that node's cache:

```bash
bottleneck-agent eval --suite suites/lulesh.yaml --env OMP_NUM_THREADS=18
```

On some nodes `-s 120` is not big enough. The working set grows with s³; set the suite's `run`
line to match.

The agent never sees the study or its results. Its prompts contain no application-specific
knowledge, and the reference file is in the hidden suite directory. The study is only used to score
what the agent concludes from its own measurements.

| Case | LULESH (OpenMP, 4 threads) | CoMD |
| --- | --- | --- |
| `negative-control` | unmodified; scored against the roofline study | unmodified, MPI with 4 ranks (the same setup as its load-imbalance case) |
| `load-imbalance` | extra work for the first quarter of elements in `CalcFBHourglassForceForElems`; OpenMP's static schedule gives all of it to thread 0 | rank 0 does extra work at the end of `ljForce` every step; the other ranks wait in the halo exchange (MPI, 4 ranks) |
| `memory-bound` | each element of `CalcKinematicsForElems` (the one hot spot the study finds not memory-bound) follows dependent, scattered reads through a table of 64 entries per element | atoms in `ljForce` follow one dependent chain of scattered reads through a table of 1024 entries per atom |

The memory-bound injections' tables grow with the problem, as a real working set does: 1 GiB at
the default sizes, and 16–32 MiB at small ones. So the working-set experiment can detect them.

Measured on the development VM (4 vCPUs, software events only), against the unmodified build:

| Case | Effect |
| --- | --- |
| LULESH load imbalance (`-s 90 -i 100 -b 0 -c 0`) | wall 26.6 → 37.4 s; `CalcFBHourglassForceForElems` 9% → 19% of samples; per-thread imbalance factor 2.9 (unmodified 1.0) |
| LULESH memory-bound (`-s 120 -i 40 -b 0 -c 0`) | wall 22.7 → 42.2 s; `CalcKinematicsForElems` 5% → 48% of samples; cost per element 0.46× when the table fits in cache (unmodified 0.94×) |
| CoMD load imbalance | rank 0 spends 11.7 s in `force` vs 9.3 s for the fastest rank; the other ranks wait up to 2.4 s in the halo exchange (rank 0: 0.2 s) |
| CoMD memory-bound (30³ cells) | wall 15.3 → 84.0 s; `ljForce` 98% of samples; cost per atom-step 0.25–0.49× when the table fits in cache at 10³ cells (0.85× at 12³; unmodified 1.07–1.13×) |

The suites originally had a "disabled vectorization" case instead of `memory-bound`. It was replaced
because it could not be detected. GCC barely vectorizes either app's hot loops at `-O3`, whether
built for generic x86-64 or for `x86-64-v3` (AVX2). LULESH's hourglass kernel gathers through node
indices, and CoMD's pair loop has data-dependent branches. Disabling vectorization in the hot kernel
changed run time by −2% to +10%, and the profile not at all, so a blind run looked the same as the
unmodified one.

### Suite format

```yaml
name: comd                         # lowercase letters, digits, '-'
target: comd
config: ../configs/comd.yaml       # relative to the suite file
cases:
  - id: negative-control
    mpi: true                      # optional: use the config's MPI variant
    expect: none                   # nothing injected
  - id: load-imbalance
    patch: patches/comd-load-imbalance.patch
    mpi: true
    expect:
      function: ljForce            # a plain name; perf's decorated names (Foo(int) [clone ...]) match it
      bottleneck_types: [load_imbalance]   # any of these counts as the right diagnosis
```

Two optional keys apply to reference-based suites like LULESH's:

```yaml
run: "{binary} -s 120 -i 40 -b 0 -c 0"   # replaces the config's run command for every case
env: {OMP_PROC_BIND: close}              # optional: environment for every case (eval --env overrides)
cases:
  - id: negative-control
    expect: none
    reference: {file: references/lulesh-afzal2024.yaml, variant: openmp}   # unmodified cases only
```

A reference file names its source and URL, and optionally the `conditions` under which it applies
(`min_workers`, `min_working_set_to_llc`). Under `variants`, it maps each function to `class:
memory_bound` or `class: not_memory_bound`, optionally with the measured `intensity`.

The suite is validated before anything runs, and all problems are reported together: keys,
ids, patch paths, bottleneck types from the findings schema, the target config, and an MPI case
without an MPI config.

### Scoring

For each case, `results.json` records:

- **function found:** a finding names the expected function.
- **type found:** a finding names the expected function with one of the expected bottleneck
  types. That finding's rank is recorded.
- **false positives:** findings that report a bottleneck another case of the suite injects (its
  expected function and one of its expected types), in a case that does not inject it. For
  example, `CalcKinematicsForElems` reported as memory-bound in LULESH's negative control or
  load-imbalance case. Other findings are not counted: they may be real properties of the
  unmodified code on this host, such as a memory-sensitive loop or OpenMP spin-waiting.
- **reference agreement** (cases with a `reference`): for each reference function the agent
  reports, its class is compared with the reference's. The class is taken from the agent's
  best-ranked finding for that function:
  - `memory_latency_bound` and `memory_bandwidth_bound` count as memory-bound.
  - `compute_bound`, `poor_vectorization` and `branch_misprediction` count as not memory-bound.
  - Other types, such as `load_imbalance`, make no roofline claim and count as a disagreement.

  The results list each function as agrees, disagrees or not reported.
- **time:** wall-clock time for the case, including the build and all profiling.
- **experiments:** how many experiment artifact sets the agent created, and their names.
- the run's exit status, whether `findings.json` was valid, and every finding's rank, function,
  type and confidence.

A case is **correct** when the run succeeded with valid findings, the injected function and type
were found (not required for a negative control), and there are no false positives. For a case
with a reference, the agent must also report at least one reference function, and every one it
reports must agree. An inconclusive result on an injected case counts as a miss.

`results.json` holds every case plus a summary: cases correct, functions and types found,
reference functions reported and agreeing, cases with false positives, total experiments and total
time. `results.md` presents the same as a table,
plus each case's reported findings. Both are rewritten after every case, so an interrupted suite
keeps its results.

A case that fails is recorded with its error, and the suite continues with the next case. A failed
case is, for example, a patch that no longer applies, a build error, or a crashed run.
`eval` exits 1 if any case errored.

### Keeping the agent blind

- Cases run as `case-1`, `case-2`, …, so neither paths nor prompts say what was injected.
- The source copy has a single neutral commit, so `git log` and `git diff` show nothing.
- The suite directory (expectations and patches), the eval's `results.json` and `results.md`, and
  the other cases' directories are hidden from the agent. Shell commands that name them are
  refused, and `read_file` and `search` skip them. Like the other command checks, this is
  best-effort pattern matching. Each case records `hidden_access_attempts`, and a non-zero count
  means the agent went looking.

`evals/` is git-ignored.

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

- **One agent.** It is one conversation with the model: a loop in `sage/agent.py` sends the
  messages, runs the tool calls the model asks for, and returns the results, until the model
  answers without a tool call. The retry continues the same conversation.
- **Four tools**, defined in `sage/tools.py`:
  - `bash`: runs a command in the run directory, checked by the guard first.
  - `read_file`: reads a text file with line numbers.
  - `search`: a regex search over files.
  - `write_findings`: writes `findings.json`, and nothing else.

  There are no subagents, no web access and no other tools. Each command has a time limit, and
  output is truncated before it goes back to the model.

  These tools are written for SAGE because the OpenAI API, unlike the Claude Agent SDK, has no
  built-in local shell or file tools.
- **Writing.** The only file-writing tool is `write_findings`, and it refuses to write through a
  symlink. `report.md` is always generated by SAGE: one written by the agent is deleted. The shell
  can still create perf output and notes, because experiments need it.
- **Every run is recorded.** The application may only be run through `profile_target.sh`.
  - Direct `perf stat`/`perf record`/`perf top`/`perf trace` and MPI launchers are denied.
  - Running the target binary directly is denied, except `--help`/`-h`.
  - `perf report`, `perf annotate` and `perf script` on existing data are allowed.
  - Artifact sets may only go to `runs/<run-id>/baseline` or `runs/<run-id>/experiments/`.
    Experiments beyond `--max-experiments` are refused.
- **perf only.** The guard (`sage/guard.py`) refuses shell commands that call other profilers
  (gprof, valgrind, VTune, likwid, HPCToolkit, Nsight, strace, `-pg`, and others).
- **Source is read-only.** The same guard refuses commands that would edit, move or delete a file
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
| 1 | preflight failed (no usable perf, compiler or events), or the model API failed (e.g. the model is not allowed for your HokieAI subscription, or the gateway is unreachable) |
| 2 | invalid command line, target or config |
| 3 | application source was modified |
| 4 | `findings.json` is missing or still invalid after one retry; no report was written |
| 5 | the wall-clock limit (`--timeout`) was reached |

For `eval`: 0 when every case ran (whatever its score), 1 if any case errored, 2 for an invalid suite or options.

## Tests

The tests need no HokieAI access, MPI, perf binary, hardware counters, compiler or network.
The OpenAI client is replaced with a fake that returns scripted Chat Completions responses. The
tool calls in those responses run for real in a temporary directory. These tests cover:

- prompt loading and the HokieAI key and `.env` handling
- limits: turns, wall-clock and experiments
- the tool loop, the retry and the transcript
- API errors such as `ModelNotAllowed`
- the tools themselves perf, `mpirun` and the compiler are
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
