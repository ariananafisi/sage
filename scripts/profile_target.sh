#!/usr/bin/env bash
# Create a complete perf artifact set for one run of a target, in <runs>/<run-id>/:
#   metadata.json     what ran, where and how: command, timings, perf summary, build, CPU,
#                     thread settings, environment (secret-looking variables redacted)
#   stdout.log        application output (all runs, one header per step)
#   stderr.log        application and perf messages
#   perf-stat.txt     perf stat -d, 3 repeats
#   perf.data         perf record -g
#   perf-report.txt   self time per DSO and symbol, highest first
# MPI (--ranks N) uses the config's mpi section: perf stat covers the whole job, and each rank
# gets rank-<r>/perf.data and rank-<r>/perf-report.txt instead of the top-level pair.
# The target must already be built (scripts/build_target.sh <target> [--mpi]).
# Exit status: 0 = all artifacts written, 1 = a step failed (metadata.json says which),
# 2 = bad arguments or config.
#
# usage: scripts/profile_target.sh <target> <run-id> [--label TEXT] [--cmd COMMAND] [--env NAME=VALUE]... [--ranks N]
#                                  [--config PATH] [--work DIR] [--runs DIR]
#   --label TEXT   free-form note stored in metadata.json
#   --cmd COMMAND  run COMMAND instead of the config's run command; {binary} expands to the binary.
#                  It may not set environment variables itself; use --env so they are recorded
#   --env NAME=VALUE  set an environment variable for the application (repeatable), e.g.
#                  OMP_NUM_THREADS=1; it overrides the config's env and is recorded in metadata.json
#   --ranks N      MPI run with N ranks (needs --cmd unless N is the config's mpi.ranks)
#   --config PATH  target config (default: configs/<target>.yaml in this repository)
#   --work DIR     checkouts and builds (default: .work)
#   --runs DIR     parent of the run directory (default: runs)
set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }
bad() { echo "profile_target: $*" >&2; exit 2; }
die() { echo "profile_target: $*" >&2; exit 1; }
q() { printf '%q' "$1"; }
now() { date +%s%N; }  # nanoseconds

target="" run_id="" label="" cmd="" ranks="" config="" work_root=.work runs=runs overrides=()
command_line=$(printf '%q ' "$0" "$@"); command_line=${command_line% }
while [[ $# -gt 0 ]]; do
  case $1 in
    --label) label=${2?--label needs text}; shift 2 ;;
    --cmd) cmd=${2:?--cmd needs a command}; shift 2 ;;
    --env) overrides+=("${2:?--env needs NAME=VALUE}"); shift 2 ;;
    --ranks) ranks=${2:?--ranks needs a number}; shift 2 ;;
    --config) config=${2:?--config needs a path}; shift 2 ;;
    --work) work_root=${2:?--work needs a directory}; shift 2 ;;
    --runs) runs=${2:?--runs needs a directory}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; bad "unknown option: $1" ;;
    *) if [[ -z $target ]]; then target=$1; elif [[ -z $run_id ]]; then run_id=$1; else bad "unexpected argument: $1"; fi
       shift ;;
  esac
done
[[ -n $target && -n $run_id ]] || { usage >&2; exit 2; }
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || bad "run id must be letters, digits, '.', '_' or '-' (got '$run_id')"
[[ -z $ranks || $ranks =~ ^[1-9][0-9]*$ ]] || bad "--ranks must be a positive integer (got '$ranks')"
config=${config:-$repo/configs/$target.yaml}

PYTHON=${PYTHON:-python3}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
vars=$("$PYTHON" -m sage.config "$config" "$target" ${ranks:+--mpi}) || exit 2
eval "$vars"
# Environment overrides: applied after the config's env, so the application really runs with them,
# and metadata.json (which records the environment the runs inherit) shows the values used.
for kv in "${overrides[@]}"; do
  [[ $kv =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || bad "--env needs NAME=VALUE (got '$kv')"
  [[ ${BASH_REMATCH[1]} != SAGE_* ]] || bad "--env may not set SAGE_ variables"
  export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
done
if [[ $cmd =~ ^[[:space:]]*(env[[:space:]]|[A-Za-z_][A-Za-z0-9_]*=) ]]; then
  bad "--cmd may not set environment variables (they would not be recorded); use --env NAME=VALUE instead"
fi
if [[ -n $ranks && $ranks != "$SAGE_RANKS" && -z $cmd ]]; then
  bad "the config's MPI run command is written for $SAGE_RANKS ranks; pass --cmd for $ranks ranks"
fi

# --- preconditions (nothing is written until these pass) ---
work="$(mkdir -p "$work_root" && cd "$work_root" && pwd)/$target"
binary=$work/$SAGE_BINARY
build_info=$work/build-info${ranks:+-mpi}.json
[[ -x $binary ]] || die "$binary is not built; run: scripts/build_target.sh $target${ranks:+ --mpi}"
command -v perf >/dev/null || die "perf not found on PATH (see scripts/preflight.sh)"
perf_version=$(perf --version 2>&1) && [[ $perf_version == "perf version"* ]] \
  || die "perf does not run: $(head -n1 <<<"$perf_version") (see scripts/preflight.sh)"
run_cmd=${cmd:-$SAGE_RUN}
run_cmd=${run_cmd//\{binary\}/$binary}
launcher=""
if [[ -n $ranks ]]; then
  launcher=${SAGE_LAUNCHER//\{ranks\}/$ranks}
  command -v "${launcher%% *}" >/dev/null || die "MPI launcher '${launcher%% *}' not found on PATH"
fi
mkdir -p "$runs"
run_dir="$(cd "$runs" && pwd)/$run_id"
[[ ! -e $run_dir ]] || bad "$run_dir already exists; choose another run id"
mkdir "$run_dir"

# --- bookkeeping ---
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) t0=$(now) stat_ns="" record_ns=""
stat_cmd="" record_cmd="" report_cmd=""
meta() {  # meta <status> [error]
  "$PYTHON" -m sage.artifacts metadata "$run_dir" \
    --set "run_id=$run_id" --set "label=$label" --set "target=$target" --set "variant=$SAGE_VARIANT" \
    --set "status=$1" --set "error=${2:-}" --set "started_at=$started_at" \
    --set "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" --set "command_line=$command_line" \
    --set "run_command=$run_cmd" --set "launcher=$launcher" --set "ranks=$ranks" \
    --set "env_overrides=$(IFS=$'\n'; echo "${overrides[*]}")" \
    --set "stat_command=$stat_cmd" --set "record_command=$record_cmd" --set "report_command=$report_cmd" \
    --set "perf_version=$perf_version" --set "build_info=$build_info" \
    --set "stat_ns=$stat_ns" --set "record_ns=$record_ns" --set "total_ns=$(( $(now) - t0 ))"
}
fail() {  # fail <message>: record the failure, show the evidence, stop
  meta failed "$1" || echo "profile_target: warning: could not write $run_dir/metadata.json" >&2
  echo "profile_target: $1" >&2
  if [[ -s $run_dir/stderr.log ]]; then
    tail -n 15 "$run_dir/stderr.log" | sed 's/^/  | /' >&2
  fi
  echo "profile_target: stderr: $run_dir/stderr.log" >&2
  echo "profile_target: metadata: $run_dir/metadata.json" >&2
  exit 1
}
step() {  # step <name> <command>: run in the run directory, logging both streams with a header
  printf '\n===== %s: %s\n' "$1" "$2" | tee -a "$run_dir/stdout.log" >> "$run_dir/stderr.log"
  (cd "$run_dir" && bash -c "$2") >> "$run_dir/stdout.log" 2>> "$run_dir/stderr.log"
}
: > "$run_dir/stdout.log"; : > "$run_dir/stderr.log"
echo "profile_target: $run_dir" >&2

# --- 1. perf stat, 3 repeats ---
stat_cmd="perf stat -r 3 -d -o $(q "$run_dir/perf-stat.txt") -- ${launcher:+$launcher }$run_cmd"
echo "profile_target: perf stat (3 repeats)" >&2
t=$(now); step "perf stat" "$stat_cmd"; rc=$?; stat_ns=$(( $(now) - t ))
(( rc == 0 )) || fail "perf stat failed (exit $rc): the application or perf returned an error"
grep -q "Performance counter stats" "$run_dir/perf-stat.txt" 2>/dev/null \
  || fail "perf stat exited 0 but wrote no counter statistics to perf-stat.txt"

# --- 2. perf record ---
if [[ -z $ranks ]]; then
  record_cmd="perf record -g -o $(q "$run_dir/perf.data") -- $run_cmd"
else
  # Each rank records into its own directory; the rank number comes from the MPI runtime.
  wrapper=$run_dir/perf-rank-wrapper.sh
  cat > "$wrapper" <<EOF
#!/usr/bin/env bash
r=\${OMPI_COMM_WORLD_RANK:-\${PMIX_RANK:-\${PMI_RANK:-\${MV2_COMM_WORLD_RANK:-\${SLURM_PROCID:-}}}}}
[[ -n \$r ]] || { echo "rank-wrapper: cannot determine the MPI rank of this process" >&2; exit 97; }
mkdir -p $(q "$run_dir")/rank-\$r
exec perf record -g -o $(q "$run_dir")/rank-\$r/perf.data -- "\$@"
EOF
  chmod +x "$wrapper"
  record_cmd="$launcher $(q "$wrapper") $run_cmd"
fi
echo "profile_target: perf record" >&2
t=$(now); step "perf record" "$record_cmd"; rc=$?; record_ns=$(( $(now) - t ))
(( rc == 0 )) || fail "perf record failed (exit $rc): the application or perf returned an error"

# --- 3. perf report: self time, grouped by DSO and symbol ---
report_cmd="perf report --stdio --no-children --sort overhead,dso,sym -g none"
report() {  # report <dir>: perf.data in <dir> -> perf-report.txt in <dir>
  [[ -s $1/perf.data ]] || fail "perf record wrote no $1/perf.data"
  perf report -i "$1/perf.data" --stdio --no-children --sort overhead,dso,sym -g none \
    > "$1/perf-report.txt" 2>> "$run_dir/stderr.log" || fail "perf report failed for $1/perf.data (exit $?)"
  grep -qE '^ +[0-9.]+%' "$1/perf-report.txt" || fail "perf report found no samples in $1/perf.data"
}
echo "profile_target: perf report" >&2
if [[ -z $ranks ]]; then
  report "$run_dir"
else
  found=$(find "$run_dir" -maxdepth 1 -type d -name 'rank-*' | wc -l)
  (( found == ranks )) || fail "expected perf data for $ranks ranks, found $found rank directories"
  for ((r = 0; r < ranks; r++)); do report "$run_dir/rank-$r"; done
fi

meta ok || die "could not write $run_dir/metadata.json"
echo "profile_target: done: $run_dir (metadata.json, perf-stat.txt, ${ranks:+rank-*/}perf.data, ${ranks:+rank-*/}perf-report.txt, stdout.log, stderr.log)" >&2
