#!/usr/bin/env bash
# Check out a target's pinned upstream commit and build it with the config's compiler and flags.
# Records the commit, compiler version, flags and tools in .work/<target>/build-info.json and
# the full build output in .work/<target>/build.log (build-info-mpi.json and build-mpi.log for
# the MPI variant). Upstream source files are never edited:
# the build fails if any tracked file differs from the pinned commit before or after building.
# Exit status: 0 = built, 1 = checkout or build failed, 2 = bad arguments or config.
#
# usage: scripts/build_target.sh <lulesh|comd> [--mpi] [--config PATH] [--work DIR] [--clean]
#   --mpi          build the config's MPI variant (its mpi: section)
#   --config PATH  target config (default: configs/<target>.yaml in this repository)
#   --work DIR     root for checkouts and builds (default: .work)
#   --clean        delete DIR/<target> first and build from scratch
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "build_target: $*" >&2; exit 1; }

target="" config="" work_root=.work clean=false mpi=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --config) config=${2:?--config needs a path}; shift 2 ;;
    --work) work_root=${2:?--work needs a directory}; shift 2 ;;
    --clean) clean=true; shift ;;
    --mpi) mpi=--mpi; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "build_target: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) [[ -z $target ]] || { echo "build_target: unexpected argument: $1" >&2; exit 2; }; target=$1; shift ;;
  esac
done
[[ -n $target ]] || { usage >&2; exit 2; }
config=${config:-$repo/configs/$target.yaml}

# Validate the config with the same code the CLI uses, and load it as shell variables.
PYTHON=${PYTHON:-python3}
vars=$(PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m sage.config "$config" "$target" $mpi) || exit 2
eval "$vars"
export SAGE_COMPILER SAGE_FLAGS

mkdir -p "$work_root"
work="$(cd "$work_root" && pwd)/$target"
sfx=${mpi:+-mpi}
src=$work/src log=$work/build$sfx.log info_file=$work/build-info$sfx.json
if $clean; then rm -rf "$work"; fi
mkdir -p "$work"

# --- checkout ---
if [[ ! -d $src/.git ]]; then
  if [[ -e $src && -n $(ls -A "$src") ]]; then die "$src exists but is not a git checkout; rerun with --clean"; fi
  echo "build_target: cloning $SAGE_REPO_URL" >&2
  git clone --quiet "$SAGE_REPO_URL" "$src" || die "git clone of $SAGE_REPO_URL failed"
fi
origin=$(git -C "$src" remote get-url origin)
[[ $origin == "$SAGE_REPO_URL" ]] || die "$src was cloned from $origin but the config says $SAGE_REPO_URL; rerun with --clean"
dirty=$(git -C "$src" status --porcelain --untracked-files=no)
[[ -z $dirty ]] || die "$src has local changes to tracked upstream files (upstream source must not be edited); rerun with --clean:
$dirty"
if [[ $(git -C "$src" rev-parse HEAD) != "$SAGE_COMMIT" ]]; then
  git -C "$src" cat-file -e "$SAGE_COMMIT^{commit}" 2>/dev/null || git -C "$src" fetch --quiet origin || die "git fetch failed"
  git -C "$src" checkout --quiet --detach "$SAGE_COMMIT" || die "cannot check out $SAGE_COMMIT from $SAGE_REPO_URL"
fi
commit=$(git -C "$src" rev-parse HEAD)
[[ $commit == "$SAGE_COMMIT" ]] || die "checkout is at $commit, expected $SAGE_COMMIT"

# --- build ---
command -v "$SAGE_COMPILER" >/dev/null || die "compiler '$SAGE_COMPILER' not found"
compiler_version=$("$SAGE_COMPILER" --version 2>&1 | head -n1)
rm -f "$info_file"  # only a successful build leaves a record
{
  echo "# target:   $SAGE_TARGET ($SAGE_VARIANT)"
  echo "# commit:   $commit ($SAGE_REPO_URL)"
  echo "# compiler: $SAGE_COMPILER ($compiler_version)"
  echo "# flags:    $SAGE_FLAGS"
  echo "# started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log"
for cmd in "${SAGE_BUILD[@]}"; do
  echo "build_target: $cmd" >&2
  printf '\n$ %s\n' "$cmd" >> "$log"
  rc=0
  (cd "$work" && bash -c "$cmd") >> "$log" 2>&1 || rc=$?
  if (( rc != 0 )); then
    echo "build_target: build failed (exit $rc) running: $cmd" >&2
    tail -n 15 "$log" | sed 's/^/  | /' >&2
    echo "build_target: full log: $log" >&2
    exit 1
  fi
done

binary=$work/$SAGE_BINARY
[[ -f $binary && -x $binary ]] || die "build finished but $binary is not an executable file; see $log"
dirty=$(git -C "$src" status --porcelain --untracked-files=no)
[[ -z $dirty ]] || die "the build modified tracked upstream files; see $log:
$dirty"

# --- build record ---
tool_version() { command -v "$1" >/dev/null && "$1" --version 2>&1 | head -n1 || true; }
TARGET=$SAGE_TARGET REPO_URL=$SAGE_REPO_URL COMMIT=$commit COMPILER=$SAGE_COMPILER \
COMPILER_PATH=$(command -v "$SAGE_COMPILER") COMPILER_VERSION=$compiler_version FLAGS=$SAGE_FLAGS \
BINARY=$binary LOG=$log CONFIG=$(cd "$(dirname "$config")" && pwd)/$(basename "$config") \
CMAKE_VERSION=$(tool_version cmake) MAKE_VERSION=$(tool_version make) OUT=$info_file \
VARIANT=$SAGE_VARIANT LAUNCHER=$SAGE_LAUNCHER RANKS=$SAGE_RANKS MPIRUN_VERSION=${mpi:+$(tool_version mpirun)} \
"$PYTHON" - "${SAGE_BUILD[@]}" <<'EOF'
import json, os, sys, time
e = os.environ
info = {
    "target": e["TARGET"], "variant": e["VARIANT"], "repo_url": e["REPO_URL"], "commit": e["COMMIT"],
    "compiler": e["COMPILER"], "compiler_path": e["COMPILER_PATH"],
    "compiler_version": e["COMPILER_VERSION"], "flags": e["FLAGS"],
    "build_commands": sys.argv[1:], "binary": e["BINARY"], "build_log": e["LOG"],
    "config": e["CONFIG"], "tools": {"cmake": e["CMAKE_VERSION"] or None, "make": e["MAKE_VERSION"] or None},
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
if e["VARIANT"] == "mpi":
    info["mpi"] = {"launcher": e["LAUNCHER"], "default_ranks": int(e["RANKS"]), "mpirun": e["MPIRUN_VERSION"] or None}
with open(e["OUT"], "w") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
EOF
echo "build_target: built $binary ($commit, $compiler_version, $SAGE_FLAGS)" >&2
echo "build_target: build info: $info_file" >&2
