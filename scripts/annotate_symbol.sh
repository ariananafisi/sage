#!/usr/bin/env bash
# Save perf annotate output (source and disassembly, with sample percentages) for one symbol of a
# run made by scripts/profile_target.sh. Writes <run>/annotate/<symbol>.txt, or
# <run>/rank-<r>/annotate/<symbol>.txt for an MPI run.
# Exit status: 0 = written, 1 = perf failed or the symbol has no samples, 2 = bad arguments.
#
# usage: scripts/annotate_symbol.sh <run-id> <symbol> [--rank R] [--runs DIR] [--output PATH]
#   --rank R       rank to annotate (required for MPI runs)
#   --runs DIR     parent of the run directory (default: runs)
#   --output PATH  where to write (default: see above)
set -uo pipefail

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }
bad() { echo "annotate_symbol: $*" >&2; exit 2; }
die() { echo "annotate_symbol: $*" >&2; exit 1; }

run_id="" symbol="" rank="" runs=runs output=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --rank) rank=${2:?--rank needs a number}; shift 2 ;;
    --runs) runs=${2:?--runs needs a directory}; shift 2 ;;
    --output) output=${2:?--output needs a path}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; bad "unknown option: $1" ;;
    *) if [[ -z $run_id ]]; then run_id=$1; elif [[ -z $symbol ]]; then symbol=$1; else bad "unexpected argument: $1"; fi
       shift ;;
  esac
done
[[ -n $run_id && -n $symbol ]] || { usage >&2; exit 2; }
run_dir=$runs/$run_id
[[ -d $run_dir ]] || bad "run directory $run_dir not found"
if [[ -n $rank ]]; then
  dir=$run_dir/rank-$rank
  [[ -d $dir ]] || bad "$run_dir has no rank $rank"
elif compgen -G "$run_dir/rank-*/" >/dev/null; then
  bad "$run_id is an MPI run; pass --rank (ranks: $(cd "$run_dir" && ls -d rank-*/ | sed 's/rank-//; s/\/$//' | sort -n | tr '\n' ' '))"
else
  dir=$run_dir
fi
[[ -s $dir/perf.data ]] || die "no perf.data in $dir"
command -v perf >/dev/null || die "perf not found on PATH"

output=${output:-$dir/annotate/$(tr -c 'A-Za-z0-9._-' '_' <<<"$symbol" | sed 's/_$//').txt}
mkdir -p "$(dirname "$output")"
tmp=$output.tmp
perf annotate -i "$dir/perf.data" --stdio -s "$symbol" > "$tmp" 2> "$tmp.err"
rc=$?
if (( rc != 0 )); then
  err=$(head -n 5 "$tmp.err"); rm -f "$tmp" "$tmp.err"
  die "perf annotate failed (exit $rc): $err"
fi
# perf annotate exits 0 with no output when the symbol has no samples.
if ! grep -q "Percent" "$tmp"; then
  rm -f "$tmp" "$tmp.err"
  die "no samples for symbol '$symbol' in $dir/perf.data; see $dir/perf-report.txt for symbol names"
fi
rm -f "$tmp.err"
mv "$tmp" "$output"
echo "annotate_symbol: wrote $output" >&2
echo "$output"
