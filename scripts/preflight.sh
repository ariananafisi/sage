#!/usr/bin/env bash
# Check that this machine can build the targets and profile them with perf.
# Results go to a JSON file (default .work/preflight.json); a summary goes to stderr.
# Exit status: 0 = usable (possibly with warnings), 1 = a required check failed.
#
# usage: scripts/preflight.sh [--output PATH]
set -uo pipefail

out=.work/preflight.json
while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output) out=${2:?--output needs a path}; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "preflight: unknown argument: $1" >&2; exit 2 ;;
  esac
done

warnings=() errors=()
warn() { warnings+=("$1"); }
fail() { errors+=("$1"); }
first_line() { "$@" 2>&1 | head -n1; }

# --- host and CPU ---
arch=$(uname -m) kernel=$(uname -r)
cpu_vendor=$(awk -F': ' '/^vendor_id/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)
cpu_model=$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null)
cpus=$(nproc 2>/dev/null || echo "")
is_root=false; [[ $(id -u) -eq 0 ]] && is_root=true

# --- compilers ---
compilers=""
for c in gcc g++ clang clang++; do
  command -v "$c" >/dev/null && compilers+="$c=$(first_line "$c" --version)"$'\n'
done
if ! command -v gcc >/dev/null && ! command -v clang >/dev/null; then fail "no C compiler found (need gcc or clang)"; fi
if ! command -v g++ >/dev/null && ! command -v clang++ >/dev/null; then fail "no C++ compiler found (need g++ or clang++)"; fi

# --- perf ---
perf_ok=false perf_version=""
if ! command -v perf >/dev/null; then
  fail "perf not found on PATH (install linux-tools / linux-perf / perf for your distro)"
elif perf_version=$(perf --version 2>&1) && [[ $perf_version == "perf version"* ]]; then
  perf_ok=true
else
  # Typically Ubuntu's wrapper refusing to run because the kernel version has no matching package.
  real=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | tail -n1)
  fail "perf is on PATH but does not run: $(head -n1 <<<"$perf_version")${real:+ (a perf binary exists at $real; put it first on PATH)}"
  perf_version=""
fi

# --- perf_event_paranoid ---
paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "")
if [[ -z $paranoid ]]; then
  warn "cannot read /proc/sys/kernel/perf_event_paranoid"
elif ! $is_root && (( paranoid >= 2 )); then
  warn "perf_event_paranoid=$paranoid: unprivileged users get user-space samples only (lower it with: sudo sysctl kernel.perf_event_paranoid=1)"
fi

# --- events ---
# perf stat -x, prints "<value>,<unit>,<event>,..." and a value of "<not supported>" or
# "<not counted>" when an event cannot be measured, often with exit status 0.
counted() {  # counted <event-name> <perf stat csv>
  awk -F, -v ev="$1" '$1 ~ /^[0-9.]+$/ && $3 ~ ("(^|/)" ev "(/|:|$)") {found=1} END {exit !found}' <<<"$2"
}
cycles=false instructions=false software=false topdown=false
if $perf_ok; then
  hw=$(perf stat -x, -e cycles,instructions -- true 2>&1)
  counted cycles "$hw" && cycles=true
  counted instructions "$hw" && instructions=true
  sw=$(perf stat -x, -e task-clock -- true 2>&1)
  counted task-clock "$sw" && software=true
  td=$(perf stat -x, -M TopdownL1 -- true 2>&1) && ! grep -qiE "not supported|not counted|cannot find" <<<"$td" && topdown=true

  if ! $software && ! $cycles; then
    fail "perf cannot measure any events (check perf_event_paranoid or run as root): $(head -n1 <<<"$sw")"
  elif ! $cycles || ! $instructions; then
    warn "only software events are available (cycles: $cycles, instructions: $instructions); IPC and cache/branch counters cannot be measured, sampling falls back to cpu-clock"
  fi
  $topdown || warn "topdown metrics (perf stat -M TopdownL1) are not supported on this machine"
fi

# --- write JSON ---
mkdir -p "$(dirname "$out")"
join() { local IFS=$'\n'; echo "$*"; }
ARCH=$arch KERNEL=$kernel CPU_VENDOR=$cpu_vendor CPU_MODEL=$cpu_model CPUS=$cpus IS_ROOT=$is_root \
COMPILERS=$compilers PERF_OK=$perf_ok PERF_VERSION=$perf_version PARANOID=$paranoid \
CYCLES=$cycles INSTRUCTIONS=$instructions SOFTWARE=$software TOPDOWN=$topdown \
WARNINGS=$(join "${warnings[@]}") ERRORS=$(join "${errors[@]}") OUT=$out \
"${PYTHON:-python3}" - <<'EOF' || { echo "preflight: could not write $out" >&2; exit 1; }
import json, os, time
e = os.environ
flag = lambda k: e[k] == "true"
lines = lambda k: [l for l in e[k].split("\n") if l]
result = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "ok": not lines("ERRORS"),
    "host": {"arch": e["ARCH"], "kernel": e["KERNEL"], "cpu_vendor": e["CPU_VENDOR"],
             "cpu_model": e["CPU_MODEL"], "cpus": int(e["CPUS"]) if e["CPUS"] else None,
             "root": flag("IS_ROOT")},
    "compilers": dict(l.split("=", 1) for l in lines("COMPILERS")),
    "perf": {"available": flag("PERF_OK"), "version": e["PERF_VERSION"] or None},
    "perf_event_paranoid": int(e["PARANOID"]) if e["PARANOID"] else None,
    "events": {k.lower(): flag(k) for k in ("CYCLES", "INSTRUCTIONS", "SOFTWARE", "TOPDOWN")},
    "warnings": lines("WARNINGS"),
    "errors": lines("ERRORS"),
}
with open(e["OUT"], "w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
EOF

{
  echo "preflight: ${cpu_model:-unknown CPU} ($cpus CPUs), kernel $kernel"
  echo "  perf: ${perf_version:-unavailable} | paranoid: ${paranoid:-?} | cycles: $cycles instructions: $instructions software: $software topdown: $topdown"
  for w in "${warnings[@]}"; do echo "  warning: $w"; done
  for x in "${errors[@]}"; do echo "  error: $x"; done
  echo "  results: $out"
} >&2
(( ${#errors[@]} == 0 ))
