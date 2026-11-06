"""Parse perf output and write metadata.json for a run directory (see scripts/profile_target.sh)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

NOT_MEASURED = ("<not supported>", "<not counted>")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH|COOKIE|SESSION|PROXY", re.I)
THREAD_VARS = re.compile(r"(OMP|KMP|GOMP|MKL|OPENBLAS|BLIS|TBB)_\w+")


def parse_perf_stat(text: str) -> dict:
    """Parse the human-readable output of ``perf stat [-r N] -o FILE``."""
    out = {"command": None, "runs": 1, "elapsed_s": None, "elapsed_stddev_s": None,
           "elapsed_variance_pct": None, "user_s": None, "sys_s": None, "events": []}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if m := re.match(r"Performance counter stats for '(.*)'(?: \((\d+) runs\))?:", s):
            out["command"], out["runs"] = m.group(1), int(m.group(2) or 1)
        elif m := re.match(r"([\d.]+)(?: \+- ([\d.]+))? seconds time elapsed(?:\s+\(\s*\+-\s*([\d.]+)%\s*\))?", s):
            out["elapsed_s"] = float(m.group(1))
            out["elapsed_stddev_s"] = float(m.group(2)) if m.group(2) else None
            out["elapsed_variance_pct"] = float(m.group(3)) if m.group(3) else None
        elif m := re.match(r"([\d.]+) seconds (user|sys)$", s):
            out[f"{m.group(2)}_s"] = float(m.group(1))
        elif event := _parse_event(s):
            out["events"].append(event)

    by_name = {e["event"].split(":")[0]: e["value"] for e in out["events"]}
    cycles, instructions = by_name.get("cycles"), by_name.get("instructions")
    out["ipc"] = round(instructions / cycles, 3) if cycles and instructions else None
    return out


def _parse_event(line: str) -> dict | None:
    multiplexed = None
    if m := re.search(r"\s*\((\d+(?:\.\d+)?)%\)$", line):  # share of time the event was counted
        multiplexed, line = float(m.group(1)), line[:m.start()]
    variance = None
    if m := re.search(r"\s*\(\s*\+-\s*([\d.]+)%\s*\)$", line):
        variance, line = float(m.group(1)), line[:m.start()]
    left, _, comment = line.partition("#")
    left = left.strip()

    status = next((n for n in NOT_MEASURED if left.startswith(n)), None)
    if status:
        value, rest = None, left[len(status):].split()
    else:
        first, *rest = left.split() or [""]
        try:
            value = float(first.replace(",", ""))
        except ValueError:
            return None  # not an event line (e.g. a topdown metric heading)
        if value.is_integer() and "." not in first:
            value = int(value)
    if not rest:
        return None
    unit, event = (rest[0], " ".join(rest[1:])) if len(rest) > 1 else (None, rest[0])
    return {"event": event, "value": value, "unit": unit, "supported": status != "<not supported>",
            "counted": status is None, "variance_pct": variance, "multiplexed_pct": multiplexed,
            "comment": " ".join(comment.split()) or None}


def parse_perf_report(text: str, top: int = 10) -> dict:
    """Parse ``perf report --stdio --sort overhead,dso,sym`` output (self time per DSO and symbol)."""
    out = {"event": None, "samples": None, "event_count": None, "lost_samples": None, "rows": 0, "top": []}
    for line in text.splitlines():
        if m := re.match(r"# Samples: (\S+)\s+of event '([^']+)'", line):
            out["samples"], out["event"] = m.group(1), m.group(2)
        elif m := re.match(r"# Event count \(approx\.\): (\d+)", line):
            out["event_count"] = int(m.group(1))
        elif m := re.match(r"# Total Lost Samples: (\d+)", line):
            out["lost_samples"] = int(m.group(1))
        elif m := re.match(r"\s*([\d.]+)%\s+(\S+)\s+\[(.)\]\s+(.+?)\s*$", line):
            out["rows"] += 1
            if len(out["top"]) < top:
                out["top"].append({"self_pct": float(m.group(1)), "dso": m.group(2),
                                   "kernel": m.group(3) == "k", "symbol": m.group(4)})
    return out


def cpu_info() -> dict:
    info = {"arch": platform.machine(), "kernel": platform.release(), "hostname": socket.gethostname(),
            "logical_cpus": os.cpu_count()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for key, name in (("vendor_id", "vendor"), ("model name", "model")):
            if m := re.search(rf"^{key}\s*:\s*(.+)$", cpuinfo, re.M):
                info[name] = m.group(1).strip()
    except OSError:
        pass
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        lscpu = ""
    wanted = {"Socket(s)": "sockets", "Core(s) per socket": "cores_per_socket",
              "Thread(s) per core": "threads_per_core", "NUMA node(s)": "numa_nodes",
              "CPU max MHz": "max_mhz", "L1d cache": "l1d_cache", "L2 cache": "l2_cache",
              "L3 cache": "l3_cache", "Hypervisor vendor": "hypervisor"}
    for line in lscpu.splitlines():
        key, _, value = line.partition(":")
        if key.strip() in wanted:
            info[wanted[key.strip()]] = value.strip()
    return info


def thread_settings(env: dict, ranks: int | None) -> dict:
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = []
    return {
        "cpu_affinity": _ranges(cpus), "cpus_available": len(cpus) or os.cpu_count(),
        "mpi_ranks": ranks,
        "variables": {k: v for k, v in sorted(env.items()) if THREAD_VARS.fullmatch(k)},
    }


def _ranges(nums: list[int]) -> str:
    parts, start = [], None
    for i, n in enumerate(nums):
        if start is None:
            start = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            parts.append(f"{start}" if start == n else f"{start}-{n}")
            start = None
    return ",".join(parts)


def redacted(env: dict) -> dict:
    return {k: ("<redacted>" if SECRET_NAME.search(k) else v) for k, v in sorted(env.items())}


def build_metadata(run_dir: Path, fields: dict, env: dict) -> dict:
    """Assemble metadata.json from the script's ``fields`` and whatever artifacts exist."""
    secs = lambda k: round(int(fields[k]) / 1e9, 3) if fields.get(k) else None  # script passes nanoseconds
    ranks = int(fields["ranks"]) if fields.get("ranks") else None
    read = lambda p: p.read_text(errors="replace") if p.is_file() else None

    stat_text = read(run_dir / "perf-stat.txt")
    stat = parse_perf_stat(stat_text) if stat_text else None
    if ranks:
        reports = {d.name.removeprefix("rank-"): parse_perf_report(read(d / "perf-report.txt") or "")
                   for d in sorted(run_dir.glob("rank-*")) if d.is_dir()}
        report = None
    else:
        reports = None
        report_text = read(run_dir / "perf-report.txt")
        report = parse_perf_report(report_text) if report_text else None

    build_path = Path(fields["build_info"]) if fields.get("build_info") else None
    build = json.loads(build_path.read_text()) if build_path and build_path.is_file() else None
    artifacts = [{"path": str(p.relative_to(run_dir)), "bytes": p.stat().st_size}
                 for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "metadata.json"]

    return {
        "run_id": fields.get("run_id"), "label": fields.get("label") or None,
        "target": fields.get("target"), "variant": fields.get("variant"),
        "status": fields.get("status"), "error": fields.get("error") or None,
        "started_at": fields.get("started_at"), "finished_at": fields.get("finished_at"),
        "command_line": fields.get("command_line"),
        "run_command": fields.get("run_command"),
        "env_overrides": dict(kv.split("=", 1) for kv in (fields.get("env_overrides") or "").splitlines() if "=" in kv),
        "mpi": {"ranks": ranks, "launcher": fields.get("launcher")} if ranks else None,
        "wall_clock": {
            "total_s": secs("total_ns"), "perf_stat_s": secs("stat_ns"), "perf_record_s": secs("record_ns"),
            "app_mean_s": stat and stat["elapsed_s"], "app_stddev_s": stat and stat["elapsed_stddev_s"],
            "app_variance_pct": stat and stat["elapsed_variance_pct"], "repeats": stat and stat["runs"],
        },
        "perf": {
            "version": fields.get("perf_version"),
            "commands": {k: fields.get(f"{k}_command") for k in ("stat", "record", "report")},
            "stat": stat, "report": report, "rank_reports": reports,
        },
        "build": build,
        "host": cpu_info(),
        "threads": thread_settings(env, ranks),
        "environment": redacted(env),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sage.artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    meta = sub.add_parser("metadata", help="write RUN_DIR/metadata.json")
    meta.add_argument("run_dir", type=Path)
    meta.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    for name in ("stat", "report"):
        p = sub.add_parser(f"parse-{name}", help=f"parse a perf {name} text file and print JSON")
        p.add_argument("file", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "metadata":
        fields = dict(item.split("=", 1) for item in args.set)
        data = build_metadata(args.run_dir, fields, dict(os.environ))
        (args.run_dir / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
    else:
        parse = parse_perf_stat if args.cmd == "parse-stat" else parse_perf_report
        print(json.dumps(parse(args.file.read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
