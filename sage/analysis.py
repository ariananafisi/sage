"""Per-function analysis of an artifact set, and comparison of two, for diagnosing bottlenecks.

    python -m sage.analysis summary RUN_DIR      write RUN_DIR/functions.json and RUN_DIR/summary.md
    python -m sage.analysis compare A B [--work WA WB]
                                                 per-function change from artifact set A to B
    python -m sage.analysis maxrss OUT -- CMD... run CMD, write the peak RSS (KiB) of its largest process

functions.json splits each function's time over the workers that ran it: the threads of the process,
or the MPI ranks (each rank's perf.data). Uneven work shows as an imbalance factor (max / mean over
workers). compare puts two artifact sets side by side, optionally per unit of work, so an experiment
can be read function by function (e.g. does a function get cheaper per element when the data fits
in cache?).
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import sys
from pathlib import Path

from .artifacts import parse_perf_stat

# Libraries whose time is mostly a parallel runtime waiting or spinning, not application work.
RUNTIME_DSO = re.compile(r"libgomp|libiomp|libomp\b|libmpi|libopen-pal|libopen-rte|libucp|libucs|libuct|libfabric"
                         r"|libpsm|libmlx|mca_|libpmix")
ROW = re.compile(r"^\s*([\d.]+)%\s+(\d+):(\S+)\s+(\S+)\s+\[(.)\]\s+(.+?)\s*$")
TOP_FUNCTIONS = 40
MIN_WORKER_SHARE = 1.0  # % of all samples: smaller threads (helpers, launchers) are not counted as workers


def parse_worker_report(text: str) -> dict:
    """Parse `perf report --stdio --no-children --sort pid,dso,sym -g none`."""
    out = {"event": None, "event_count": None, "rows": []}
    for line in text.splitlines():
        if m := re.match(r"# Samples: \S+\s+of event '([^']+)'", line):
            out["event"] = m.group(1)
        elif m := re.match(r"# Event count \(approx\.\): (\d+)", line):
            out["event_count"] = int(m.group(1))
        elif m := ROW.match(line):
            out["rows"].append({"pct": float(m.group(1)), "tid": m.group(2), "comm": m.group(3), "dso": m.group(4),
                                "kernel": m.group(5) == "k", "symbol": m.group(6)})
    return out


def unit_of(event: str | None) -> tuple[str, float]:
    """Clock events count nanoseconds: report seconds. Anything else is reported in its own units."""
    base = (event or "").split(":")[0]
    return ("s", 1e-9) if base in ("task-clock", "cpu-clock") else (f"{base or 'event'} counts", 1.0)


def function_table(reports: dict[str, dict], kind: str) -> dict:
    """Combine parsed worker reports into per-function amounts per worker.

    ``reports`` maps a source (a rank, or "" for a single process) to parse_worker_report() output.
    For threads, each thread id with enough samples is a worker; for ranks, each rank is one worker.
    """
    event = next((r["event"] for r in reports.values() if r["event"]), None)
    unit, scale = unit_of(event)
    funcs: dict[tuple[str, str], dict] = {}
    worker_totals: dict[str, float] = {}
    total = 0.0
    for source, report in reports.items():
        count = (report["event_count"] or 0) * scale
        for row in report["rows"]:
            amount = row["pct"] / 100 * count
            worker = f"rank {source}" if kind == "rank" else f"thread {row['tid']}"
            total += amount
            worker_totals[worker] = worker_totals.get(worker, 0.0) + amount
            f = funcs.setdefault((row["dso"], row["symbol"]), {
                "symbol": row["symbol"], "dso": row["dso"], "kernel": row["kernel"],
                "runtime": bool(RUNTIME_DSO.search(row["dso"])), "amount": 0.0, "per_worker": {}})
            f["amount"] += amount
            f["per_worker"][worker] = f["per_worker"].get(worker, 0.0) + amount
    workers = sorted((w for w, a in worker_totals.items() if kind == "rank" or total and a / total * 100 >= MIN_WORKER_SHARE),
                     key=_worker_key)
    rows = sorted(funcs.values(), key=lambda f: -f["amount"])[:TOP_FUNCTIONS]
    for f in rows:
        values = [f["per_worker"].get(w, 0.0) for w in workers]
        mean = sum(values) / len(values) if values else 0.0
        f["imbalance"] = round(max(values) / mean, 2) if mean > 0 and len(values) > 1 else None
        f["share_pct"] = round(f["amount"] / total * 100, 2) if total else 0.0
        f["amount"] = round(f["amount"], 4)
        f["per_worker"] = {w: round(f["per_worker"].get(w, 0.0), 4) for w in workers}
    runtime = sum(f["amount"] for f in funcs.values() if f["runtime"])
    runtime_per_worker = {w: round(sum(f["per_worker"].get(w, 0.0) for f in funcs.values() if f["runtime"]), 4)
                          for w in workers}
    kernel = sum(f["amount"] for f in funcs.values() if f["kernel"])
    return {"event": event, "unit": unit, "worker_kind": kind, "workers": workers,
            "worker_totals": {w: round(worker_totals[w], 4) for w in workers}, "total": round(total, 4),
            "runtime_share_pct": round(runtime / total * 100, 2) if total else 0.0,
            "kernel_share_pct": round(kernel / total * 100, 2) if total else 0.0,
            "runtime_per_worker": runtime_per_worker, "functions": rows}


def _worker_key(worker: str):
    kind, _, ident = worker.partition(" ")
    return (kind, int(ident) if ident.isdigit() else ident)


def worker_report(perf_data: Path) -> str:
    r = subprocess.run(["perf", "report", "-i", str(perf_data), "--stdio", "--no-children",
                        "--sort", "pid,dso,sym", "-g", "none"], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"perf report failed for {perf_data}: {r.stderr.strip()[:300]}")
    return r.stdout


def functions_for(run_dir: Path) -> dict:
    ranks = sorted((d for d in run_dir.glob("rank-*") if (d / "perf.data").is_file()),
                   key=lambda d: int(d.name.removeprefix("rank-")))
    if ranks:
        return function_table({d.name.removeprefix("rank-"): parse_worker_report(worker_report(d / "perf.data"))
                               for d in ranks}, "rank")
    return function_table({"": parse_worker_report(worker_report(run_dir / "perf.data"))}, "thread")


def short(symbol: str) -> str:
    """Drop the argument list for readability: 'Foo(Domain&, int) [clone ._omp_fn.0]' -> 'Foo(…) [clone ._omp_fn.0]'."""
    return re.sub(r"\((?:[^()]|\([^()]*\))*\)", "(…)", symbol, count=1)


def _num(x, unit: str) -> str:
    return "-" if x is None else (f"{x:.2f}" if unit == "s" else f"{x:.3g}")


def render_summary(meta: dict, table: dict, peak_rss_kb: int | None) -> str:
    wall, perf, host = meta.get("wall_clock") or {}, meta.get("perf") or {}, meta.get("host") or {}
    events = (perf.get("stat") or {}).get("events", [])
    unit, kind = table["unit"], table["worker_kind"]
    lines = [f"# Artifact set {meta.get('run_id')}" + (f": {meta['label']}" if meta.get("label") else ""), "",
             f"- Command: `{meta.get('run_command')}`" + (f" via `{meta['mpi']['launcher']}`" if meta.get("mpi") else "")]
    if meta.get("env_overrides"):
        lines.append("- Environment changed for this run: " + ", ".join(f"{k}={v}" for k, v in meta["env_overrides"].items()))
    if wall.get("app_mean_s") is not None:
        spread = f" ± {wall['app_variance_pct']}%" if wall.get("app_variance_pct") is not None else ""
        lines.append(f"- Wall-clock: {wall['app_mean_s']} s{spread}, mean of {wall.get('repeats')} runs under perf stat")
    if peak_rss_kb:
        lines.append(f"- Peak memory of the largest process: {peak_rss_kb / 1024:.0f} MiB "
                     f"(last-level cache: {host.get('l3_cache', 'unknown')})")
    if events:
        counted = [e for e in events if e["counted"]]
        lines.append("- perf stat counted: " + ", ".join(
            f"{e['event']} {e['value']:g}{' ' + e['unit'] if e.get('unit') else ''}" for e in counted))
        missing = [e["event"] for e in events if not e["counted"]]
        if missing:
            lines.append("- Not measurable here: " + ", ".join(missing))
    lines += [f"- Profile: {table['total']:{'.2f' if unit == 's' else '.3g'}} {unit} of {table['event']} over "
              f"{len(table['workers'])} {kind}(s): " + ", ".join(
                  f"{w.split()[1]}: {_num(a, unit)}" for w, a in table["worker_totals"].items()),
              f"- Parallel runtime libraries (waiting / spinning): {table['runtime_share_pct']}% of samples; "
              f"kernel: {table['kernel_share_pct']}%"]
    if len(table["workers"]) > 1:
        lines.append(f"- Time in parallel runtime libraries per {kind}: " + ", ".join(
            f"{w.split()[1]}: {_num(a, unit)}" for w, a in table["runtime_per_worker"].items()))
    lines += ["",
              f"## Functions by self time ({unit})", "",
              f"Imbalance is max / mean of a function's time over the {len(table['workers'])} {kind}s "
              f"(1.0 = even). Per-{kind} columns are in {unit}.", "",
              "| # | Function | Library | Share | Total | Imbalance | " + " | ".join(table["workers"]) + " |",
              "| ---: | --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in table["workers"]) + " |"]
    for i, f in enumerate(table["functions"][:20], 1):
        tag = " (runtime)" if f["runtime"] else (" (kernel)" if f["kernel"] else "")
        lines.append(f"| {i} | `{short(f['symbol'])}`{tag} | {f['dso']} | {f['share_pct']}% | {_num(f['amount'], unit)} | "
                     f"{f['imbalance'] if f['imbalance'] is not None else '-'} | "
                     + " | ".join(_num(f["per_worker"][w], unit) for w in table["workers"]) + " |")
    lines += ["", "Full table: functions.json. Compare with another artifact set: "
              "`python -m sage.analysis compare <this dir> <other dir> [--work <units here> <units there>]`.", ""]
    return "\n".join(lines)


def summarize(run_dir: Path) -> dict:
    """Write functions.json and summary.md for an artifact set; returns the function table."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    table = functions_for(run_dir)
    rss_file = run_dir / ".peak_rss_kb"
    peak = int(rss_file.read_text().strip()) if rss_file.is_file() and rss_file.read_text().strip().isdigit() else None
    table["peak_rss_kb"] = peak
    (run_dir / "functions.json").write_text(json.dumps(table, indent=2) + "\n")
    (run_dir / "summary.md").write_text(render_summary(meta, table, peak))
    return table


def load_set(run_dir: Path) -> tuple[dict, dict]:
    meta = json.loads((run_dir / "metadata.json").read_text())
    path = run_dir / "functions.json"
    table = json.loads(path.read_text()) if path.is_file() else summarize(run_dir)
    return meta, table


def render_compare(a_dir: Path, a: tuple[dict, dict], b_dir: Path, b: tuple[dict, dict],
                   work: tuple[float, float] | None = None) -> str:
    (ma, ta), (mb, tb) = a, b
    unit = ta["unit"]
    ratio = lambda x, y: f"{y / x:.2f}" if x and y is not None else "-"
    lines = [f"# {a_dir.name} → {b_dir.name}", "",
             f"- A: `{ma.get('run_command')}`" + (f" ({ma['label']})" if ma.get("label") else ""),
             f"- B: `{mb.get('run_command')}`" + (f" ({mb['label']})" if mb.get("label") else "")]
    for name, m in (("A", ma), ("B", mb)):
        if m.get("env_overrides"):
            lines.append(f"- {name} environment changes: " + ", ".join(f"{k}={v}" for k, v in m["env_overrides"].items()))
    wa, wb = (ma.get("wall_clock") or {}).get("app_mean_s"), (mb.get("wall_clock") or {}).get("app_mean_s")
    lines.append(f"- Wall-clock: {wa} s → {wb} s (B/A = {ratio(wa, wb)})")
    lines.append(f"- Total profiled {unit}: {_num(ta['total'], unit)} → {_num(tb['total'], unit)} "
                 f"(B/A = {ratio(ta['total'], tb['total'])}); {len(ta['workers'])} → {len(tb['workers'])} "
                 f"{ta['worker_kind']}(s)")
    lines.append(f"- Runtime libraries: {ta['runtime_share_pct']}% → {tb['runtime_share_pct']}% of samples")
    if work:
        lines.append(f"- Work units: A = {work[0]:g}, B = {work[1]:g} (B/A = {work[1] / work[0]:.3g}); "
                     "'per unit B/A' below divides each side by its work")
    lines += ["", "| Function | Share A | Share B | A | B | B/A" + (" | per unit B/A" if work else "")
              + " | Imbalance A | Imbalance B |",
              "| --- | ---: | ---: | ---: | ---: | ---: |" + (" ---: |" if work else "") + " ---: | ---: |"]
    by_key = lambda t: {(f["dso"], f["symbol"]): f for f in t["functions"]}
    fa, fb = by_key(ta), by_key(tb)
    keys = list(dict.fromkeys([k for k in fa][:15] + [k for k in fb][:15]))
    for k in keys:
        x, y = fa.get(k), fb.get(k)
        ax, by = (x or {}).get("amount"), (y or {}).get("amount")
        per_unit = (f"{(by / work[1]) / (ax / work[0]):.2f}" if ax and by and work else "-")
        lines.append(f"| `{short(k[1])}` ({k[0]}) | {(x or {}).get('share_pct', '-')}% | {(y or {}).get('share_pct', '-')}% | "
                     f"{_num(ax, unit)} | {_num(by, unit)} | {ratio(ax, by)}" + (f" | {per_unit}" if work else "")
                     + f" | {(x or {}).get('imbalance') or '-'} | {(y or {}).get('imbalance') or '-'} |")
    lines.append("")
    return "\n".join(lines)


def maxrss(out: Path, cmd: list[str]) -> int:
    """Run cmd; record the peak resident set size (KiB) of the largest process it started."""
    rc = subprocess.run(cmd).returncode
    out.write_text(f"{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n")
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["maxrss"]:  # maxrss OUT -- CMD...: parsed by hand so CMD's options pass through untouched
        if len(argv) < 4 or argv[2] != "--":
            print("usage: python -m sage.analysis maxrss OUT -- CMD...", file=sys.stderr)
            return 2
        return maxrss(Path(argv[1]), argv[3:])
    parser = argparse.ArgumentParser(prog="python -m sage.analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="write functions.json and summary.md for an artifact set")
    s.add_argument("run_dir", type=Path)
    c = sub.add_parser("compare", help="per-function comparison of two artifact sets")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--work", nargs=2, type=float, metavar=("WA", "WB"),
                   help="units of work done by A and B (e.g. elements x iterations), to compare cost per unit")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "summary":
            summarize(args.run_dir)
            print(f"wrote {args.run_dir / 'summary.md'} and functions.json")
        else:
            if args.work and min(args.work) <= 0:
                parser.error("--work values must be positive")
            print(render_compare(args.a, load_set(args.a), args.b, load_set(args.b),
                                 tuple(args.work) if args.work else None))
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
