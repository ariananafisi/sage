"""sage.analysis: per-function, per-worker tables, summaries and comparisons, on saved perf output.

perf-report-threads-lulesh*.txt are real `perf report --sort pid,dso,sym` output (top rows) from
unmodified LULESH and from its injected-imbalance build, 4 OpenMP threads, on the development VM.
"""

import json
import sys
from pathlib import Path

import pytest

from sage import analysis
from sage.analysis import function_table, parse_worker_report, render_compare, render_summary, short

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FB = "CalcFBHourglassForceForElems(Domain&, double*, double*, double*, double*, double*, double*, double*, double, int, int) [clone ._omp_fn.0]"


def table(name):
    return function_table({"": parse_worker_report((FIXTURES / f"perf-report-threads-{name}.txt").read_text())}, "thread")


def fn(t, symbol):
    return next(f for f in t["functions"] if f["symbol"] == symbol)


def test_parse_worker_report():
    report = parse_worker_report((FIXTURES / "perf-report-threads-lulesh.txt").read_text())
    assert report["event"] == "task-clock:ppp" and report["event_count"] == 99433250000
    first = report["rows"][0]
    assert first == {"pct": 2.34, "tid": "1375", "comm": "lulesh2.0", "dso": "[kernel.kallsyms]", "kernel": True,
                     "symbol": "_raw_spin_unlock_irqrestore"}
    assert any(r["symbol"] == FB and r["dso"] == "lulesh2.0" for r in report["rows"])


def test_even_work_has_imbalance_near_one():
    t = table("lulesh")
    assert t["unit"] == "s" and t["worker_kind"] == "thread" and len(t["workers"]) == 4
    f = fn(t, FB)
    assert f["imbalance"] == pytest.approx(1.02, abs=0.03)
    assert set(f["per_worker"]) == set(t["workers"])
    assert f["amount"] == pytest.approx(sum(f["per_worker"].values()), rel=1e-3)
    assert t["runtime_share_pct"] > 10  # libgomp spin-waiting
    gomp = next(f for f in t["functions"] if "libgomp" in f["dso"])
    assert gomp["runtime"] and not gomp["kernel"]
    assert next(f for f in t["functions"] if f["kernel"])["dso"] == "[kernel.kallsyms]"


def test_injected_imbalance_is_visible():
    t = table("lulesh-imbalance")
    f = fn(t, FB)
    assert f["imbalance"] > 2.5
    heavy = max(f["per_worker"], key=f["per_worker"].get)
    others = [v for w, v in f["per_worker"].items() if w != heavy]
    assert f["per_worker"][heavy] > 5 * max(others)
    assert min(t["runtime_per_worker"], key=t["runtime_per_worker"].get) == heavy  # the others wait for it
    assert t["functions"][0]["symbol"] == FB  # the uneven function is now the top one


def test_ranks_are_workers_and_other_events_keep_their_units():
    def report(pct_lj, pct_wait, count):
        return parse_worker_report(f"# Samples: 1K of event 'cycles'\n# Event count (approx.): {count}\n"
                                   f"    {pct_lj}%  101:CoMD-mpi  CoMD-mpi  [.] ljForce\n"
                                   f"    {pct_wait}%  101:CoMD-mpi  libopen-pal.so.40  [.] opal_progress\n"
                                   f"     3.00%  102:CoMD-mpi  CoMD-mpi  [.] ljForce\n")
    t = function_table({"0": report(90, 7, 1000), "1": report(60, 37, 1000), "2": report(60, 37, 1000)}, "rank")
    assert t["unit"] == "cycles counts" and t["workers"] == ["rank 0", "rank 1", "rank 2"]
    lj = fn(t, "ljForce")
    assert lj["per_worker"] == {"rank 0": 930.0, "rank 1": 630.0, "rank 2": 630.0}  # both threads of a rank count
    assert lj["imbalance"] == pytest.approx(930 / 730, abs=0.01)
    assert fn(t, "opal_progress")["runtime"]
    assert t["runtime_share_pct"] == pytest.approx(81 / 300 * 100, abs=0.01)  # opal_progress / all samples
    assert t["runtime_per_worker"] == {"rank 0": 70.0, "rank 1": 370.0, "rank 2": 370.0}  # the busy rank waits least


def test_small_threads_are_not_workers():
    text = ("# Samples: 1K of event 'task-clock'\n# Event count (approx.): 1000000000\n"
            "    60.00%  1:app  app  [.] work\n    39.50%  2:app  app  [.] work\n     0.50%  3:app  app  [.] helper\n")
    t = function_table({"": parse_worker_report(text)}, "thread")
    assert t["workers"] == ["thread 1", "thread 2"] and fn(t, "work")["imbalance"] == pytest.approx(0.6 / 0.4975, abs=0.01)
    assert fn(t, "work")["amount"] == pytest.approx(0.995)


@pytest.mark.parametrize("symbol,expected", [
    (FB, "CalcFBHourglassForceForElems(…) [clone ._omp_fn.0]"),
    ("ljForce", "ljForce"),
    ("f(std::vector<int, std::allocator<int> > const&)", "f(…)"),
    ("g(int (*)(int))", "g(…)"),
])
def test_short(symbol, expected):
    assert short(symbol) == expected


META = {"run_id": "baseline", "label": "baseline", "run_command": "/w/app -s 90", "mpi": None, "env_overrides": {},
        "wall_clock": {"app_mean_s": 26.6, "app_variance_pct": 1.2, "repeats": 3}, "host": {"l3_cache": "33 MiB"},
        "perf": {"stat": {"events": [{"event": "task-clock", "value": 99000.0, "unit": "msec", "counted": True},
                                     {"event": "cycles", "value": None, "unit": None, "counted": False}]}}}


def test_render_summary():
    md = render_summary(META, table("lulesh-imbalance"), peak_rss_kb=665600)
    assert md.startswith("# Artifact set baseline: baseline")
    assert "- Wall-clock: 26.6 s ± 1.2%, mean of 3 runs under perf stat" in md
    assert "- Peak memory of the largest process: 650 MiB (last-level cache: 33 MiB)" in md
    assert "- Not measurable here: cycles" in md and "task-clock 99000 msec" in md
    assert "| 1 | `CalcFBHourglassForceForElems(…) [clone ._omp_fn.0]` | lulesh2.0 | " in md
    assert "(runtime) | libgomp.so.1.0.0 |" in md and "(kernel) | [kernel.kallsyms] |" in md
    assert "| thread 1426 | thread 1428 | thread 1429 | thread 1430 |" in md
    assert "- Time in parallel runtime libraries per thread: 1426: " in md


def test_render_compare_per_unit():
    a = (META, table("lulesh"))
    b = ({**META, "run_command": "/w/app -s 30", "label": "small", "wall_clock": {"app_mean_s": 1.0}}, table("lulesh-imbalance"))
    md = render_compare(Path("baseline"), a, Path("experiments/small"), b, work=("100", "50"))
    assert md.startswith("# baseline → small")
    assert "- Wall-clock: 26.6 s → 1.0 s (B/A = 0.04)" in md
    assert "- Work units: A = 100, B = 50 (B/A = 0.5)" in md
    row = next(l for l in md.splitlines() if l.startswith("| `CalcFBHourglassForceForElems"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    ax, by = fn(a[1], FB)["amount"], fn(b[1], FB)["amount"]
    assert cells[5] == f"{by / ax:.2f}" and cells[6] == f"{(by / 50) / (ax / 100):.2f}"
    assert cells[8] == str(fn(b[1], FB)["imbalance"])
    assert "per unit B/A" not in render_compare(Path("a"), a, Path("b"), b)


@pytest.mark.parametrize("expr,value", [("2.56e7", 2.56e7), ("4000*2000", 8e6), (" 4 * 40^3 * 100 ", 2.56e7),
                                        ("1.5*2", 3.0)])
def test_work_units(expr, value):
    assert analysis.work_units(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["", "4000x2000", "2+3", "__import__('os')", "4**3", "-5"])
def test_work_units_rejects_other_expressions(expr):
    with pytest.raises(ValueError):
        analysis.work_units(expr)


def test_compare_shows_the_work_products():
    a = (META, table("lulesh"))
    md = render_compare(Path("baseline"), a, Path("small"), a, work=("256000*100", "4000*2000"))
    assert "- Work units: A = 256000*100 = 2.56e+07, B = 4000*2000 = 8e+06 (B/A = 0.312)" in md


def test_maxrss_records_largest_child_and_passes_exit_code(tmp_path):
    out = tmp_path / "rss"
    code = "import sys; b = bytearray(80 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); sys.exit(3)"
    assert analysis.main(["maxrss", str(out), "--", sys.executable, "-c", code]) == 3
    assert int(out.read_text()) >= 80 * 1024
    assert analysis.main(["maxrss", str(out)]) == 2


def test_summary_and_compare_cli(tmp_path, monkeypatch, capsys):
    reports = {"a": (FIXTURES / "perf-report-threads-lulesh.txt").read_text(),
               "b": (FIXTURES / "perf-report-threads-lulesh-imbalance.txt").read_text()}
    monkeypatch.setattr(analysis, "worker_report", lambda data: reports[data.parent.name])
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "perf.data").write_bytes(b"x")
        (d / "metadata.json").write_text(json.dumps({**META, "run_id": name}))
        (d / ".peak_rss_kb").write_text("2048\n")
    assert analysis.main(["summary", str(tmp_path / "a")]) == 0
    saved = json.loads((tmp_path / "a" / "functions.json").read_text())
    assert saved["peak_rss_kb"] == 2048 and saved["functions"][0]["symbol"] == FB
    assert "Peak memory of the largest process: 2 MiB" in (tmp_path / "a" / "summary.md").read_text()
    capsys.readouterr()
    # b has no functions.json yet: compare creates it
    assert analysis.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--work", "2", "1"]) == 0
    assert "# a → b" in capsys.readouterr().out and (tmp_path / "b" / "functions.json").is_file()
    assert analysis.main(["compare", str(tmp_path / "a"), str(tmp_path / "missing")]) == 1
    for bad in (["2", "0"], ["2", "2+1"]):
        with pytest.raises(SystemExit):
            analysis.main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--work", *bad])


def test_mpi_run_dir_uses_rank_reports(tmp_path, monkeypatch):
    for r in (0, 1):
        (tmp_path / f"rank-{r}").mkdir()
        (tmp_path / f"rank-{r}" / "perf.data").write_bytes(b"x")
    text = lambda pct: (f"# Samples: 1K of event 'task-clock'\n# Event count (approx.): 2000000000\n"
                        f"    {pct}%  7:CoMD-mpi  CoMD-mpi  [.] ljForce\n")
    monkeypatch.setattr(analysis, "worker_report", lambda data: text(90 if data.parent.name == "rank-0" else 45))
    t = analysis.functions_for(tmp_path)
    assert t["worker_kind"] == "rank" and t["workers"] == ["rank 0", "rank 1"]
    assert fn(t, "ljForce")["per_worker"] == {"rank 0": 1.8, "rank 1": 0.9}
