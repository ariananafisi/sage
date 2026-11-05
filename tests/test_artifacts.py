"""Parsers and metadata.json assembly, tested on saved perf output (no perf needed).

perf-stat-software.txt, perf-report.txt and perf-annotate.txt are real output from a VM without
hardware counters. perf-stat-hardware.txt and perf-stat-single.txt follow the same perf 6.x
format for a machine with counters (multiplexed events, a topdown block) and a single run.
"""

import json
from pathlib import Path

import pytest

from sage.artifacts import build_metadata, main, parse_perf_report, parse_perf_stat, redacted, thread_settings

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text()


def event(stat, name):
    return next(e for e in stat["events"] if e["event"] == name)


def test_perf_stat_software_only():
    stat = parse_perf_stat(fixture("perf-stat-software.txt"))
    assert stat["command"] == "/work/comd/src/bin/CoMD-serial -x 6 -y 6 -z 6 -N 10"
    assert stat["runs"] == 3
    assert (stat["elapsed_s"], stat["elapsed_stddev_s"], stat["elapsed_variance_pct"]) == (0.0931, 0.0319, 34.29)
    assert event(stat, "task-clock") == {
        "event": "task-clock", "value": 33.64, "unit": "msec", "supported": True, "counted": True,
        "variance_pct": 15.4, "multiplexed_pct": None, "comment": "0.361 CPUs utilized"}
    assert event(stat, "page-faults")["value"] == 1101 and event(stat, "cpu-migrations")["variance_pct"] is None
    for name in ("cycles", "instructions", "branches", "LLC-load-misses"):
        assert event(stat, name)["supported"] is False and event(stat, name)["value"] is None
    assert len(stat["events"]) == 12
    assert stat["ipc"] is None


def test_perf_stat_hardware_counters():
    stat = parse_perf_stat(fixture("perf-stat-hardware.txt"))
    assert stat["runs"] == 3 and stat["elapsed_s"] == 6.2512 and stat["elapsed_variance_pct"] == 0.41
    cycles = event(stat, "cycles")
    assert cycles["value"] == 87_113_562_903 and cycles["multiplexed_pct"] == 71.42 and cycles["variance_pct"] == 0.4
    assert event(stat, "branch-misses")["comment"] == "0.50% of all branches"
    assert stat["ipc"] == 1.107
    llc = event(stat, "LLC-load-misses")
    assert llc["supported"] is True and llc["counted"] is False and llc["multiplexed_pct"] == 0.0
    assert not any("tma" in e["event"] or "Topdown" in e["event"] for e in stat["events"])  # metric lines skipped
    assert len(stat["events"]) == 12


def test_perf_stat_single_run():
    stat = parse_perf_stat(fixture("perf-stat-single.txt"))
    assert stat["runs"] == 1
    assert stat["elapsed_s"] == 0.513091734 and stat["elapsed_stddev_s"] is None
    assert (stat["user_s"], stat["sys_s"]) == (0.498212, 0.014003)
    assert [e["event"] for e in stat["events"]] == ["task-clock", "context-switches", "cycles"]


def test_perf_stat_empty_or_garbage():
    stat = parse_perf_stat("garbage\n\n# comment\n")
    assert stat["events"] == [] and stat["elapsed_s"] is None and stat["runs"] == 1


def test_perf_report_self_time_by_dso_and_symbol():
    report = parse_perf_report(fixture("perf-report.txt"), top=5)
    assert report["event"] == "task-clock:ppp"
    assert report["samples"] == "448" and report["event_count"] == 112000000 and report["lost_samples"] == 0
    assert report["rows"] == 35
    assert report["top"][0] == {"self_pct": 85.04, "dso": "CoMD-serial", "kernel": False, "symbol": "ljForce"}
    assert [r["symbol"] for r in report["top"]] == [
        "ljForce", "putAtomInBox", "msort_with_tmp.part.0", "exc_page_fault", "timestep"]
    assert report["top"][3]["kernel"] is True and report["top"][3]["dso"] == "[kernel.kallsyms]"
    pcts = [r["self_pct"] for r in parse_perf_report(fixture("perf-report.txt"), top=100)["top"]]
    assert pcts == sorted(pcts, reverse=True)  # sorted by self time


def test_perf_report_without_samples():
    assert parse_perf_report("") == {"event": None, "samples": None, "event_count": None,
                                     "lost_samples": None, "rows": 0, "top": []}


def test_redaction_and_thread_settings():
    env = {"ANTHROPIC_API_KEY": "sk-1", "GH_TOKEN": "t", "HTTPS_PROXY": "http://u:p@h", "DB_PASSWORD": "x",
           "OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "true", "MKL_NUM_THREADS": "1", "HOME": "/root"}
    clean = redacted(env)
    assert all(clean[k] == "<redacted>" for k in ("ANTHROPIC_API_KEY", "GH_TOKEN", "HTTPS_PROXY", "DB_PASSWORD"))
    assert clean["OMP_NUM_THREADS"] == "4" and clean["HOME"] == "/root"
    threads = thread_settings(env, ranks=4)
    assert threads["variables"] == {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "true"}
    assert threads["mpi_ranks"] == 4 and threads["cpus_available"] >= 1 and threads["cpu_affinity"]


FIELDS = {
    "run_id": "r1", "label": "baseline", "target": "comd", "variant": "serial", "status": "ok",
    "started_at": "2026-11-05T20:00:00Z", "finished_at": "2026-11-05T20:00:09Z",
    "command_line": "scripts/profile_target.sh comd r1 --label baseline", "run_command": "/work/bin/app -N 10",
    "stat_command": "perf stat -r 3 -d -o perf-stat.txt -- /work/bin/app -N 10", "perf_version": "perf version 6.8",
    "stat_ns": "1500000000", "record_ns": "750000000", "total_ns": "2600000000",
}


def test_metadata_serial_run(tmp_path):
    (tmp_path / "perf-stat.txt").write_text(fixture("perf-stat-software.txt"))
    (tmp_path / "perf-report.txt").write_text(fixture("perf-report.txt"))
    (tmp_path / "perf.data").write_bytes(b"x" * 10)
    build = tmp_path / "build-info.json"
    build.write_text(json.dumps({"commit": "abc", "flags": "-O3 -g -fno-omit-frame-pointer"}))
    meta = build_metadata(tmp_path, {**FIELDS, "build_info": str(build)}, {"OMP_NUM_THREADS": "2", "MY_TOKEN": "s"})

    assert meta["status"] == "ok" and meta["label"] == "baseline" and meta["error"] is None
    assert meta["mpi"] is None and meta["perf"]["rank_reports"] is None
    assert meta["wall_clock"] == {"total_s": 2.6, "perf_stat_s": 1.5, "perf_record_s": 0.75, "app_mean_s": 0.0931,
                                  "app_stddev_s": 0.0319, "app_variance_pct": 34.29, "repeats": 3}
    assert meta["perf"]["report"]["top"][0]["symbol"] == "ljForce"
    assert meta["perf"]["commands"]["stat"].startswith("perf stat -r 3")
    assert meta["build"]["commit"] == "abc"
    assert meta["threads"]["variables"] == {"OMP_NUM_THREADS": "2"}
    assert meta["environment"]["MY_TOKEN"] == "<redacted>"
    assert meta["host"]["arch"] and meta["host"]["logical_cpus"] >= 1
    assert {a["path"] for a in meta["artifacts"]} == {"perf-stat.txt", "perf-report.txt", "perf.data", "build-info.json"}


def test_metadata_mpi_run_has_per_rank_reports(tmp_path):
    for r in (0, 1):
        (tmp_path / f"rank-{r}").mkdir()
        (tmp_path / f"rank-{r}" / "perf-report.txt").write_text(fixture("perf-report.txt"))
    (tmp_path / "perf-rank-wrapper.sh").write_text("#!/bin/sh\n")
    meta = build_metadata(tmp_path, {**FIELDS, "variant": "mpi", "ranks": "2", "launcher": "mpirun -np 2"}, {})
    assert meta["mpi"] == {"ranks": 2, "launcher": "mpirun -np 2"}
    assert meta["perf"]["report"] is None
    assert set(meta["perf"]["rank_reports"]) == {"0", "1"}
    assert meta["perf"]["rank_reports"]["1"]["top"][0]["symbol"] == "ljForce"
    assert meta["perf"]["stat"] is None  # no perf-stat.txt in this directory


def test_metadata_failed_run_with_partial_artifacts(tmp_path):
    (tmp_path / "stderr.log").write_text("perf: boom\n")
    meta = build_metadata(tmp_path, {**FIELDS, "status": "failed", "error": "perf record failed (exit 3)",
                                     "record_ns": ""}, {})
    assert meta["status"] == "failed" and meta["error"] == "perf record failed (exit 3)"
    assert meta["wall_clock"]["perf_record_s"] is None and meta["wall_clock"]["app_mean_s"] is None
    assert meta["perf"]["stat"] is None and meta["perf"]["report"] is None and meta["build"] is None


def test_cli_writes_metadata_and_parses_files(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SECRET_THING", "hidden")
    assert main(["metadata", str(tmp_path), "--set", "run_id=x", "--set", "label=a=b", "--set", "status=ok"]) == 0
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["run_id"] == "x" and meta["label"] == "a=b" and meta["environment"]["SECRET_THING"] == "<redacted>"
    assert main(["parse-report", str(FIXTURES / "perf-report.txt")]) == 0
    assert json.loads(capsys.readouterr().out)["top"][0]["symbol"] == "ljForce"
    with pytest.raises(SystemExit):
        main(["parse-stat"])
