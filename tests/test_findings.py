"""findings.json schema, validator and report.md generation, tested on fixtures in tests/fixtures/findings/."""

import copy
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from sage.findings import SCHEMA, FindingsError, load, main, render, validate, write_report

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FINDINGS = FIXTURES / "findings"
ARTIFACTS = ["perf-report.txt", "annotate/ljForce.txt", "experiments/size-40/perf-stat.txt"]
REQUIRED = ["rank", "function", "source_location", "sample_percent", "bottleneck_type", "supporting_evidence",
            "likely_cause", "experiment", "experiment_result", "confidence", "artifact_paths"]


@pytest.fixture
def run_dir(tmp_path):
    """A run directory named like the fixtures' run_id, holding the artifacts they cite."""
    d = tmp_path / "fixture-run"
    for rel in ARTIFACTS:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("perf output\n")
    shutil.copy(FIXTURES / "perf-report.txt", d / "perf-report.txt")
    return d


def fixture(name):
    return json.loads((FINDINGS / name).read_text())


def use(run_dir, data):
    (run_dir / "findings.json").write_text(json.dumps(data) if not isinstance(data, str) else data)
    return run_dir


def errors(run_dir, data):
    return validate(data, run_dir)


def test_schema_is_valid_draft7():
    Draft7Validator.check_schema(SCHEMA)
    assert SCHEMA["definitions"]["finding"]["required"] == REQUIRED
    assert SCHEMA["properties"]["status"]["enum"] == ["bottlenecks_found", "inconclusive"]


# --- valid fixtures ---------------------------------------------------------------

def test_valid_bottlenecks_fixture(run_dir):
    data = load(use(run_dir, fixture("valid-bottlenecks.json")))
    assert data["status"] == "bottlenecks_found" and [f["rank"] for f in data["findings"]] == [1, 2]


def test_inconclusive_report_without_findings_passes(run_dir):
    data = fixture("valid-inconclusive-empty.json")
    assert data["findings"] == []
    assert errors(run_dir, data) == []
    assert load(use(run_dir, data))["status"] == "inconclusive"


def test_inconclusive_may_carry_tentative_findings(run_dir):
    data = {**fixture("valid-bottlenecks.json"), "status": "inconclusive"}
    assert errors(run_dir, data) == []


def test_limitations_are_optional(run_dir):
    data = fixture("valid-inconclusive-empty.json")
    del data["limitations"]
    assert errors(run_dir, data) == []


def test_artifact_paths_may_be_absolute_inside_run_dir_and_directories(run_dir):
    data = fixture("valid-bottlenecks.json")
    data["findings"][0]["artifact_paths"] = [str(run_dir / "perf-report.txt"), "experiments/size-40", "./annotate/ljForce.txt"]
    assert errors(run_dir, data) == []


# --- invalid fixtures -------------------------------------------------------------

def test_finding_without_artifact_evidence_fails(run_dir):
    errs = errors(run_dir, fixture("invalid-no-artifact-evidence.json"))
    assert errs == ["findings[1].artifact_paths: finding has no artifact evidence; "
                    "list at least one file in the run directory"]


def test_finding_missing_artifact_paths_key_fails(run_dir):
    data = fixture("valid-bottlenecks.json")
    del data["findings"][0]["artifact_paths"]
    assert errors(run_dir, data) == ["findings[0]: finding has no artifact evidence; 'artifact_paths' is required"]


def test_artifact_that_does_not_exist_fails(run_dir):
    errs = errors(run_dir, fixture("invalid-missing-artifact.json"))
    assert len(errs) == 1
    assert errs[0].startswith("findings[0].artifact_paths[1]: 'annotate/doesNotExist.txt' does not exist in the run directory")


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "annotate/../../outside.txt"])
def test_artifact_outside_run_dir_fails(run_dir, path):
    (run_dir.parent / "outside.txt").write_text("not evidence")
    data = fixture("invalid-artifact-outside-run.json")
    data["findings"][0]["artifact_paths"] = [path]
    [err] = errors(run_dir, data)
    assert "is outside the run directory" in err


def test_artifact_symlink_escaping_run_dir_fails(run_dir):
    (run_dir.parent / "secret.txt").write_text("x")
    (run_dir / "link.txt").symlink_to(run_dir.parent / "secret.txt")
    data = fixture("valid-bottlenecks.json")
    data["findings"][0]["artifact_paths"] = ["link.txt"]
    assert "is outside the run directory" in errors(run_dir, data)[0]


@pytest.mark.parametrize("path", ["findings.json", "report.md"])
def test_generated_outputs_are_not_evidence(run_dir, path):
    (run_dir / path).write_text("{}")
    data = fixture("valid-bottlenecks.json")
    data["findings"][1]["artifact_paths"] = [path]
    assert "is a generated output, not evidence" in errors(run_dir, data)[0]


def test_bottlenecks_found_needs_findings(run_dir):
    assert errors(run_dir, fixture("invalid-bottlenecks-without-findings.json")) == [
        "findings: status is bottlenecks_found but there are no findings (use inconclusive if nothing was established)"]


def test_invalid_fields_fixture_reports_every_problem(run_dir):
    errs = errors(run_dir, fixture("invalid-fields.json"))
    assert len(errs) == 4
    assert any(e.startswith("status: 'done' is not one of") for e in errs)
    assert any(e.startswith("findings[0].sample_percent: 140 is greater than the maximum of 100") for e in errs)
    assert any(e.startswith("findings[0].confidence: 'certain' is not one of") for e in errs)
    assert "findings[1]: 'likely_cause' is a required property" in errs


@pytest.mark.parametrize("field", [f for f in REQUIRED if f != "artifact_paths"])
def test_every_finding_field_is_required(run_dir, field):
    data = fixture("valid-bottlenecks.json")
    del data["findings"][0][field]
    assert errors(run_dir, data) == [f"findings[0]: '{field}' is a required property"]


@pytest.mark.parametrize("field,value,message", [
    ("rank", 0, "0 is less than the minimum of 1"),
    ("rank", 1.5, "1.5 is not of type 'integer'"),
    ("function", "", "should be non-empty"),
    ("source_location", "ljForce.c", "does not match"),
    ("source_location", "ljForce.c:0", "does not match"),
    ("source_location", "ljForce.c:12-", "does not match"),
    ("sample_percent", -1, "less than the minimum of 0"),
    ("sample_percent", "91%", "is not of type 'number'"),
    ("bottleneck_type", "slow", "is not one of"),
    ("supporting_evidence", [], "should be non-empty"),
    ("supporting_evidence", "it is hot", "is not of type 'array'"),
    ("supporting_evidence", [""], "should be non-empty"),
    ("likely_cause", "", "should be non-empty"),
    ("experiment", None, "is not of type 'string'"),
    ("experiment_result", "", "should be non-empty"),
    ("confidence", "High", "is not one of"),
    ("artifact_paths", "perf-report.txt", "is not of type 'array'"),
    ("artifact_paths", ["perf-report.txt", "perf-report.txt"], "has non-unique elements"),
    ("extra_field", 1, "Additional properties are not allowed"),
])
def test_invalid_finding_values(run_dir, field, value, message):
    data = fixture("valid-bottlenecks.json")
    data["findings"][0][field] = value
    errs = errors(run_dir, data)
    assert len(errs) == 1 and message in errs[0], errs


@pytest.mark.parametrize("change,message", [
    ({"schema_version": 2}, "schema_version: 1 was expected"),
    ({"target": "hpcg"}, "target: 'hpcg' is not one of"),
    ({"summary": ""}, "summary: '' should be non-empty"),
    ({"findings": {}}, "findings: {} is not of type 'array'"),
    ({"limitations": "none"}, "limitations: 'none' is not of type 'array'"),
    ({"notes": "x"}, "Additional properties are not allowed ('notes' was unexpected)"),
])
def test_invalid_top_level(run_dir, change, message):
    errs = errors(run_dir, {**fixture("valid-inconclusive-empty.json"), **change})
    assert len(errs) == 1 and message in errs[0], errs


def test_missing_top_level_keys_and_non_object(run_dir):
    errs = errors(run_dir, {"status": "inconclusive"})
    for key in ("schema_version", "run_id", "target", "summary", "findings"):
        assert f"findings.json: '{key}' is a required property" in errs
    assert errors(run_dir, []) == ["findings.json: [] is not of type 'object'"]


def test_run_id_must_match_directory(run_dir):
    data = {**fixture("valid-inconclusive-empty.json"), "run_id": "other-run"}
    assert errors(run_dir, data) == ["run_id: 'other-run' does not match the run directory 'fixture-run'"]


@pytest.mark.parametrize("ranks", [[1, 1], [1, 3], [2, 3]])
def test_ranks_must_be_one_to_n(run_dir, ranks):
    data = fixture("valid-bottlenecks.json")
    for f, r in zip(data["findings"], ranks):
        f["rank"] = r
    assert errors(run_dir, data) == [f"findings: ranks must be 1..2 with no gaps or repeats, got {sorted(ranks)}"]


def test_load_file_errors(run_dir):
    with pytest.raises(FindingsError, match="findings.json not found"):
        load(run_dir)
    with pytest.raises(FindingsError, match="is not valid JSON"):
        load(use(run_dir, "{not json"))
    with pytest.raises(FindingsError) as exc:
        load(use(run_dir, fixture("invalid-fields.json")))
    assert len(exc.value.errors) == 4


# --- report.md ----------------------------------------------------------------------

METADATA = {
    "run_id": "fixture-run", "target": "comd", "variant": "serial", "label": "baseline", "mpi": None,
    "run_command": "/w/CoMD-serial -x 30 -y 30 -z 30 -N 100",
    "build": {"repo_url": "https://github.com/ECP-copa/CoMD.git", "commit": "3d48396b77ca8caa3124bc2391f9139c3ffb556c",
              "compiler_version": "gcc 13.3.0", "flags": "-O3 -g -fno-omit-frame-pointer"},
    "wall_clock": {"app_mean_s": 18.707, "app_variance_pct": 1.09, "repeats": 3},
    "host": {"model": "Intel Xeon", "logical_cpus": 4, "kernel": "6.18"},
    "perf": {"version": "perf version 6.8.12", "stat": {"events": [
        {"event": "task-clock", "counted": True}, {"event": "cycles", "counted": False}]}},
}


def test_render_bottlenecks_report():
    report = render(fixture("valid-bottlenecks.json"), METADATA)
    assert report.startswith("# COMD performance bottlenecks: run fixture-run\n\n**Status:** Bottlenecks found\n")
    assert "| 1 | `ljForce` | `src-mpi/ljForce.c:139-188` | 91.75% | poor vectorization | high |" in report
    assert "| 2 | `sortAtomsInCell` |" in report
    assert "### 1. `ljForce`: poor vectorization (high confidence)" in report
    assert "- 91.75% of 83K task-clock samples are in ljForce (perf-report.txt)." in report
    assert "**Likely cause:** Data-dependent continue statements" in report
    assert "**Experiment:** Profiled lattice sizes" in report and "**Result:** Per-atom cost" in report
    assert "**Artifacts:** [perf-report.txt](perf-report.txt), [annotate/ljForce.txt](annotate/ljForce.txt)" in report
    assert "## Limitations\n\n- No hardware PMU events" in report
    assert "- Wall-clock: 18.707 s ± 1.09% (mean of 3 runs under perf stat)" in report
    assert "- Build: https://github.com/ECP-copa/CoMD.git @ `3d48396b77ca`, gcc 13.3.0, `-O3 -g -fno-omit-frame-pointer`" in report
    assert "not available: cycles" in report
    assert report.index("### 1.") < report.index("### 2.")


def test_render_orders_by_rank_and_escapes_table_cells():
    data = fixture("valid-bottlenecks.json")
    data["findings"].reverse()
    data["findings"][0]["function"] = "operator|"
    report = render(data)
    assert report.index("### 1.") < report.index("### 2.")
    assert "`operator\\|`" in report
    assert "## Run" not in report  # no metadata given


def test_render_inconclusive_without_findings():
    report = render(fixture("valid-inconclusive-empty.json"))
    assert "**Status:** Inconclusive" in report
    assert "No bottleneck was established with enough evidence." in report
    assert "| Rank |" not in report and "###" not in report


def test_write_report_validates_first(run_dir):
    use(run_dir, fixture("invalid-no-artifact-evidence.json"))
    with pytest.raises(FindingsError):
        write_report(run_dir)
    assert not (run_dir / "report.md").exists()
    use(run_dir, fixture("valid-bottlenecks.json"))
    (run_dir / "metadata.json").write_text(json.dumps(METADATA))
    out = write_report(run_dir)
    assert out == run_dir / "report.md" and "## Run" in out.read_text()


def test_cli(run_dir, capsys):
    use(run_dir, fixture("valid-inconclusive-empty.json"))
    assert main(["validate", str(run_dir)]) == 0
    assert "valid: inconclusive, 0 finding(s)" in capsys.readouterr().out
    assert main(["render", str(run_dir)]) == 0 and (run_dir / "report.md").is_file()
    use(run_dir, fixture("invalid-missing-artifact.json"))
    assert main(["validate", str(run_dir)]) == 1
    err = capsys.readouterr().err
    assert "findings.json is invalid (1 problem(s)):" in err and "doesNotExist.txt" in err
    assert main(["bogus"]) == 2
