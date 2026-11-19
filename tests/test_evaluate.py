"""bottleneck-agent eval: suite loading, scoring and the harness, on saved fixtures.

No API and no perf: findings come from tests/fixtures/eval/*.json, the agent run is replaced by a
function that writes them, and a local git repository stands in for the upstream source.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from sage import agent, evaluate
from sage.evaluate import (Case, Expect, SuiteError, cache_mib, function_matches, load_reference, load_suite,
                           reference_conditions, score, score_reference, summarize)

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval"
LULESH_FN = ("CalcFBHourglassForceForElems(Domain&, double*, double*, double*, double*, double*, double*, "
             "double*, double, int, int) [clone ._omp_fn.0]")


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


MEMORY = ("memory_latency_bound", "memory_bandwidth_bound")
FB_IMBALANCE = Expect("CalcFBHourglassForceForElems", ("load_imbalance",))
KIN_MEMORY = Expect("CalcKinematicsForElems", MEMORY)
IMBALANCE = Case(id="load-imbalance", description="", patch=None, mpi=False, expect=FB_IMBALANCE,
                 false_positives=(KIN_MEMORY,))
CONTROL = Case(id="negative-control", description="", patch=None, mpi=False, expect=None,
               false_positives=(FB_IMBALANCE, KIN_MEMORY))


# --- the shipped suites ---------------------------------------------------------------------

@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_shipped_suites_are_valid(target):
    suite = load_suite(REPO / "suites" / f"{target}.yaml")
    assert suite.target == target and suite.config == (REPO / "configs" / f"{target}.yaml").resolve()
    kinds = {c.id: c for c in suite.cases}
    assert set(kinds) == {"negative-control", "load-imbalance", "memory-bound"}
    assert kinds["negative-control"].patch is None and kinds["negative-control"].expect is None
    assert kinds["load-imbalance"].expect.bottleneck_types == ("load_imbalance",)
    assert set(kinds["memory-bound"].expect.bottleneck_types) == set(MEMORY)
    # each case's false positives are the bottlenecks the other cases inject
    imbalance, memory = kinds["load-imbalance"].expect, kinds["memory-bound"].expect
    assert kinds["negative-control"].false_positives == (imbalance, memory)
    assert kinds["load-imbalance"].false_positives == (memory,)
    assert kinds["memory-bound"].false_positives == (imbalance,)
    assert kinds["load-imbalance"].mpi == (target == "comd") == kinds["negative-control"].mpi
    for case in (kinds["load-imbalance"], kinds["memory-bound"]):
        assert case.patch.is_file() and case.patch.suffix == ".patch"
        assert case.patch.parent == REPO / "suites" / "patches"


# --- suite validation -----------------------------------------------------------------------

def test_fixture_suite_loads():
    suite = load_suite(FIXTURES / "suite.yaml")
    assert suite.name == "fixture" and [c.id for c in suite.cases] == ["negative-control", "load-imbalance", "broken"]
    assert suite.cases[1].false_positives == (Expect("main", ("poor_vectorization",)),)
    assert suite.cases[0].false_positives == (FB_IMBALANCE, Expect("main", ("poor_vectorization",)))


def test_own_expected_types_are_not_false_positives():
    mine = Case(id="a", description="", patch=None, mpi=False, expect=Expect("f", ("load_imbalance", "compute_bound")))
    other = Case(id="b", description="", patch=None, mpi=False, expect=Expect("f", ("compute_bound", "other")))
    same = Case(id="c", description="", patch=None, mpi=False, expect=Expect("f", ("load_imbalance",)))
    assert evaluate.injected_elsewhere(mine, [mine, other, same]) == (Expect("f", ("other",)),)


def write_suite(tmp_path, data):
    (tmp_path / "p.patch").write_text("x")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


BASE = {"name": "s", "target": "lulesh", "config": str(REPO / "configs" / "lulesh.yaml"),
        "cases": [{"id": "control", "expect": "none"}]}


@pytest.mark.parametrize("change,message", [
    ({"name": "Bad Name"}, "name must be lowercase"),
    ({"target": "hpcg"}, "target must be one of comd, lulesh"),
    ({"config": "missing.yaml"}, "config file not found"),
    ({"cases": []}, "cases must be a non-empty list"),
    ({"extra": 1}, "unknown key 'extra'"),
    ({"false_positive_types": ["load_imbalance"]}, "unknown key 'false_positive_types'"),
    ({"cases": [{"id": "a", "expect": "none"}, {"id": "a", "expect": "none"}]}, "cases[1].id 'a' is used twice"),
    ({"cases": [{"id": "A b", "expect": "none"}]}, "cases[0].id must be lowercase"),
    ({"cases": [{"id": "a"}]}, "cases[0].expect is required"),
    ({"cases": [{"id": "a", "patch": "nope.patch", "expect": "none"}]}, "cases[0].patch: file not found"),
    ({"cases": [{"id": "a", "expect": {"function": "f"}}]}, "must have exactly 'function' and 'bottleneck_types'"),
    ({"cases": [{"id": "a", "expect": {"function": "f(int)", "bottleneck_types": ["load_imbalance"]}}]},
     "must be a plain function name"),
    ({"cases": [{"id": "a", "expect": {"function": "f", "bottleneck_types": []}}]}, "must be a non-empty list"),
    ({"cases": [{"id": "a", "expect": {"function": "f", "bottleneck_types": ["slow"]}}]}, "unknown bottleneck type"),
    ({"cases": [{"id": "a", "mpi": "yes", "expect": "none"}]}, "cases[0].mpi must be true or false"),
    ({"cases": [{"id": "a", "expect": "none", "when": 1}]}, "cases[0]: unknown key 'when'"),
])
def test_invalid_suites(tmp_path, change, message):
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(tmp_path, {**BASE, **change}))
    assert message in str(exc.value)


def test_mpi_case_needs_mpi_config(tmp_path):
    config = yaml.safe_load((REPO / "configs" / "lulesh.yaml").read_text())
    del config["mpi"]
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    data = {**BASE, "config": str(tmp_path / "c.yaml"), "cases": [{"id": "a", "mpi": True, "expect": "none"}]}
    with pytest.raises(SuiteError, match="mpi is true but the target config has no mpi section"):
        load_suite(write_suite(tmp_path, data))


def test_suite_file_errors(tmp_path):
    with pytest.raises(SuiteError, match="suite file not found"):
        load_suite(tmp_path / "missing.yaml")
    (tmp_path / "s.yaml").write_text("- a list\n")
    with pytest.raises(SuiteError, match="top level must be a mapping"):
        load_suite(tmp_path / "s.yaml")


# --- scoring ---------------------------------------------------------------------------------

@pytest.mark.parametrize("reported,expected,match", [
    (LULESH_FN, "CalcFBHourglassForceForElems", True),
    ("CalcFBHourglassForceForElems", "CalcFBHourglassForceForElems", True),
    ("ljForce", "ljForce", True),
    ("ljForce (inner pair loop)", "ljForce", True),
    ("ljForceHelper", "ljForce", False),
    ("my_ljForce", "ljForce", False),
    ("CalcFBHourglassForce", "CalcFBHourglassForceForElems", False),
    ("ns::ljForce(SimFlat*)", "ljForce", True),
])
def test_function_matches(reported, expected, match):
    assert function_matches(reported, expected) is match


def run_dir_with(tmp_path, experiments=()):
    d = tmp_path / "run"
    for e in experiments:
        (d / "experiments" / e).mkdir(parents=True)
    d.mkdir(exist_ok=True)
    return d


def test_score_correct_detection(tmp_path):
    result = score(IMBALANCE, fixture("found-imbalance.json"), agent.OK, run_dir_with(tmp_path, ["threads-1"]), 312.46)
    assert result["function_found"] and result["type_found"] and result["matched_rank"] == 2
    assert result["false_positives"] == [] and result["correct"]
    assert result["experiments"] == 1 and result["experiment_names"] == ["threads-1"]
    assert result["duration_s"] == 312.5 and result["status"] == "ok" and result["findings_valid"]
    assert [f["rank"] for f in result["findings"]] == [1, 2]


def test_score_right_function_wrong_type(tmp_path):
    result = score(IMBALANCE, fixture("wrong-type.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert result["function_found"] and not result["type_found"] and result["matched_rank"] is None
    assert not result["correct"] and result["experiments"] == 0


def test_score_type_on_wrong_function_does_not_count(tmp_path):
    findings = fixture("found-imbalance.json")
    findings["findings"][1]["function"] = "CalcKinematicsForElems(Domain&, double, int) [clone ._omp_fn.0]"
    result = score(IMBALANCE, findings, agent.OK, run_dir_with(tmp_path), 10)
    assert not result["function_found"] and not result["type_found"] and not result["correct"]


def test_score_negative_control(tmp_path):
    clean = score(CONTROL, fixture("control-clean.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert clean["correct"] and clean["false_positives"] == [] and clean["function_found"] is None
    bad = score(CONTROL, fixture("control-false-positive.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert not bad["correct"]
    # rank 2 claims the imbalance the load-imbalance case injects: counted; rank 1 (memory-bound
    # elsewhere) and rank 3 (imbalance in libgomp) may be real properties of the code: not counted
    assert bad["false_positives"] == [{"rank": 2, "function": LULESH_FN, "bottleneck_type": "load_imbalance"}]
    assert score(CONTROL, fixture("inconclusive.json"), agent.OK, run_dir_with(tmp_path), 10)["correct"]


def test_score_false_positive_in_injected_case(tmp_path):
    findings = fixture("found-imbalance.json")
    findings["findings"][0]["bottleneck_type"] = "memory_latency_bound"  # on libgomp: not injected there
    assert score(IMBALANCE, findings, agent.OK, run_dir_with(tmp_path), 10)["correct"]
    findings["findings"][0]["function"] = "CalcKinematicsForElems"  # the memory-bound case injects it here
    result = score(IMBALANCE, findings, agent.OK, run_dir_with(tmp_path), 10)
    assert result["type_found"] and len(result["false_positives"]) == 1 and not result["correct"]


@pytest.mark.parametrize("findings,status,label", [(None, agent.INVALID_FINDINGS, "invalid findings"),
                                                   (None, agent.TIMED_OUT, "timed out"),
                                                   ("found-imbalance.json", agent.SOURCE_MODIFIED, "source modified")])
def test_score_failed_runs_are_not_correct(tmp_path, findings, status, label):
    result = score(IMBALANCE, fixture(findings) if findings else None, status, run_dir_with(tmp_path), 10)
    assert not result["correct"] and result["status"] == label
    assert result["findings_valid"] is (findings is not None)


def test_inconclusive_injected_case_is_a_miss(tmp_path):
    result = score(IMBALANCE, fixture("inconclusive.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert result["function_found"] is False and not result["correct"] and result["findings_status"] == "inconclusive"


def test_summarize_and_markdown(tmp_path):
    cases = [
        {"id": "negative-control", "run_id": "case-1", "patch": None, "expected": None, "run_dir": "r1",
         **score(CONTROL, fixture("control-false-positive.json"), agent.OK, run_dir_with(tmp_path), 100), "error": None},
        {"id": "load-imbalance", "run_id": "case-2", "patch": "imb.patch", "run_dir": "r2", "error": None,
         **score(IMBALANCE, fixture("found-imbalance.json"), agent.OK, run_dir_with(tmp_path, ["t1"]), 200)},
        {"id": "memory-bound", "run_id": "case-3", "patch": "mem.patch", "run_dir": "r3",
         "expected": {"function": "ljForce", "bottleneck_types": list(MEMORY)},
         "status": "error", "error": "RuntimeError: patch does not apply", "correct": False},
    ]
    s = summarize(cases)
    assert s == {"cases": 3, "correct": 1, "completed": 2, "errors": 1, "injected_cases": 2, "function_found": 1,
                 "type_found": 1, "negative_controls": 1, "reference_functions": 0, "reference_reported": 0,
                 "reference_agreed": 0, "reference_not_scored": 0, "cases_with_false_positives": 1, "false_positives": 1,
                 "experiments": 1, "total_duration_s": 300.0}
    md = evaluate.render_markdown({"suite": "lulesh", "target": "lulesh", "model": "gpt-5", "started_at": "t",
                                   "suite_path": "suites/lulesh.yaml", "cases": cases, "summary": s})
    assert "**1 of 3 cases correct.**" in md
    assert "| negative-control (case-1) | none | nothing (negative control) | ok | n/a | n/a | " \
           f"{LULESH_FN} (load_imbalance) | 0 | 100.0 | no |" in md
    assert "| load-imbalance (case-2) | imb.patch | `CalcFBHourglassForceForElems`: load_imbalance | ok | yes | yes | " \
           "none | 1 | 200.0 | **yes** |" in md
    assert "| memory-bound (case-3) | mem.patch |" in md and "| error |" in md
    assert "Error: `RuntimeError: patch does not apply`" in md


# --- fresh source copies --------------------------------------------------------------------------

def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    (repo / "main.c").write_text("int main(void) { return 0; }\n")
    git("add", ".", cwd=repo)
    git("commit", "-qm", "upstream history that must not leak", cwd=repo)
    return repo, git("rev-parse", "HEAD", cwd=repo)


def test_prepare_source_applies_patch_without_history(tmp_path, upstream):
    repo, commit = upstream
    dest = tmp_path / "case" / "src"
    sha = evaluate.prepare_source(repo, commit, "https://example.invalid/up.git",
                                  FIXTURES / "patches" / "add-comment.patch", dest)
    assert (dest / "main.c").read_text() == "/* injected */\nint main(void) { return 0; }\n"
    assert git("rev-parse", "HEAD", cwd=dest) == sha != commit
    assert git("log", "--format=%s", cwd=dest) == "source"  # one neutral commit: the patch is not visible as a diff
    assert git("status", "--porcelain", cwd=dest) == ""
    assert git("remote", "get-url", "origin", cwd=dest) == "https://example.invalid/up.git"
    # deterministic: the same tree and patch give the same commit
    again = evaluate.prepare_source(repo, commit, "https://example.invalid/up.git",
                                    FIXTURES / "patches" / "add-comment.patch", tmp_path / "again")
    assert again == sha


def test_prepare_source_unmodified_and_bad_patch(tmp_path, upstream):
    repo, commit = upstream
    evaluate.prepare_source(repo, commit, "https://x.invalid/u.git", None, tmp_path / "plain")
    assert (tmp_path / "plain" / "main.c").read_text() == "int main(void) { return 0; }\n"
    with pytest.raises(RuntimeError, match="patch does-not-apply.patch does not apply"):
        evaluate.prepare_source(repo, commit, "https://x.invalid/u.git",
                                FIXTURES / "patches" / "does-not-apply.patch", tmp_path / "bad")


# --- the harness ------------------------------------------------------------------------------------

@pytest.fixture
def harness(monkeypatch, upstream, tmp_path):
    """Run the fixture suite with a fake agent: case-1 writes a clean control, case-2 finds the imbalance."""
    repo, commit = upstream
    monkeypatch.setattr(evaluate, "pinned_cache", lambda root, url, c: repo)
    real_load = evaluate.load_config

    def load_config(path, target):  # the fixture suite's config, pointed at the local upstream
        cfg = real_load(path, target)
        return cfg if Path(path).name == "config.yaml" else type(cfg)(**{**cfg.__dict__, "commit": commit})
    monkeypatch.setattr(evaluate, "load_config", load_config)
    calls = []

    def fake_run(cfg, config_path, root, run_id, *, client, mpi, limits, base_url, hidden):
        calls.append({"run_id": run_id, "root": root, "commit": cfg.commit, "mpi": mpi, "config": config_path,
                      "hidden": hidden})
        run_dir = Path(root) / "runs" / run_id
        (run_dir / "baseline").mkdir(parents=True)
        (run_dir / "baseline" / "perf-report.txt").write_text("x")
        name = {"case-1": "control-clean.json", "case-2": "found-imbalance.json"}[run_id]
        data = fixture(name)
        data["run_id"] = run_id
        (run_dir / "findings.json").write_text(json.dumps(data))
        if run_id == "case-2":
            (run_dir / "experiments" / "threads-1").mkdir(parents=True)
        return agent.OK
    monkeypatch.setattr(agent, "run", fake_run)
    return calls


def test_suite_runs_every_case_despite_a_failure(tmp_path, harness):
    suite = load_suite(FIXTURES / "suite.yaml")
    out = tmp_path / "out"
    results = evaluate.run_suite(suite, out, tmp_path, client=None, limits=agent.Limits(), base_url="u")

    assert [c["run_id"] for c in harness] == ["case-1", "case-2"]  # case-3's patch failed before the agent ran
    case1, case2, case3 = results["cases"]
    assert case1["id"] == "negative-control" and case1["correct"] and case1["error"] is None
    assert case2["id"] == "load-imbalance" and case2["correct"] and case2["matched_rank"] == 2
    assert case2["experiments"] == 1
    assert case3["id"] == "broken" and case3["status"] == "error" and not case3["correct"]
    assert "patch does-not-apply.patch does not apply" in case3["error"]
    assert (out / "case-3-error.txt").read_text().startswith("Traceback")
    assert results["summary"]["cases"] == 3 and results["summary"]["correct"] == 2 and results["summary"]["errors"] == 1

    saved = json.loads((out / "results.json").read_text())
    assert saved["summary"] == results["summary"] and saved["finished_at"]
    assert "**2 of 3 cases correct.**" in (out / "results.md").read_text()

    # each case ran on its own fresh copy, with the patch committed as the case's pinned commit
    assert (out / "cases/case-2/.work/lulesh/src/main.c").read_text().startswith("/* injected */")
    assert (out / "cases/case-1/.work/lulesh/src/main.c").read_text().startswith("int main")
    assert harness[0]["commit"] != harness[1]["commit"]
    assert yaml.safe_load(Path(harness[1]["config"]).read_text())["commit"] == harness[1]["commit"]


def test_run_ids_do_not_reveal_the_case(tmp_path, harness):
    evaluate.run_suite(load_suite(FIXTURES / "suite.yaml"), tmp_path / "out", tmp_path, None, agent.Limits(), "u")
    for call in harness:
        path = str(Path(call["root"]) / "runs" / call["run_id"])
        assert "imbalance" not in path and "control" not in path and "broken" not in path


def test_answers_and_other_cases_are_hidden_from_the_agent(tmp_path, harness):
    out = tmp_path / "out"
    evaluate.run_suite(load_suite(FIXTURES / "suite.yaml"), out, tmp_path, None, agent.Limits(), "u")
    hidden = set(harness[1]["hidden"])  # case-2
    assert FIXTURES.resolve() in hidden  # the suite file, its expectations and its patches
    assert {out / "results.json", out / "results.md", out / "cases/case-1", out / "cases/case-3"} <= hidden
    assert out / "cases/case-2" not in hidden


def test_hidden_access_attempts_are_counted(tmp_path):
    t = tmp_path / "transcript.jsonl"
    t.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "tool", "result": "ERROR: That path is outside the analysis and cannot be accessed."},
        {"kind": "tool", "result": "DENIED: That path is outside the analysis and cannot be accessed."},
        {"kind": "tool", "result": "exit code 0"}, {"kind": "response"}]) + "\nnot json\n")
    assert evaluate.hidden_attempts(t) == 2
    assert evaluate.hidden_attempts(tmp_path / "missing") == 0


def test_agent_crash_is_recorded_and_the_suite_continues(tmp_path, harness, monkeypatch):
    real = agent.run

    def crash_first(cfg, config_path, root, run_id, **kw):
        if run_id == "case-1":
            raise RuntimeError("agent exploded")
        return real(cfg, config_path, root, run_id, **kw)
    monkeypatch.setattr(agent, "run", crash_first)
    results = evaluate.run_suite(load_suite(FIXTURES / "suite.yaml"), tmp_path / "out", tmp_path, None,
                                 agent.Limits(), "u")
    assert results["cases"][0]["error"] == "RuntimeError: agent exploded"
    assert results["cases"][1]["correct"]


def test_exhausted_budget_stops_the_suite(tmp_path, harness, monkeypatch):
    def broke(cfg, config_path, root, run_id, **kw):
        harness.append(run_id)
        return agent.BUDGET_EXHAUSTED
    monkeypatch.setattr(agent, "run", broke)
    out = tmp_path / "out"
    results = evaluate.run_suite(load_suite(FIXTURES / "suite.yaml"), out, tmp_path, None, agent.Limits(), "u")
    assert harness == ["case-1"] and [c["status"] for c in results["cases"]] == ["API budget used up"]
    assert results["not_run"] == ["load-imbalance", "broken"]
    assert "Not run, because the API budget was used up:** load-imbalance, broken" in (out / "results.md").read_text()


def test_only_selected_cases(tmp_path, harness):
    results = evaluate.run_suite(load_suite(FIXTURES / "suite.yaml"), tmp_path / "out", tmp_path, None,
                                 agent.Limits(), "u", only=["load-imbalance"])
    assert [c["id"] for c in results["cases"]] == ["load-imbalance"] and results["cases"][0]["run_id"] == "case-1"


# --- CLI ----------------------------------------------------------------------------------------------

@pytest.fixture
def cli(monkeypatch, tmp_path):
    seen = {}

    def fake_run_suite(suite, out, root, client, limits, base_url, only=None, env=None):
        seen.update(suite=suite, out=out, limits=limits, only=only, client=client, env=env)
        return {"summary": {"correct": 1, "cases": 1, "errors": seen.get("errors", 0)}}
    monkeypatch.setattr(evaluate, "run_suite", fake_run_suite)
    monkeypatch.setenv("HOKIEAI_API", "k")
    monkeypatch.chdir(tmp_path)
    return seen


def test_cli_eval(cli, tmp_path, capsys):
    assert agent.main(["eval", "--suite", str(REPO / "suites/comd.yaml"), "--max-turns", "30", "--timeout", "40m",
                       "--cases", "load-imbalance,negative-control"]) == 0
    assert cli["suite"].name == "comd" and cli["only"] == ["load-imbalance", "negative-control"]
    assert cli["limits"] == agent.Limits(max_turns=30, timeout_s=2400)
    assert cli["out"].parent == (tmp_path / "evals").resolve() and cli["out"].name.startswith("comd-")
    assert "1 of 1 cases correct" in capsys.readouterr().out


def test_cli_eval_errors(cli, tmp_path, capsys):
    for argv, message in [
        (["eval", "--suite", "missing.yaml"], "suite file not found"),
        (["eval", "--suite", str(REPO / "suites/comd.yaml"), "--cases", "nope"], "unknown case id(s): nope"),
        (["eval", "--suite", str(REPO / "suites/comd.yaml"), "--timeout", "x"], "invalid duration"),
    ]:
        with pytest.raises(SystemExit) as exc:
            agent.main(argv)
        assert exc.value.code == 2 and message in capsys.readouterr().err
    (tmp_path / "taken").mkdir()
    with pytest.raises(SystemExit):
        agent.main(["eval", "--suite", str(REPO / "suites/comd.yaml"), "--out", "taken"])
    assert "already exists" in capsys.readouterr().err
    assert not cli


# --- reference classification (e.g. a published roofline study) ----------------------------------

def test_shipped_lulesh_reference():
    suite = load_suite(REPO / "suites" / "lulesh.yaml")
    assert suite.run == "{binary} -s 120 -i 40 -b 0 -c 0"
    control = {c.id: c for c in suite.cases}["negative-control"]
    ref = control.reference
    assert ref.variant == "openmp" and "2412.08792" in ref.url and ref.path.parent.name == "references"
    classes = dict(ref.classes)
    assert len(classes) == 6 and classes["CalcKinematicsForElems"] == "not_memory_bound"
    assert sum(c == "memory_bound" for c in classes.values()) == 5
    assert (ref.min_workers, ref.min_working_set_to_llc) == (13, 10)
    mpi, _ = load_reference(ref.path, "mpi")
    assert dict(mpi.classes)["IntegrateStressForElems"] == "not_memory_bound"
    # the memory-bound injection targets the one hot spot the reference finds not memory-bound
    assert {c.id: c for c in suite.cases}["memory-bound"].expect.function == "CalcKinematicsForElems"


REF, _ = load_reference(FIXTURES / "references" / "roofline.yaml", "openmp")
REF_CASE = Case(id="negative-control", description="", patch=None, mpi=False, expect=None,
                false_positives=(FB_IMBALANCE, KIN_MEMORY), reference=REF)


def test_score_reference_rows():
    result = score_reference(REF, fixture("reference-diagnosis.json")["findings"])
    rows = {r["function"]: r for r in result["functions"]}
    assert rows["CalcFBHourglassForceForElems"] == {
        "function": "CalcFBHourglassForceForElems", "reference_class": "memory_bound", "reported": True, "rank": 1,
        "agent_type": "memory_bandwidth_bound", "agent_class": "memory_bound", "agrees": True}
    assert rows["CalcKinematicsForElems"]["agent_class"] == "memory_bound" and rows["CalcKinematicsForElems"]["agrees"] is False
    assert rows["CalcHourglassControlForElems"]["reported"] is False and rows["CalcHourglassControlForElems"]["agrees"] is None
    assert (result["total"], result["reported"], result["agreed"]) == (3, 2, 1)


@pytest.mark.parametrize("bottleneck_type,agent_class,agrees", [
    ("memory_latency_bound", "memory_bound", True), ("memory_bandwidth_bound", "memory_bound", True),
    ("compute_bound", "not_memory_bound", False), ("poor_vectorization", "not_memory_bound", False),
    ("branch_misprediction", "not_memory_bound", False), ("load_imbalance", None, False), ("other", None, False),
])
def test_bottleneck_types_map_to_roofline_classes(bottleneck_type, agent_class, agrees):
    found = [{"rank": 1, "function": "CalcFBHourglassForceForElems", "bottleneck_type": bottleneck_type}]
    row = score_reference(REF, found)["functions"][1]
    assert (row["agent_class"], row["agrees"]) == (agent_class, agrees)


def test_reference_case_scoring(tmp_path):
    result = score(REF_CASE, fixture("reference-diagnosis.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert result["reference"]["agreed"] == 1 and result["reference"]["reported"] == 2
    # rank 2 (memory-bound in CalcKinematicsForElems) is what the memory-bound case injects: a false positive
    assert [f["rank"] for f in result["false_positives"]] == [2]
    assert not result["correct"]


def test_reference_case_correct_when_all_reported_functions_agree(tmp_path):
    findings = fixture("reference-diagnosis.json")
    findings["findings"][1]["bottleneck_type"] = "compute_bound"
    result = score(REF_CASE, findings, agent.OK, run_dir_with(tmp_path), 10)
    assert result["reference"]["agreed"] == result["reference"]["reported"] == 2
    assert result["false_positives"] == [] and result["correct"]
    nothing = score(REF_CASE, fixture("inconclusive.json"), agent.OK, run_dir_with(tmp_path), 10)
    assert nothing["reference"]["reported"] == 0 and not nothing["correct"]  # reporting nothing is not agreement


@pytest.mark.parametrize("content,variant,message", [
    ("source: s\nurl: u\nvariants: {openmp: {f: {class: memory_bound}}}\n", "mpi", "no variant 'mpi'"),
    ("source: s\nurl: u\nvariants: {openmp: {f: {class: fast}}}\n", "openmp", "f.class must be one of"),
    ("source: s\nurl: u\nvariants: {openmp: {'f(int)': {class: memory_bound}}}\n", "openmp", "not a plain function name"),
    ("url: u\nvariants: {openmp: {f: {class: memory_bound}}}\n", "openmp", "missing key 'source'"),
    ("source: s\nurl: u\nvariants: {openmp: {}}\n", "openmp", "must map function names to classes"),
    ("- a\n", "openmp", "top level must be a mapping"),
])
def test_invalid_reference(tmp_path, content, variant, message):
    (tmp_path / "r.yaml").write_text(content)
    ref, errors = load_reference(tmp_path / "r.yaml", variant)
    assert ref is None and any(message in e for e in errors), errors
    assert "reference file not found" in load_reference(tmp_path / "missing.yaml", "openmp")[1][0]


@pytest.mark.parametrize("change,message", [
    ({"run": "./lulesh -s 5"}, "run must be a command string using the {binary} placeholder"),
    ({"cases": [{"id": "a", "expect": "none", "reference": {"file": "x.yaml"}}]},
     "reference must have exactly 'file' and 'variant'"),
    ({"cases": [{"id": "a", "expect": {"function": "f", "bottleneck_types": ["load_imbalance"]},
                 "reference": {"file": "x.yaml", "variant": "openmp"}}]}, "reference applies to unmodified code"),
    ({"cases": [{"id": "a", "expect": "none", "reference": {"file": "missing.yaml", "variant": "openmp"}}]},
     "cases[0].reference file not found"),
])
def test_invalid_reference_in_suite(tmp_path, change, message):
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(tmp_path, {**BASE, **change}))
    assert message in str(exc.value)


def test_reference_summary_and_markdown(tmp_path):
    case = {"id": "negative-control", "run_id": "case-1", "patch": None, "expected": None, "run_dir": "r", "error": None,
            **score(REF_CASE, fixture("reference-diagnosis.json"), agent.OK, run_dir_with(tmp_path), 10)}
    s = summarize([case])
    assert (s["reference_functions"], s["reference_reported"], s["reference_agreed"]) == (3, 2, 1)
    md = evaluate.render_markdown({"suite": "lulesh", "target": "lulesh", "model": "m", "started_at": "t",
                                   "suite_path": "p", "cases": [case], "summary": s})
    assert "Reference classification: 1 of 2 reported hot functions agree (2 of 3 reported)." in md
    assert "| negative-control (case-1) | none | reference: 1 of 2 agree |" in md
    assert "Compared with Fixture study (https://example.invalid/paper), variant `openmp`:" in md
    assert "| `CalcFBHourglassForceForElems` | memory bound | rank 1: memory_bandwidth_bound | memory bound | yes |" in md
    assert "| `CalcHourglassControlForElems` | memory bound | not reported | - | n/a |" in md
    assert "| `CalcKinematicsForElems` | not memory bound | rank 2: memory_latency_bound | memory bound | no |" in md


def test_suite_run_command_reaches_the_case_config(tmp_path, harness):
    suite = load_suite(FIXTURES / "suite.yaml")
    suite = evaluate.Suite(**{**suite.__dict__, "run": "{binary} -s 7 -b 0"})
    evaluate.run_suite(suite, tmp_path / "out", tmp_path, None, agent.Limits(), "u", only=["negative-control"])
    assert yaml.safe_load(Path(harness[0]["config"]).read_text())["run"] == "{binary} -s 7 -b 0"


# --- reference conditions: only score against a reference on hardware like the study's -------------

@pytest.mark.parametrize("text,mib", [("260 MiB (1 instance)", 260), ("108 MiB (2 instances)", 54),
                                      ("512 KiB", 0.5), ("1 GiB", 1024), ("", None), (None, None)])
def test_cache_mib(text, mib):
    assert cache_mib(text) == mib





def conditional_reference():
    ref, _ = load_reference(FIXTURES / "references" / "roofline.yaml", "openmp")
    return evaluate.Reference(**{**ref.__dict__, "min_workers": 13, "min_working_set_to_llc": 10})


def baseline_meta(threads="4", rss=1525.0, l3="260 MiB (1 instance)", ranks=None):
    return {"mpi": {"ranks": ranks} if ranks else None, "peak_rss_mb": rss, "host": {"l3_cache": l3},
            "threads": {"variables": {"OMP_NUM_THREADS": threads} if threads else {}}}


@pytest.mark.parametrize("meta,applicable,reason", [
    (baseline_meta(threads="18", rss=600.0, l3="54 MiB (1 instance)"), True, None),
    (baseline_meta(threads="4", rss=600.0, l3="54 MiB (1 instance)"), False, "4 worker(s) (threads or ranks); the reference needs at least 13"),
    (baseline_meta(threads="18", rss=1525.0, l3="260 MiB (1 instance)"), False, "working set 1525 MiB is 5.9x the 260 MiB last-level cache"),
    (baseline_meta(threads=None, ranks=27, rss=1000.0, l3="108 MiB (2 instances)"), True, None),
    (baseline_meta(threads="18", rss=None), False, "the working set or the last-level cache size is unknown"),
    (None, False, "the baseline's metadata.json is missing"),
])
def test_reference_conditions(meta, applicable, reason):
    c = reference_conditions(conditional_reference(), meta)
    assert c["applicable"] is applicable
    if reason:
        assert any(reason in r for r in c["reasons"]), c["reasons"]


def test_unmet_conditions_report_but_do_not_score(tmp_path):
    run_dir = run_dir_with(tmp_path)
    (run_dir / "baseline").mkdir()
    (run_dir / "baseline" / "metadata.json").write_text(json.dumps(baseline_meta(threads="4")))
    case = Case(**{**REF_CASE.__dict__, "reference": conditional_reference()})
    findings = fixture("reference-diagnosis.json")
    findings["findings"][1]["bottleneck_type"] = "compute_bound"  # everything else would be correct
    result = score(case, findings, agent.OK, run_dir, 10)
    ref = result["reference"]
    assert ref["applicable"] is False and ref["workers"] == 4 and ref["agreed"] == 2  # still shown
    # not scored: the classification is not judged against the reference, only against the injections
    assert result["false_positives"] == [] and result["correct"]
    s = summarize([{"id": "c", "run_id": "case-1", **result}])
    assert s["reference_functions"] == 0 and s["reference_not_scored"] == 1
    md = evaluate.render_markdown({"suite": "l", "target": "lulesh", "model": "m", "started_at": "t", "suite_path": "p",
                                   "cases": [{"id": "c", "run_id": "case-1", "patch": None, "run_dir": "r", **result}],
                                   "summary": s})
    assert "**Not scored on this host**" in md and "(not scored on this host)" in md
    assert "Reference comparisons not scored on this host: 1." in md
    # the same diagnosis on a matching node is scored and correct
    (run_dir / "baseline" / "metadata.json").write_text(json.dumps(baseline_meta(threads="18", rss=600.0, l3="54 MiB")))
    good = score(case, findings, agent.OK, run_dir, 10)
    assert good["reference"]["applicable"] and good["false_positives"] == [] and good["correct"]


def test_invalid_conditions(tmp_path):
    (tmp_path / "r.yaml").write_text("source: s\nurl: u\nconditions: {min_workers: 0, cores: 3}\n"
                                     "variants: {openmp: {f: {class: memory_bound}}}\n")
    _, errors = load_reference(tmp_path / "r.yaml", "openmp")
    assert any("conditions may only set" in e for e in errors)


def test_env_reaches_every_case_config(tmp_path, harness):
    suite = load_suite(FIXTURES / "suite.yaml")
    suite = evaluate.Suite(**{**suite.__dict__, "env": (("OMP_PROC_BIND", "close"), ("OMP_NUM_THREADS", "8"))})
    results = evaluate.run_suite(suite, tmp_path / "out", tmp_path, None, agent.Limits(), "u",
                                 only=["negative-control"], env={"OMP_NUM_THREADS": "18"})
    env = yaml.safe_load(Path(harness[0]["config"]).read_text())["env"]
    assert env["OMP_NUM_THREADS"] == "18" and env["OMP_PROC_BIND"] == "close"  # the command line wins
    assert results["env"] == {"OMP_PROC_BIND": "close", "OMP_NUM_THREADS": "18"}


def test_suite_env_validation(tmp_path):
    with pytest.raises(SuiteError, match="env must map variable names"):
        load_suite(write_suite(tmp_path, {**BASE, "env": {"SAGE_X": 1}}))


def test_cli_eval_env(cli, capsys):
    assert agent.main(["eval", "--suite", str(REPO / "suites/lulesh.yaml"), "--env", "OMP_NUM_THREADS=18"]) == 0
    assert cli["env"] == {"OMP_NUM_THREADS": "18"}
    with pytest.raises(SystemExit):
        agent.main(["eval", "--suite", str(REPO / "suites/lulesh.yaml"), "--env", "bad"])
    assert "--env needs NAME=VALUE" in capsys.readouterr().err
