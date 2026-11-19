"""bottleneck-agent eval: run the agent on cases with known bottlenecks and score its diagnosis.

A suite YAML names a target and its cases. Each case is the pinned upstream source, optionally
changed by a patch file, and says what the agent should find: an injected bottleneck, or, for an
unmodified case, a published reference classification of the hot functions (e.g. a roofline study). Every case runs on a fresh copy of
the source, under an opaque id (case-1, case-2, ...), so neither the paths nor the source history
tell the agent what was injected. Results go to <out>/results.json and <out>/results.md.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from . import agent
from .config import TARGETS, ConfigError, load_config, placeholders
from .findings import SCHEMA, FindingsError, load as load_findings
from .guard import HIDDEN_REASON

BOTTLENECK_TYPES = SCHEMA["definitions"]["finding"]["properties"]["bottleneck_type"]["enum"]
CASE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
STATUS_NAMES = {0: "ok", 1: "failed", 3: "source modified", 4: "invalid findings", 5: "timed out",
                6: "API budget used up"}
REFERENCE_CLASSES = ("memory_bound", "not_memory_bound")
# How a finding's bottleneck type maps onto a roofline classification. Types that say nothing about
# memory versus in-core limits (load imbalance, synchronization, ...) map to None: no classification.
ROOFLINE_CLASS = {"memory_latency_bound": "memory_bound", "memory_bandwidth_bound": "memory_bound",
                  "compute_bound": "not_memory_bound", "poor_vectorization": "not_memory_bound",
                  "branch_misprediction": "not_memory_bound"}


class SuiteError(Exception):
    pass


@dataclass(frozen=True)
class Expect:
    function: str  # symbol name, matched against perf's (possibly decorated) function names
    bottleneck_types: tuple[str, ...]  # any of these counts as the right diagnosis


@dataclass(frozen=True)
class Reference:
    """A published classification of an unmodified application's hot functions."""
    source: str
    url: str
    variant: str
    classes: tuple[tuple[str, str], ...]  # (function, memory_bound | not_memory_bound), in the source's order
    path: Path
    min_workers: int | None = None  # the reference only applies with at least this many threads/ranks ...
    min_working_set_to_llc: float | None = None  # ... and a working set at least this many times the LLC


@dataclass(frozen=True)
class Case:
    id: str
    description: str
    patch: Path | None  # applied to the pinned source; None for an unmodified case
    mpi: bool
    expect: Expect | None  # None: a negative control, nothing was injected
    # bottlenecks injected by the suite's other cases; reporting one here is a false positive
    false_positives: tuple[Expect, ...] = ()
    reference: Reference | None = None  # unmodified cases only: score against a published classification


@dataclass(frozen=True)
class Suite:
    name: str
    target: str
    config: Path
    cases: tuple[Case, ...]
    path: Path
    run: str | None = None  # replaces the config's run command for every case ("{binary} ...")
    env: tuple[tuple[str, str], ...] = ()  # environment variables set for every case (over the config's env)


def load_reference(path: Path, variant: str) -> tuple[Reference | None, list[str]]:
    """Read a reference file; returns the reference for ``variant``, or the problems found."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        return None, [f"reference file not found: {path}"]
    except (OSError, yaml.YAMLError) as e:
        return None, [f"cannot read reference {path}: {e}"]
    if not isinstance(data, dict):
        return None, [f"reference {path}: top level must be a mapping"]
    errors = [f"reference {path}: missing key '{k}'" for k in ("source", "url", "variants") if k not in data]
    variants = data.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        return None, errors + [f"reference {path}: no variant '{variant}' under 'variants'"]
    functions = variants[variant]
    if not isinstance(functions, dict) or not functions:
        return None, errors + [f"reference {path}: variant '{variant}' must map function names to classes"]
    classes = []
    for fn, entry in functions.items():
        cls = entry.get("class") if isinstance(entry, dict) else None
        if not re.fullmatch(r"[A-Za-z_]\w*", str(fn)):
            errors.append(f"reference {path}: '{fn}' is not a plain function name")
        elif cls not in REFERENCE_CLASSES:
            errors.append(f"reference {path}: {fn}.class must be one of {', '.join(REFERENCE_CLASSES)}, got {cls!r}")
        else:
            classes.append((fn, cls))
    conditions = data.get("conditions") or {}
    if not isinstance(conditions, dict) or set(conditions) - {"min_workers", "min_working_set_to_llc"}:
        errors.append(f"reference {path}: conditions may only set min_workers and min_working_set_to_llc")
        conditions = {}
    for key, value in conditions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"reference {path}: conditions.{key} must be a positive number, got {value!r}")
    if errors:
        return None, errors
    return Reference(source=str(data["source"]), url=str(data["url"]), variant=variant, classes=tuple(classes),
                     path=Path(path).resolve(), min_workers=conditions.get("min_workers"),
                     min_working_set_to_llc=conditions.get("min_working_set_to_llc")), []


def cache_mib(text: str | None) -> float | None:
    """lscpu's cache size, per instance: '260 MiB (1 instance)' -> 260, '108 MiB (2 instances)' -> 54."""
    m = re.match(r"\s*([\d.]+)\s*([KMG])i?B?(?:\s*\((\d+) instances?\))?", text or "")
    if not m:
        return None
    size = float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)]
    return size / int(m.group(3) or 1)


def reference_conditions(reference: Reference, baseline: dict | None) -> dict:
    """Whether a case's baseline run met the reference's conditions, from its metadata.json."""
    baseline = baseline or {}
    mpi = baseline.get("mpi") or {}
    threads = (baseline.get("threads") or {}).get("variables", {}).get("OMP_NUM_THREADS", "")
    workers = mpi.get("ranks") or (int(threads) if str(threads).isdigit() else 1)
    peak, llc = baseline.get("peak_rss_mb"), cache_mib((baseline.get("host") or {}).get("l3_cache"))
    ratio = round(peak / llc, 1) if peak and llc else None
    reasons = []
    if not baseline and (reference.min_workers or reference.min_working_set_to_llc):
        reasons.append("the baseline's metadata.json is missing")
    if reference.min_workers and workers < reference.min_workers:
        reasons.append(f"{workers} worker(s) (threads or ranks); the reference needs at least {reference.min_workers}")
    if reference.min_working_set_to_llc and baseline:
        if ratio is None:
            reasons.append("the working set or the last-level cache size is unknown")
        elif ratio < reference.min_working_set_to_llc:
            reasons.append(f"working set {peak:g} MiB is {ratio}x the {llc:g} MiB last-level cache; the reference "
                           f"needs at least {reference.min_working_set_to_llc:g}x")
    return {"applicable": not reasons, "reasons": reasons, "workers": workers, "peak_rss_mb": peak,
            "llc_mib": llc, "working_set_to_llc": ratio}


def load_suite(path: Path) -> Suite:
    """Read and validate a suite; every problem is reported at once."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise SuiteError(f"suite file not found: {path}") from None
    except (OSError, yaml.YAMLError) as e:
        raise SuiteError(f"cannot read suite {path}: {e}") from None
    if not isinstance(data, dict):
        raise SuiteError(f"invalid suite {path}:\n  - top level must be a mapping")
    errors = [f"missing required key '{k}'" for k in ("name", "target", "config", "cases") if k not in data]
    errors += [f"unknown key '{k}'" for k in data
               if k not in ("name", "target", "config", "cases", "description", "run", "env")]
    base = path.parent
    name, target = data.get("name"), data.get("target")
    if "name" in data and not (isinstance(name, str) and CASE_ID.fullmatch(name)):
        errors.append(f"name must be lowercase letters, digits and '-', got {name!r}")
    if "target" in data and target not in TARGETS:
        errors.append(f"target must be one of {', '.join(TARGETS)}, got {target!r}")
    cfg = None
    if isinstance(data.get("config"), str) and target in TARGETS:
        try:
            cfg = load_config(base / data["config"], target)
        except ConfigError as e:
            errors.append(f"config: {e}")
    elif "config" in data:
        errors.append("config must be a path to the target config, relative to the suite file")
    suite_env = data.get("env") or {}
    if not isinstance(suite_env, dict) or not all(
            re.fullmatch(r"[A-Za-z_]\w*", str(k)) and not str(k).startswith("SAGE_")
            and isinstance(v, (str, int, float)) and not isinstance(v, bool) for k, v in suite_env.items()):
        errors.append("env must map variable names (not SAGE_*) to strings or numbers")
        suite_env = {}
    suite_run = data.get("run")
    if suite_run is not None and placeholders(suite_run) != {"binary"}:
        errors.append("run must be a command string using the {binary} placeholder and no other {…} fields")

    cases, seen = [], set()
    if "cases" in data and (not isinstance(data["cases"], list) or not data["cases"]):
        errors.append("cases must be a non-empty list")
    for i, raw in enumerate(data.get("cases") or []):
        where = f"cases[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{where} must be a mapping")
            continue
        errors += [f"{where}: unknown key '{k}'" for k in raw
                   if k not in ("id", "description", "patch", "mpi", "expect", "reference")]
        cid = raw.get("id")
        if not (isinstance(cid, str) and CASE_ID.fullmatch(cid)):
            errors.append(f"{where}.id must be lowercase letters, digits and '-', got {cid!r}")
        elif cid in seen:
            errors.append(f"{where}.id '{cid}' is used twice")
        seen.add(cid)
        patch = None
        if raw.get("patch") is not None:
            if not isinstance(raw["patch"], str) or not (base / raw["patch"]).is_file():
                errors.append(f"{where}.patch: file not found: {base / str(raw['patch'])}")
            else:
                patch = (base / raw["patch"]).resolve()
        mpi = raw.get("mpi", False)
        if not isinstance(mpi, bool):
            errors.append(f"{where}.mpi must be true or false")
        elif mpi and cfg is not None and cfg.mpi is None:
            errors.append(f"{where}.mpi is true but the target config has no mpi section")
        expect = None
        if "expect" not in raw:
            errors.append(f"{where}.expect is required: a mapping, or 'none' for a negative control")
        elif raw["expect"] != "none":
            e = raw["expect"]
            if not isinstance(e, dict) or set(e) != {"function", "bottleneck_types"}:
                errors.append(f"{where}.expect must have exactly 'function' and 'bottleneck_types' (or be 'none')")
            else:
                if not (isinstance(e["function"], str) and re.fullmatch(r"[A-Za-z_]\w*", e["function"])):
                    errors.append(f"{where}.expect.function must be a plain function name, got {e['function']!r}")
                types = e["bottleneck_types"]
                errors += _check_types(types, f"{where}.expect.bottleneck_types", required=True)
                if not errors:
                    expect = Expect(function=e["function"], bottleneck_types=tuple(types))
        reference = None
        if "reference" in raw:
            r = raw["reference"]
            if not isinstance(r, dict) or set(r) != {"file", "variant"}:
                errors.append(f"{where}.reference must have exactly 'file' and 'variant'")
            elif raw.get("expect") != "none" or patch is not None:
                errors.append(f"{where}.reference applies to unmodified code: it needs 'expect: none' and no patch")
            else:
                reference, problems = load_reference(base / str(r["file"]), str(r["variant"]))
                errors += [f"{where}.{e}" for e in problems]
        if expect is None and raw.get("expect") != "none":
            continue
        if isinstance(cid, str):
            cases.append(Case(id=cid, description=str(raw.get("description", "")), patch=patch, mpi=bool(mpi),
                              expect=expect, reference=reference))
    if errors:
        raise SuiteError(f"invalid suite {path}:\n" + "\n".join(f"  - {e}" for e in errors))
    cases = [replace(c, false_positives=injected_elsewhere(c, cases)) for c in cases]
    return Suite(name=name, target=target, config=(base / data["config"]).resolve(), cases=tuple(cases), path=path,
                 run=suite_run, env=tuple((str(k), str(v)) for k, v in suite_env.items()))


def injected_elsewhere(case: Case, cases: list[Case]) -> tuple[Expect, ...]:
    """The bottlenecks the other cases inject, less any this case expects for the same function."""
    out = []
    for other in cases:
        if other.expect is None or other is case:
            continue
        own = case.expect.bottleneck_types if case.expect and case.expect.function == other.expect.function else ()
        types = tuple(t for t in other.expect.bottleneck_types if t not in own)
        if types and Expect(other.expect.function, types) not in out:
            out.append(Expect(other.expect.function, types))
    return tuple(out)


def _check_types(value, where: str, required: bool = False) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        return [f"{where} must be a {'non-empty ' if required else ''}list of bottleneck types"]
    bad = [t for t in value if t not in BOTTLENECK_TYPES]
    return [f"{where}: unknown bottleneck type(s) {bad}; use {', '.join(BOTTLENECK_TYPES)}"] if bad else []


# --- scoring ---------------------------------------------------------------------------

def function_matches(reported: str, expected: str) -> bool:
    """perf names look like 'Foo(Domain&, int) [clone ._omp_fn.0]' or 'ns::Foo'; match the bare name."""
    return re.search(rf"(?<![\w]){re.escape(expected)}(?![\w])", reported) is not None


def score(case: Case, findings: dict | None, status: int, run_dir: Path, duration_s: float) -> dict:
    """Score one case from its findings.json (None if missing or invalid) and run directory."""
    found = (findings or {}).get("findings", [])
    experiments = sorted(p.name for p in (run_dir / "experiments").iterdir() if p.is_dir()) \
        if (run_dir / "experiments").is_dir() else []
    result = {
        "status": STATUS_NAMES.get(status, f"exit {status}"), "exit_code": status,
        "findings_valid": findings is not None,
        "findings_status": findings.get("status") if findings else None,
        "findings": [{"rank": f["rank"], "function": f["function"], "bottleneck_type": f["bottleneck_type"],
                      "confidence": f["confidence"]} for f in sorted(found, key=lambda f: f["rank"])],
        "duration_s": round(float(duration_s), 1),
        "experiments": len(experiments), "experiment_names": experiments,
        "expected": asdict(case.expect) if case.expect else None,
        "function_found": None, "type_found": None, "matched_rank": None,
    }
    if case.expect:
        hits = [f for f in found if function_matches(f["function"], case.expect.function)]
        right = [f for f in hits if f["bottleneck_type"] in case.expect.bottleneck_types]
        result["function_found"] = bool(hits)
        result["type_found"] = bool(right)
        result["matched_rank"] = min(f["rank"] for f in right) if right else None
    result["reference"] = None
    if case.reference:
        baseline_meta = run_dir / "baseline" / "metadata.json"
        conditions = reference_conditions(
            case.reference, json.loads(baseline_meta.read_text()) if baseline_meta.is_file() else None)
        result["reference"] = {**score_reference(case.reference, found), **conditions}
    scored_reference = case.reference if result["reference"] and result["reference"]["applicable"] else None
    result["false_positives"] = [{"rank": f["rank"], "function": f["function"], "bottleneck_type": f["bottleneck_type"]}
                                 for f in sorted(found, key=lambda f: f["rank"])
                                 if any(function_matches(f["function"], e.function)
                                        and f["bottleneck_type"] in e.bottleneck_types for e in case.false_positives)]
    reference_ok = (scored_reference is None
                    or (result["reference"]["reported"] > 0
                        and result["reference"]["agreed"] == result["reference"]["reported"]))
    result["correct"] = (findings is not None and status == agent.OK and not result["false_positives"]
                         and (case.expect is None or result["type_found"]) and reference_ok)
    return result


def score_reference(reference: Reference, found: list[dict]) -> dict:
    """Compare the agent's classification of each reference function with the reference's.

    A function counts as reported if a finding names it; its class comes from the best-ranked such
    finding. A type with no roofline meaning (e.g. load_imbalance) disagrees with either class.
    """
    rows = []
    for function, cls in reference.classes:
        hits = sorted((f for f in found if function_matches(f["function"], function)), key=lambda f: f["rank"])
        agent_type = hits[0]["bottleneck_type"] if hits else None
        agent_class = ROOFLINE_CLASS.get(agent_type) if agent_type else None
        rows.append({"function": function, "reference_class": cls, "reported": bool(hits),
                     "rank": hits[0]["rank"] if hits else None, "agent_type": agent_type,
                     "agent_class": agent_class, "agrees": (agent_class == cls) if hits else None})
    return {"source": reference.source, "url": reference.url, "variant": reference.variant, "functions": rows,
            "total": len(rows), "reported": sum(r["reported"] for r in rows),
            "agreed": sum(bool(r["agrees"]) for r in rows)}


def summarize(cases: list[dict]) -> dict:
    injected = [c for c in cases if c.get("expected")]
    controls = [c for c in cases if not c.get("expected")]
    finished = [c for c in cases if c.get("duration_s") is not None]
    return {
        "cases": len(cases),
        "correct": sum(bool(c.get("correct")) for c in cases),
        "completed": sum(c.get("exit_code") == agent.OK for c in cases),
        "errors": sum(c.get("error") is not None for c in cases),
        "injected_cases": len(injected),
        "function_found": sum(bool(c.get("function_found")) for c in injected),
        "type_found": sum(bool(c.get("type_found")) for c in injected),
        "negative_controls": len(controls),
        "reference_functions": sum(c["reference"]["total"] for c in cases if _scored(c)),
        "reference_reported": sum(c["reference"]["reported"] for c in cases if _scored(c)),
        "reference_agreed": sum(c["reference"]["agreed"] for c in cases if _scored(c)),
        "reference_not_scored": sum(bool(c.get("reference")) and not _scored(c) for c in cases),
        "cases_with_false_positives": sum(bool(c.get("false_positives")) for c in cases),
        "false_positives": sum(len(c.get("false_positives") or []) for c in cases),
        "experiments": sum(c.get("experiments") or 0 for c in cases),
        "total_duration_s": round(sum(c["duration_s"] for c in finished), 1),
    }


def _scored(case: dict) -> bool:
    return bool(case.get("reference")) and case["reference"].get("applicable", True)


def render_markdown(results: dict) -> str:
    s = results["summary"]
    yes = lambda v: "n/a" if v is None else ("yes" if v else "no")
    out = [f"# Evaluation: {results['suite']} ({results['target']})", "",
           f"Model `{results['model']}`, started {results['started_at']}, suite `{results['suite_path']}`.", "",
           f"**{s['correct']} of {s['cases']} cases correct.** Injected bottlenecks: function found in "
           f"{s['function_found']} of {s['injected_cases']}, function and type found in {s['type_found']} of "
           f"{s['injected_cases']}. Cases with false positives: {s['cases_with_false_positives']}. "
           f"{s['experiments']} experiment run(s), {s['total_duration_s']:g} s in total."
           + (f" Reference classification: {s['reference_agreed']} of {s['reference_reported']} reported hot "
              f"functions agree ({s['reference_reported']} of {s['reference_functions']} reported)."
              if s.get("reference_functions") else "")
           + (f" Reference comparisons not scored on this host: {s['reference_not_scored']}."
              if s.get("reference_not_scored") else ""), "",
           "| Case | Injected | Expected | Status | Function found | Type found | False positives | Experiments | Time (s) | Correct |",
           "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |"]
    for c in results["cases"]:
        expected = (f"`{c['expected']['function']}`: {' or '.join(c['expected']['bottleneck_types'])}"
                    if c.get("expected") else
                    (f"reference: {c['reference']['agreed']} of {c['reference']['reported']} agree"
                     + ("" if _scored(c) else " (not scored on this host)"))
                    if c.get("reference") else "nothing (negative control)")
        fps = ", ".join(f"{f['function']} ({f['bottleneck_type']})" for f in c.get("false_positives") or []) or "none"
        out.append(f"| {c['id']} ({c['run_id']}) | {c['patch'] or 'none'} | {expected} | "
                   f"{c.get('status', 'error')} | {yes(c.get('function_found'))} | {yes(c.get('type_found'))} | "
                   f"{fps.replace('|', '/')} | {c.get('experiments', 0)} | {c.get('duration_s') or '-'} | "
                   f"{'**yes**' if c.get('correct') else 'no'} |")
    out.append("")
    if results.get("not_run"):
        out += [f"**Not run, because the API budget was used up:** {', '.join(results['not_run'])}. "
                "Rerun them with `--cases` once it is renewed.", ""]
    for c in results["cases"]:
        out += [f"## {c['id']}", "", c.get("description") or "", ""]
        if c.get("error"):
            out += [f"Error: `{c['error']}`", ""]
        if c.get("reference"):
            ref = c["reference"]
            out += [f"Compared with {ref['source']} ({ref['url']}), variant `{ref['variant']}`:", ""]
            if not ref.get("applicable", True):
                out += ["**Not scored on this host**: the run did not meet the reference's conditions ("
                        + "; ".join(ref["reasons"]) + "). The comparison is shown for information.", ""]
            out += [
                    "| Function | Reference | Agent's finding | Agent's class | Agrees |",
                    "| --- | --- | --- | --- | --- |"]
            for row in ref["functions"]:
                finding = f"rank {row['rank']}: {row['agent_type']}" if row["reported"] else "not reported"
                out.append(f"| `{row['function']}` | {row['reference_class'].replace('_', ' ')} | {finding} | "
                           f"{(row['agent_class'] or 'no roofline class').replace('_', ' ') if row['reported'] else '-'} | "
                           f"{yes(row['agrees'])} |")
            out.append("")
        if c.get("findings"):
            out += ["Reported findings:", ""] + [f"{f['rank']}. `{f['function']}`: {f['bottleneck_type']} "
                                                  f"({f['confidence']} confidence)" for f in c["findings"]] + [""]
        elif c.get("findings_valid") is False:
            out += ["No valid findings.json.", ""]
        out += [f"Run directory: `{c['run_dir']}`", ""]
    return "\n".join(out)


# --- running ---------------------------------------------------------------------------

def git(*args: str, cwd: Path, input: bytes | None = None) -> bytes:
    env = {**os.environ, "GIT_AUTHOR_NAME": "sage", "GIT_AUTHOR_EMAIL": "sage@localhost",
           "GIT_COMMITTER_NAME": "sage", "GIT_COMMITTER_EMAIL": "sage@localhost",
           "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"}
    r = subprocess.run(["git", *args], cwd=cwd, env=env, input=input, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def pinned_cache(root: Path, repo_url: str, commit: str) -> Path:
    """A clone of the upstream that contains the pinned commit (reused across cases and suites)."""
    cache = root / ".work" / "eval-cache" / re.sub(r"\W+", "-", repo_url).strip("-")
    if not (cache / ".git").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", repo_url, str(cache), cwd=cache.parent)
    if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=cache, capture_output=True).returncode:
        git("fetch", "--quiet", "origin", cwd=cache)
    return cache


def prepare_source(cache: Path, commit: str, repo_url: str, patch: Path | None, dest: Path) -> str:
    """dest = the pinned tree (+ patch) as a one-commit repo; returns that commit's SHA.

    The copy has no upstream history, so `git log`/`git diff` cannot reveal the injected change.
    """
    dest.mkdir(parents=True)
    tar = git("archive", "--format=tar", commit, cwd=cache)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)
    git("init", "--quiet", cwd=dest)
    if patch:
        try:
            git("apply", "--check", str(patch), cwd=dest)
        except RuntimeError as e:
            raise RuntimeError(f"patch {patch.name} does not apply to {commit[:12]}: {e}") from None
        git("apply", str(patch), cwd=dest)
    git("add", "-A", cwd=dest)
    git("commit", "--quiet", "-m", "source", cwd=dest)
    git("remote", "add", "origin", repo_url, cwd=dest)
    return git("rev-parse", "HEAD", cwd=dest).decode().strip()


def run_case(suite: Suite, case: Case, run_id: str, case_dir: Path, root: Path, client, limits: agent.Limits,
             base_url: str, hidden: tuple[Path, ...] = (), env: dict[str, str] | None = None) -> dict:
    """Prepare and run one case. Raises on setup problems; the caller records them."""
    cfg = load_config(suite.config, suite.target)
    cache = pinned_cache(root, cfg.repo_url, cfg.commit)
    commit = prepare_source(cache, cfg.commit, cfg.repo_url, case.patch, case_dir / ".work" / suite.target / "src")
    raw = yaml.safe_load(suite.config.read_text())
    raw["commit"] = commit
    if suite.run:  # e.g. the settings a reference study used
        (raw["mpi"] if case.mpi else raw)["run"] = suite.run
    if env:  # e.g. OMP_NUM_THREADS for a node like the reference study's
        raw["env"] = {**(raw.get("env") or {}), **env}
        if case.mpi:
            raw["mpi"]["env"] = {**(raw["mpi"].get("env") or {}), **env}  # the case's own source: pinned + patch, verified by the source check
    config_path = case_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    case_cfg = load_config(config_path, suite.target)
    start = time.monotonic()
    status = agent.run(case_cfg, config_path, case_dir, run_id, client=client, mpi=case.mpi, limits=limits,
                       base_url=base_url, hidden=hidden)
    duration = time.monotonic() - start
    run_dir = case_dir / "runs" / run_id
    try:
        findings = load_findings(run_dir)
    except FindingsError:
        findings = None
    result = score(case, findings, status, run_dir, duration)
    result["hidden_access_attempts"] = hidden_attempts(run_dir / "transcript.jsonl")
    return result


def hidden_attempts(transcript: Path) -> int:
    """How often the agent tried to read the suite's answers or other cases (all were refused)."""
    if not transcript.is_file():
        return 0
    count = 0
    for line in transcript.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "tool" and HIDDEN_REASON in str(record.get("result", "")):
            count += 1
    return count


def run_suite(suite: Suite, out: Path, root: Path, client, limits: agent.Limits, base_url: str,
              only: list[str] | None = None, env: dict[str, str] | None = None) -> dict:
    out.mkdir(parents=True)
    env = {**dict(suite.env), **(env or {})}
    results = {"suite": suite.name, "suite_path": str(suite.path), "target": suite.target, "model": limits.model,
               "limits": asdict(limits), "base_url": base_url, "env": env, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "cases": [], "summary": {}}
    cases = [c for c in suite.cases if not only or c.id in only]
    for n, case in enumerate(cases, 1):
        run_id = f"case-{n}"  # opaque: the agent must not learn what was injected from names
        case_dir = out / "cases" / run_id
        record = {"id": case.id, "run_id": run_id, "description": case.description,
                  "patch": case.patch.name if case.patch else None, "mpi": case.mpi,
                  "expected": asdict(case.expect) if case.expect else None,
                  "run_dir": str(case_dir / "runs" / run_id), "error": None}
        print(f"\n=== eval {suite.name}: {case.id} as {run_id} ({n}/{len(cases)}) ===", file=sys.stderr)
        # The agent must not see the expectations, the patches, the results so far or the other cases.
        hidden = (suite.path.parent.resolve(), out / "results.json", out / "results.md",
                  *[out / "cases" / f"case-{k}" for k in range(1, len(cases) + 1) if k != n])
        try:
            record.update(run_case(suite, case, run_id, case_dir, root, client, limits, base_url, hidden, env))
        except Exception as e:  # one broken case must not stop the others
            record.update({"status": "error", "error": f"{type(e).__name__}: {e}", "correct": False})
            (out / f"{run_id}-error.txt").write_text(traceback.format_exc())
            print(f"eval: {case.id} failed: {e}", file=sys.stderr)
        results["cases"].append(record)
        results["summary"] = summarize(results["cases"])
        write_results(results, out)  # after every case, so partial results survive an interruption
        if record.get("exit_code") == agent.BUDGET_EXHAUSTED:  # every later case would fail the same way
            results["not_run"] = [c.id for c in cases[n:]]
            if results["not_run"]:
                print(f"eval: the API budget is used up; not running {', '.join(results['not_run'])}", file=sys.stderr)
            break
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_results(results, out)
    return results


def write_results(results: dict, out: Path):
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "results.md").write_text(render_markdown(results))
