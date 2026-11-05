"""Validate a run's findings.json and generate report.md from it.

    python -m sage.findings validate RUN_DIR   # exit 0 if valid, 1 with one error per line if not
    python -m sage.findings render RUN_DIR     # validate, then write RUN_DIR/report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft7Validator

SCHEMA_PATH = Path(__file__).with_name("findings.schema.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text())
GENERATED = {"findings.json", "report.md"}  # outputs of the analysis, not evidence for it


class FindingsError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


def load(run_dir: Path) -> dict:
    """Read and fully validate RUN_DIR/findings.json; raise FindingsError listing every problem."""
    path = Path(run_dir) / "findings.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FindingsError([f"{path} not found"]) from None
    except (OSError, UnicodeDecodeError) as e:
        raise FindingsError([f"cannot read {path}: {e}"]) from None
    except json.JSONDecodeError as e:
        raise FindingsError([f"{path} is not valid JSON: {e}"]) from None
    errors = validate(data, Path(run_dir))
    if errors:
        raise FindingsError(errors)
    return data


def validate(data: object, run_dir: Path) -> list[str]:
    errors = [_message(e) for e in sorted(Draft7Validator(SCHEMA).iter_errors(data), key=lambda e: list(e.absolute_path))]
    if errors or not isinstance(data, dict):
        return errors  # the checks below assume the schema holds

    run_dir = Path(run_dir).resolve()
    if data["run_id"] != run_dir.name:
        errors.append(f"run_id: '{data['run_id']}' does not match the run directory '{run_dir.name}'")
    ranks = sorted(f["rank"] for f in data["findings"])
    if ranks != list(range(1, len(ranks) + 1)):
        errors.append(f"findings: ranks must be 1..{len(ranks)} with no gaps or repeats, got {ranks}")
    for i, finding in enumerate(data["findings"]):
        for j, artifact in enumerate(finding["artifact_paths"]):
            problem = _artifact_problem(run_dir, artifact)
            if problem:
                errors.append(f"findings[{i}].artifact_paths[{j}]: '{artifact}' {problem}")
    return errors


def _artifact_problem(run_dir: Path, artifact: str) -> str | None:
    path = (Path(artifact) if Path(artifact).is_absolute() else run_dir / artifact).resolve()
    if not path.is_relative_to(run_dir):
        return f"is outside the run directory {run_dir}"
    if not path.exists():
        return f"does not exist in the run directory {run_dir}"
    if path.relative_to(run_dir).as_posix() in GENERATED:
        return "is a generated output, not evidence; cite the perf artifacts instead"
    return None


def _message(error) -> str:
    where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path).lstrip(".") or "findings.json"
    field = error.absolute_path[-1] if error.absolute_path else None
    if field == "artifact_paths" and error.validator == "minItems":
        return f"{where}: finding has no artifact evidence; list at least one file in the run directory"
    if error.validator == "required" and "'artifact_paths'" in error.message:
        return f"{where}: finding has no artifact evidence; 'artifact_paths' is required"
    if field == "findings" and error.validator == "minItems":
        return f"{where}: status is bottlenecks_found but there are no findings (use inconclusive if nothing was established)"
    return f"{where}: {error.message}"


# --- report.md ------------------------------------------------------------------

TYPE_NAMES = {
    "compute_bound": "compute-bound", "memory_bandwidth_bound": "memory-bandwidth-bound",
    "memory_latency_bound": "memory-latency-bound", "branch_misprediction": "branch mispredictions",
    "poor_vectorization": "poor vectorization", "serial_section": "serial section",
    "load_imbalance": "load imbalance", "mpi_communication": "MPI communication",
    "synchronization": "synchronization", "kernel_or_system": "kernel / system time", "other": "other",
}


def render(data: dict, metadata: dict | None = None) -> str:
    """report.md text for validated findings (and the run's metadata.json, if any)."""
    cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
    code = lambda s: f"`{s}`"
    findings = sorted(data["findings"], key=lambda f: f["rank"])
    status = {"bottlenecks_found": "Bottlenecks found", "inconclusive": "Inconclusive"}[data["status"]]
    out = [f"# {data['target'].upper()} performance bottlenecks: run {data['run_id']}", "",
           f"**Status:** {status}", "", data["summary"], ""]

    if metadata:
        out += ["## Run", ""] + [f"- {line}" for line in _run_lines(metadata)] + [""]

    out += ["## Findings", ""]
    if not findings:
        out += ["No bottleneck was established with enough evidence. See the summary and limitations.", ""]
    else:
        out += ["| Rank | Function | Source | Self time | Bottleneck | Confidence |",
                "| ---: | --- | --- | ---: | --- | --- |"]
        out += [f"| {f['rank']} | {code(cell(f['function']))} | {code(cell(f['source_location']))} | "
                f"{f['sample_percent']:g}% | {TYPE_NAMES[f['bottleneck_type']]} | {f['confidence']} |" for f in findings]
        out.append("")
    for f in findings:
        out += [f"### {f['rank']}. {code(f['function'])}: {TYPE_NAMES[f['bottleneck_type']]} "
                f"({f['confidence']} confidence)", "",
                f"{code(f['source_location'])}, {f['sample_percent']:g}% of samples (self time)", "",
                "**Evidence**", "", *[f"- {e}" for e in f["supporting_evidence"]], "",
                f"**Likely cause:** {f['likely_cause']}", "",
                f"**Experiment:** {f['experiment']}", "",
                f"**Result:** {f['experiment_result']}", "",
                "**Artifacts:** " + ", ".join(f"[{p}]({p})" for p in f["artifact_paths"]), ""]

    if data.get("limitations"):
        out += ["## Limitations", "", *[f"- {item}" for item in data["limitations"]], ""]
    out += ["---", "", "Generated from `findings.json` by `python -m sage.findings render`.", ""]
    return "\n".join(out)


def _run_lines(m: dict) -> list[str]:
    lines = []
    variant = f" ({m['variant']}" + (f", {m['mpi']['ranks']} ranks" if m.get("mpi") else "") + ")" if m.get("variant") else ""
    lines.append(f"Target: {m.get('target')}{variant}" + (f"; label: {m['label']}" if m.get("label") else ""))
    if m.get("run_command"):
        lines.append(f"Command: `{m['run_command']}`" + (f" via `{m['mpi']['launcher']}`" if m.get("mpi") else ""))
    if b := m.get("build"):
        lines.append(f"Build: {b.get('repo_url')} @ `{str(b.get('commit'))[:12]}`, {b.get('compiler_version')}, "
                     f"`{b.get('flags')}`")
    w = m.get("wall_clock") or {}
    if w.get("app_mean_s") is not None:
        spread = f" ± {w['app_variance_pct']}%" if w.get("app_variance_pct") is not None else ""
        lines.append(f"Wall-clock: {w['app_mean_s']} s{spread} (mean of {w.get('repeats')} runs under perf stat)")
    if h := m.get("host"):
        lines.append(f"Host: {h.get('model', 'unknown CPU')}, {h.get('logical_cpus')} logical CPUs, kernel {h.get('kernel')}")
    perf = m.get("perf") or {}
    events = (perf.get("stat") or {}).get("events", [])
    if events:
        missing = [e["event"] for e in events if not e["counted"]]
        lines.append(f"perf: {perf.get('version')}; not available: {', '.join(missing) or 'none'}")
    return lines


def write_report(run_dir: Path) -> Path:
    """Validate RUN_DIR/findings.json and write RUN_DIR/report.md from it."""
    run_dir = Path(run_dir)
    data = load(run_dir)
    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    out = run_dir / "report.md"
    out.write_text(render(data, metadata))
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("validate", "render"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run_dir = Path(args[1])
    try:
        if args[0] == "validate":
            data = load(run_dir)
            print(f"findings.json is valid: {data['status']}, {len(data['findings'])} finding(s)")
        else:
            print(f"wrote {write_report(run_dir)}")
    except FindingsError as e:
        print(f"findings.json is invalid ({len(e.errors)} problem(s)):", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
