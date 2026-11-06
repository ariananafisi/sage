"""Safety rails for the agent, enforced in a PreToolUse hook and checked again after the run.

The hook (``make_guard``) denies:
  * profilers other than perf;
  * shell commands that would edit, move or delete a file tracked in the source checkout;
  * the Write/Edit tools on anything but the run's findings.json;
  * running the application other than through scripts/profile_target.sh (direct ``perf
    stat/record``, the binary itself except ``--help``, MPI launchers), so every run is recorded;
  * artifact sets outside <run>/baseline and <run>/experiments, and experiments beyond the limit.

``source_changes`` compares every file of the pinned commit with the checkout by content
afterwards, which catches what a command check cannot see (e.g. a write from ``python -c``).
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OTHER_PROFILERS = re.compile(
    r"\b(gprof|valgrind|callgrind|cachegrind|vtune|amplxe-cl|advixe-cl|"
    r"likwid-[\w-]+|hpcrun|hpcstruct|hpcprof|nsys|ncu|nvprof|tau_exec|tau_run|"
    r"scorep|scalasca|uftrace|bpftrace|py-spy|pprof|operf|ocount|opcontrol|"
    r"strace|ltrace)\b"
    r"|(?<![\w-])-pg\b"
)

EDITORS = {"vi", "vim", "nvim", "nano", "emacs", "ed", "ex", "patch"}
GIT_READ_ONLY = {"status", "log", "show", "diff", "blame", "ls-files", "ls-tree", "rev-parse", "grep",
                 "cat-file", "describe", "shortlog", "--version", "version", "help"}
WRITES_ALL_ARGS = {"rm", "mv", "tee", "touch", "truncate", "shred", "unlink"}
WRITES_LAST_ARG = {"cp", "install", "ln", "rsync"}
SHELLS = {"sh", "bash", "zsh", "dash"}
WRAPPERS = {"sudo", "env", "nice", "nohup", "time", "command", "exec", "xargs", "timeout", "taskset", "numactl"}
SEPARATORS = {";", "&&", "||", "|", "|&", "&", "(", ")", "\n"}
PERF_READ_ONLY = {"report", "annotate", "script", "diff", "evlist", "list", "version", "help",
                  "buildid-list", "data", "--version", "-v", "--help", "-h"}
LAUNCHERS = {"mpirun", "mpiexec", "orterun", "srun", "aprun", "jsrun"}
HELP_FLAGS = {"-h", "--help", "-help"}
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


def tracked(src: Path) -> set[str]:
    """Absolute paths of the files git tracks in ``src`` (empty if it is not a checkout yet)."""
    if not (Path(src) / ".git").exists():
        return set()
    r = subprocess.run(["git", "-C", str(src), "ls-files", "-z"], capture_output=True)
    if r.returncode:
        return set()
    root = Path(src).resolve()
    return {str(root / p) for p in r.stdout.decode().split("\0") if p}


def source_changes(src: Path, commit: str) -> list[str]:
    """Differences between the checkout and the pinned ``commit``, compared by file content."""
    src = Path(src)
    if not src.exists():
        return []  # never checked out during this run
    if not (src / ".git").exists():
        return [f"{src} is no longer a git checkout; its source cannot be verified"]
    git = lambda *a, **kw: subprocess.run(["git", "-C", str(src), *a], capture_output=True, **kw)
    problems = []
    head = git("rev-parse", "HEAD").stdout.decode().strip()
    if head != commit:
        problems.append(f"checkout HEAD is {head or '(none)'}, expected the pinned commit {commit}")
    tree = git("ls-tree", "-r", "-z", commit)
    if tree.returncode:
        return problems + [f"pinned commit {commit} is missing from {src}"]
    expected = {}
    for entry in tree.stdout.decode().split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob" and mode in ("100644", "100755"):
                expected[path] = sha
    present = [p for p in expected if (src / p).is_file()]
    problems += [f"{src / p} was deleted" for p in expected if p not in present]
    hashed = git("hash-object", "--no-filters", "--stdin-paths", input="\n".join(present).encode())
    actual = hashed.stdout.decode().split()
    if hashed.returncode or len(actual) != len(present):
        return problems + [f"could not hash the files in {src}"]
    problems += [f"{src / p} was modified" for p, sha in zip(present, actual) if sha != expected[p]]
    return problems


@dataclass(frozen=True)
class Policy:
    """What one agent run may do."""
    src: Path  # the source checkout (may not exist until the agent builds)
    run_dir: Path
    binaries: frozenset[str]  # basenames of the target's binaries
    max_experiments: int

    @property
    def experiments(self) -> Path:
        return self.run_dir / "experiments"

    def experiments_used(self) -> int:
        return sum(1 for p in self.experiments.iterdir() if p.is_dir()) if self.experiments.is_dir() else 0


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _segments(tokens: list[str]) -> list[list[str]]:
    segments, current = [], []
    for tok in tokens:
        if tok in SEPARATORS:
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)
    return segments


def _starts(seg: list[str]) -> list[int]:
    """Indexes where a command starts: after wrappers / VAR=val, and after find's -exec."""
    i = 0
    while i < len(seg) and (
        seg[i] in WRAPPERS or "=" in seg[i]
        or (i > 0 and seg[i - 1] in WRAPPERS and (seg[i].startswith("-") or seg[i][:1].isdigit()))
    ):
        i += 1
    return [i] + [j + 1 for j, t in enumerate(seg) if t in {"-exec", "-execdir", "-ok"}]


def _nested(verb: str, args: list[str]) -> str | None:
    """The command string inside ``bash -c '...'`` or ``eval ...``."""
    if verb in SHELLS and "-c" in args and args.index("-c") + 1 < len(args):
        return args[args.index("-c") + 1]
    if verb == "eval" and args:
        return " ".join(args)
    return None


def check_command(command: str, protected: set[str], cwd: str = "/") -> str | None:
    """Return a reason to deny ``command`` (other profilers, source edits), or None."""
    m = OTHER_PROFILERS.search(command)
    if m:
        return f"'{m.group(0)}' is not allowed: perf is the only permitted profiler."
    try:
        tokens = _tokens(command)
    except ValueError:
        return "Could not parse the command; simplify the quoting."

    def hits(token: str) -> bool:
        if not token or token.startswith("-"):
            return False
        # Match the path as resolved from cwd, and (since the command may `cd`
        # first) as a path suffix of any protected file. Globs are honoured.
        patterns = {os.path.normpath(os.path.join(cwd, token))}
        if not os.path.isabs(token):
            rel = os.path.normpath(token)
            while rel.startswith("../"):
                rel = rel[3:]
            patterns.add("*/" + rel)
        return any(fnmatch.fnmatchcase(p, pat) for p in protected for pat in patterns)

    deny = "Command would modify application source ({}). Source files are read-only."
    for i, tok in enumerate(tokens[:-1]):  # redirections: '>' '>>' '>|' '&>' followed by a target
        if ">" in tok and set(tok) <= set(">&|") and hits(tokens[i + 1]):
            return deny.format(tokens[i + 1])

    for seg in _segments(tokens):
        for s in _starts(seg):
            if s >= len(seg):
                continue
            verb, args = os.path.basename(seg[s]), seg[s + 1:]
            if s > 0:  # under find -exec the target is "{}", so judge by find's own patterns
                args = args + seg[:s]
            if verb in EDITORS:
                return f"'{verb}' is not allowed: application source must not be edited."
            if verb == "git":
                sub = next((a for a in args if not a.startswith("-") and a not in ("-C",)), None)
                if "-C" in args:  # skip the directory argument of -C
                    rest = args[args.index("-C") + 2:]
                    sub = next((a for a in rest if not a.startswith("-")), None)
                if sub is not None and sub not in GIT_READ_ONLY:
                    return f"'git {sub}' is not allowed: only read-only git commands may be used."
            if verb == "sed" and any(a.startswith("-i") or a.startswith("--in-place") for a in args):
                return "'sed -i' is not allowed: application source must not be edited."
            if verb == "perl" and any(re.fullmatch(r"-\w*i\S*", a) for a in args):
                return "'perl -i' is not allowed: application source must not be edited."
            if verb in WRITES_ALL_ARGS:
                for a in args:
                    if hits(a):
                        return deny.format(a)
            if verb in WRITES_LAST_ARG:
                targets = [a for a in args if not a.startswith("-")]
                if targets and hits(targets[-1]):
                    return deny.format(targets[-1])
            if verb == "dd":
                for a in args:
                    if a.startswith("of=") and hits(a[3:]):
                        return deny.format(a[3:])
            nested = _nested(verb, args)
            if nested and (reason := check_command(nested, protected, cwd)):
                return reason
    return None


def check_execution(command: str, policy: Policy, cwd: str) -> str | None:
    """Return a reason to deny a command that runs the application outside the recorded workflow."""
    try:
        tokens = _tokens(command)
    except ValueError:
        return "Could not parse the command; simplify the quoting."
    for seg in _segments(tokens):
        for s in _starts(seg):
            if s >= len(seg):
                continue
            verb, args = os.path.basename(seg[s]), seg[s + 1:]
            if verb in SHELLS and args and args[0].endswith(".sh"):  # bash path/to/script.sh ...
                verb, args = os.path.basename(args[0]), args[1:]
            if verb == "perf":
                sub = args[0] if args else "--help"
                if sub not in PERF_READ_ONLY:
                    return (f"'perf {sub}' runs the application outside the recorded workflow. Use "
                            "profile_target.sh for every run (it records perf stat, perf record and "
                            "metadata); use perf report/annotate/script on existing perf.data files.")
            elif verb in LAUNCHERS:
                return f"'{verb}' is not allowed directly; MPI runs go through profile_target.sh --ranks."
            elif verb in policy.binaries:
                if not args or not set(args) <= HELP_FLAGS:
                    return (f"Running {verb} directly is not allowed; run it through profile_target.sh "
                            f"(--cmd '{{binary}} ...'). Only '{verb} --help' / '-h' may be run directly.")
            elif verb == "profile_target.sh":
                if reason := _check_profile(args, policy, cwd):
                    return reason
            if (nested := _nested(verb, args)) and (reason := check_execution(nested, policy, cwd)):
                return reason
    return None


def _check_profile(args: list[str], policy: Policy, cwd: str) -> str | None:
    positional, runs, i = [], "runs", 0
    while i < len(args):
        if args[i] in ("--label", "--cmd", "--env", "--ranks", "--config", "--work", "--runs"):
            if args[i] == "--runs" and i + 1 < len(args):
                runs = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1
    run_id = positional[1] if len(positional) > 1 else ""
    runs = runs.replace("${PWD}", cwd).replace("$PWD", cwd)  # not expanded by the tokenizer
    runs_dir = Path(os.path.normpath(os.path.join(cwd, runs)))
    if runs_dir == policy.run_dir and run_id == "baseline":
        return None
    if runs_dir == policy.experiments:
        used = policy.experiments_used()
        if used >= policy.max_experiments:
            return (f"Experiment limit reached: {used} of {policy.max_experiments} experiment run(s) used. "
                    "Write findings.json from the evidence you have.")
        return None
    return (f"Artifact sets must go to {policy.run_dir} with run id 'baseline', or to "
            f"{policy.experiments} for experiments (got --runs {runs} and run id '{run_id}').")


def check_write(tool_input: dict, policy: Policy, cwd: str) -> str | None:
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    target = Path(os.path.normpath(os.path.join(cwd, path)))
    allowed = policy.run_dir / "findings.json"
    if target != allowed or target.is_symlink():
        return f"The file-writing tools may only write {allowed}."
    return None


def make_guard(policy: Policy):
    """Build the PreToolUse hook for one run."""

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        tool = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {}) or {}
        cwd = input_data.get("cwd") or str(policy.run_dir)
        reason = None
        if tool in EDIT_TOOLS:
            reason = check_write(tool_input, policy, cwd) if tool in ("Write", "Edit") else \
                f"{tool} is disabled; use Write or Edit on findings.json."
        elif tool == "Bash":
            command = tool_input.get("command", "")
            reason = check_command(command, tracked(policy.src), cwd) or check_execution(command, policy, cwd)
        if reason is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return guard
