"""Safety rails for the agent: perf is the only profiler, source code is read-only.

Two layers:
  * ``make_guard`` builds a PreToolUse hook that denies shell commands which
    would edit existing source files or invoke a profiler other than perf.
  * ``snapshot`` / ``changed`` hash source files before the run and verify them
    after, which catches anything the command check cannot see (e.g. a write
    from inside ``python -c``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any

SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".C", ".h", ".hh", ".hpp", ".hxx",
    ".f", ".F", ".f90", ".F90", ".cu", ".cmake",
}

OTHER_PROFILERS = re.compile(
    r"\b(gprof|valgrind|callgrind|cachegrind|vtune|amplxe-cl|advixe-cl|"
    r"likwid-[\w-]+|hpcrun|hpcstruct|hpcprof|nsys|ncu|nvprof|tau_exec|tau_run|"
    r"scorep|scalasca|uftrace|bpftrace|py-spy|pprof|operf|ocount|opcontrol|"
    r"strace|ltrace)\b"
    r"|(?<![\w-])-pg\b"
)

EDITORS = {"vi", "vim", "nvim", "nano", "emacs", "ed", "ex", "patch"}
GIT_WRITES = {
    "checkout", "restore", "reset", "apply", "am", "stash", "clean", "rm", "mv",
    "switch", "pull", "merge", "rebase", "cherry-pick", "revert", "commit",
}
WRITES_ALL_ARGS = {"rm", "mv", "tee", "touch", "truncate", "shred", "unlink"}
WRITES_LAST_ARG = {"cp", "install", "ln", "rsync"}
SHELLS = {"sh", "bash", "zsh", "dash"}
WRAPPERS = {"sudo", "env", "nice", "nohup", "time", "command", "exec", "xargs", "timeout"}
SEPARATORS = {";", "&&", "||", "|", "|&", "&", "(", ")", "\n"}


def is_source_file(path: Path) -> bool:
    name = path.name
    return (
        path.suffix in SOURCE_SUFFIXES
        or name in {"makefile", "GNUmakefile", "CMakeLists.txt"}
        or name.startswith("Makefile")
    )


def snapshot(root: Path) -> dict[str, str]:
    """Map absolute path -> sha256 for every source file under ``root``."""
    digests = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            path = Path(dirpath, name)
            if is_source_file(path) and path.is_file():
                digests[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def changed(before: dict[str, str]) -> list[str]:
    """Paths from ``before`` that were modified or removed."""
    out = []
    for path, digest in before.items():
        p = Path(path)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            out.append(path)
    return sorted(out)


def check_command(command: str, protected: set[str], cwd: str = "/") -> str | None:
    """Return a reason to deny ``command``, or None if it is allowed."""
    m = OTHER_PROFILERS.search(command)
    if m:
        return f"'{m.group(0)}' is not allowed: perf is the only permitted profiler."
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
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

    # Redirections: '>' '>>' '>|' '&>' followed by a target.
    for i, tok in enumerate(tokens[:-1]):
        if ">" in tok and set(tok) <= set(">&|") and hits(tokens[i + 1]):
            return deny.format(tokens[i + 1])

    segments, current = [], []
    for tok in tokens:
        if tok in SEPARATORS:
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)

    for seg in segments:
        # A command can start the segment (after wrappers / VAR=val) or follow find's -exec.
        starts, i = [], 0
        while i < len(seg) and (
            seg[i] in WRAPPERS or "=" in seg[i]
            or (i > 0 and seg[i - 1] in WRAPPERS and (seg[i].startswith("-") or seg[i][:1].isdigit()))
        ):
            i += 1
        starts.append(i)
        starts += [j + 1 for j, t in enumerate(seg) if t in {"-exec", "-execdir", "-ok"}]

        for s in starts:
            if s >= len(seg):
                continue
            verb, args = os.path.basename(seg[s]), seg[s + 1:]
            if s > 0:  # under find -exec the target is "{}", so judge by find's own patterns
                args = args + seg[:s]
            if verb in EDITORS:
                return f"'{verb}' is not allowed: application source must not be edited."
            if verb == "git" and any(a in GIT_WRITES for a in args):
                return "git commands that change the working tree are not allowed."
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
            if verb in SHELLS and "-c" in args:
                idx = args.index("-c")
                if idx + 1 < len(args):
                    reason = check_command(args[idx + 1], protected, cwd)
                    if reason:
                        return reason
            if verb == "eval" and args:
                reason = check_command(" ".join(args), protected, cwd)
                if reason:
                    return reason
    return None


def make_guard(protected: set[str]):
    """Build a PreToolUse hook that enforces the rules above."""

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        tool = input_data.get("tool_name")
        reason = None
        if tool in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            reason = "File-editing tools are disabled for this agent."
        elif tool == "Bash":
            command = input_data.get("tool_input", {}).get("command", "")
            reason = check_command(command, protected, input_data.get("cwd") or "/")
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
