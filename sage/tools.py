"""The agent's four tools, as OpenAI function-calling schemas and their implementations.

bash          run a shell command in the run directory (checked by sage.guard first)
read_file     read a text file with line numbers
search        regex search over files, like grep -rn
write_findings  write the run's findings.json (the only file-writing tool)
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .guard import HIDDEN_REASON, Policy, check_bash

MAX_OUTPUT = 20_000  # characters of tool output returned to the model
MAX_COMMAND_SECONDS = 1800  # a single shell command (builds and profiles take a few minutes)
MAX_READ_LINES = 2000
MAX_LINE = 2000
MAX_SEARCH_RESULTS = 200
MAX_SEARCH_FILE = 5_000_000

SCHEMAS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command in the run directory and return its exit code and combined "
                       "stdout/stderr (long output is truncated in the middle). Each call starts in the run "
                       "directory; a `cd` does not carry over. Commands that edit the source checkout, use "
                       "a profiler other than perf, or run the application outside profile_target.sh are refused.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["command"],
                       "properties": {"command": {"type": "string", "description": "The command to run."}}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return its lines numbered from 1. Relative paths are relative to "
                       "the run directory.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["path"],
                       "properties": {
                           "path": {"type": "string"},
                           "offset": {"type": "integer", "minimum": 1, "description": "First line to return (default 1)."},
                           "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LINES,
                                     "description": f"Number of lines to return (default 400, at most {MAX_READ_LINES})."}}}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search files for a regular expression (Python syntax) and return matching lines as "
                       "path:line: text. Searches a file, or a directory recursively (skipping .git and binary files).",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["pattern", "path"],
                       "properties": {
                           "pattern": {"type": "string"},
                           "path": {"type": "string", "description": "File or directory to search."},
                           "glob": {"type": "string", "description": "Only files whose name matches, e.g. *.cc"},
                           "ignore_case": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "write_findings",
        "description": "Write findings.json in the run directory, replacing any previous version. This is the "
                       "only way to write files besides shell redirection. Run the validate command afterwards.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["content"],
                       "properties": {"content": {"type": "string", "description": "The complete JSON document."}}}}},
]


class DeadlineExceeded(Exception):
    """The wall-clock limit ran out."""


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 5, limit * 3 // 5
    return f"{text[:head]}\n\n[... {len(text) - head - tail} characters omitted ...]\n\n{text[-tail:]}"


@dataclass
class Tools:
    policy: Policy
    env: dict[str, str]
    deadline: float  # time.monotonic() value at which the run must stop
    calls: int = field(default=0)

    @property
    def run_dir(self) -> Path:
        return self.policy.run_dir

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def call(self, name: str, arguments: str) -> str:
        """Run one tool call; problems are returned as text so the model can react to them."""
        self.calls += 1
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return f"ERROR: the arguments are not valid JSON ({e})."
        if not isinstance(args, dict):
            return "ERROR: the arguments must be a JSON object."
        handler = {"bash": self.bash, "read_file": self.read_file, "search": self.search,
                   "write_findings": self.write_findings}.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'. Available: bash, read_file, search, write_findings."
        try:
            return handler(**args)
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"

    def _path(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        return p if p.is_absolute() else self.run_dir / p

    def bash(self, command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            return "ERROR: command must be a non-empty string."
        if reason := check_bash(command, self.policy, str(self.run_dir)):
            return f"DENIED: {reason}"
        timeout = min(MAX_COMMAND_SECONDS, self.remaining())
        if timeout <= 0:
            raise DeadlineExceeded()
        proc = subprocess.Popen(["bash", "-c", command], cwd=self.run_dir, env={**os.environ, **self.env},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                start_new_session=True)  # own process group, so a timeout kills perf and the app too
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
            if self.remaining() <= 0:
                raise DeadlineExceeded() from None
            return truncate(f"TIMED OUT after {timeout:.0f} s and was killed. Output so far:\n{output}")
        return truncate(f"exit code {proc.returncode}\n{output}")

    def read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        p = self._path(path)
        if self.policy.is_hidden(p):
            return f"ERROR: {HIDDEN_REASON}"
        if not p.is_file():
            return f"ERROR: {p} is not a file."
        offset, limit = max(1, int(offset)), min(max(1, int(limit)), MAX_READ_LINES)
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            return f"ERROR: {p} looks like a binary file; use bash (e.g. perf report) to inspect it."
        lines = data.decode(errors="replace").splitlines()
        chosen = lines[offset - 1:offset - 1 + limit]
        body = "\n".join(f"{i:6d}\t{l[:MAX_LINE]}" for i, l in enumerate(chosen, offset))
        more = f"\n[{len(lines) - (offset - 1 + len(chosen))} more lines]" if offset - 1 + len(chosen) < len(lines) else ""
        return truncate(f"{p} ({len(lines)} lines)\n{body}{more}")

    def search(self, pattern: str, path: str, glob: str | None = None, ignore_case: bool = False) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return f"ERROR: invalid regular expression: {e}"
        root = self._path(path)
        if self.policy.is_hidden(root):
            return f"ERROR: {HIDDEN_REASON}"
        if not root.exists():
            return f"ERROR: {root} does not exist."
        files = [root]
        if root.is_dir():
            files = []
            for d, dirs, names in os.walk(root):
                dirs[:] = sorted(x for x in dirs if x != ".git" and not self.policy.is_hidden(Path(d, x)))
                files += [Path(d, n) for n in sorted(names) if not self.policy.is_hidden(Path(d, n))]
        results = []
        for f in files:
            if glob and not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                if f.stat().st_size > MAX_SEARCH_FILE:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            for n, line in enumerate(data.decode(errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append(f"{f}:{n}: {line.strip()[:300]}")
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return truncate("\n".join(results) + f"\n[stopped at {MAX_SEARCH_RESULTS} matches]")
        return truncate("\n".join(results)) if results else "no matches"

    def write_findings(self, content: str) -> str:
        if not isinstance(content, str):
            return "ERROR: content must be a string holding the JSON document."
        target = self.run_dir / "findings.json"
        if target.is_symlink():
            return "ERROR: findings.json is a symlink; refusing to write through it."
        target.write_text(content)
        return f"saved {target} ({len(content)} characters). Now run the validate command."
