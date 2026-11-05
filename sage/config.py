"""Per-target YAML config: where to get the app, how to build it, how to run it."""

from __future__ import annotations

import re
import shlex
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml

TARGETS = ("comd", "lulesh")
REQUIRED = ("target", "repo_url", "commit", "compiler", "flags", "build", "binary", "run")
OPTIONAL = ("env", "mpi")
MPI_REQUIRED = ("compiler", "build", "binary", "launcher", "ranks", "run")
MPI_OPTIONAL = ("env",)
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COMMIT = re.compile(r"[0-9a-f]{40}")
OPTIMIZED = re.compile(r"-O([1-3]|s|fast)")
REQUIRED_FLAGS = ("-g", "-fno-omit-frame-pointer")


class ConfigError(Exception):
    """Raised when a config file cannot be read or is invalid."""


@dataclass(frozen=True)
class MpiConfig:
    compiler: str  # e.g. mpicxx; replaces the top-level compiler for the MPI build
    build: list[str]
    binary: str
    launcher: str  # e.g. "mpirun -np {ranks}"
    ranks: int  # the rank count the default MPI run command is written for
    run: str
    env: dict[str, str] = field(default_factory=dict)  # added to (and overriding) the top-level env


@dataclass(frozen=True)
class TargetConfig:
    target: str
    repo_url: str
    commit: str
    compiler: str  # exported to the build commands as $SAGE_COMPILER
    flags: str  # exported to the build commands as $SAGE_FLAGS (shared by the MPI build)
    build: list[str]  # shell commands, run in order from .work/<target>/
    binary: str  # path relative to .work/<target>/
    run: str  # default run command; "{binary}" expands to the binary's absolute path
    env: dict[str, str] = field(default_factory=dict)
    mpi: MpiConfig | None = None

    def run_command(self, work: Path) -> str:
        return self.run.format(binary=work / self.binary)


def load_config(path: Path, target: str) -> TargetConfig:
    """Read and validate ``path``; every problem found is reported at once."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"config {path} is not valid YAML: {e}") from None

    errors = validate(data, target)
    if errors:
        raise ConfigError(f"invalid config {path}:\n" + "\n".join(f"  - {e}" for e in errors))
    env = lambda d: {k: str(v) for k, v in (d.get("env") or {}).items()}
    mpi = data.get("mpi")
    return TargetConfig(
        **{k: data[k] for k in REQUIRED}, env=env(data),
        mpi=MpiConfig(**{k: mpi[k] for k in MPI_REQUIRED}, env=env(mpi)) if mpi else None,
    )


def validate(data: object, target: str) -> list[str]:
    if not isinstance(data, dict):
        return ["top level must be a mapping of keys to values"]
    errors = keys(data, REQUIRED, OPTIONAL)

    if "target" in data:
        if data["target"] not in TARGETS:
            errors.append(f"target must be one of {', '.join(TARGETS)}, got {data['target']!r}")
        elif data["target"] != target:
            errors.append(f"config is for target '{data['target']}' but --target is '{target}'")

    if "repo_url" in data:
        url = data["repo_url"]
        parsed = urlparse(url) if isinstance(url, str) else None
        if not parsed or parsed.scheme != "https" or not parsed.netloc or parsed.path.strip("/") == "":
            errors.append(f"repo_url must be an https:// repository URL, got {url!r}")

    if "commit" in data:
        commit = data["commit"]
        if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
            errors.append(f"commit must be a full 40-character lowercase git SHA, got {commit!r}")

    if "flags" in data:
        flags = data["flags"]
        tokens = shlex.split(flags) if isinstance(flags, str) else []
        opt = [t for t in tokens if t.startswith("-O")]
        if not tokens:
            errors.append("flags must be a non-empty string of compiler flags")
        elif not opt or not OPTIMIZED.fullmatch(opt[-1]):
            errors.append(f"flags must enable optimization (-O1/-O2/-O3/-Os/-Ofast), got {flags!r}")
        missing = [f for f in REQUIRED_FLAGS if tokens and f not in tokens]
        if missing:
            errors.append(f"flags must include {' and '.join(missing)} so perf can resolve symbols and stacks")

    errors += build_fields(data, "")

    if data.get("mpi") is not None:
        mpi = data["mpi"]
        if not isinstance(mpi, dict):
            errors.append("mpi must be a mapping")
        else:
            errors += keys(mpi, MPI_REQUIRED, MPI_OPTIONAL, "mpi.")
            errors += build_fields(mpi, "mpi.")
            if "launcher" in mpi and placeholders(mpi["launcher"]) != {"ranks"}:
                errors.append("mpi.launcher must be a command string using the {ranks} placeholder "
                              "and no other {…} fields, e.g. 'mpirun -np {ranks}'")
            if "ranks" in mpi and (isinstance(mpi["ranks"], bool) or not isinstance(mpi["ranks"], int)
                                   or mpi["ranks"] < 1):
                errors.append(f"mpi.ranks must be a positive integer, got {mpi['ranks']!r}")
    return errors


def keys(data: dict, required: tuple, optional: tuple, prefix: str = "") -> list[str]:
    return ([f"missing required key '{prefix}{k}'" for k in required if k not in data]
            + [f"unknown key '{prefix}{k}'" for k in data if k not in required + optional])


def placeholders(text: object) -> set[str] | None:
    """The {…} field names used in ``text``; None if it is not a non-empty, well-formed string."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
    except ValueError:
        return None


def build_fields(data: dict, prefix: str) -> list[str]:
    """Checks shared by the top level and the mpi section: compiler, build, binary, run, env."""
    errors = []
    if "compiler" in data:
        compiler = data["compiler"]
        if not isinstance(compiler, str) or not compiler or any(c.isspace() for c in compiler):
            errors.append(f"{prefix}compiler must be a single command name or path, got {compiler!r}")

    if "build" in data:
        build = data["build"]
        if not isinstance(build, list) or not build:
            errors.append(f"{prefix}build must be a non-empty list of shell commands")
        elif not all(isinstance(c, str) and c.strip() for c in build):
            errors.append(f"every {prefix}build command must be a non-empty string")
        else:
            for var in ("SAGE_COMPILER", "SAGE_FLAGS"):
                if not any(re.search(rf"\$\{{?{var}\b", c) for c in build):
                    errors.append(f"{prefix}build commands must use ${var}, or the recorded build info would be wrong")

    if "binary" in data:
        binary = data["binary"]
        if not isinstance(binary, str) or not binary.strip():
            errors.append(f"{prefix}binary must be a non-empty path")
        elif PurePosixPath(binary).is_absolute() or ".." in PurePosixPath(binary).parts:
            errors.append(f"{prefix}binary must be a relative path inside .work/<target>/, got {binary!r}")

    if "run" in data:
        run = data["run"]
        if not isinstance(run, str) or not run.strip():
            errors.append(f"{prefix}run must be a non-empty command string")
        elif placeholders(run) is None:
            errors.append(f"{prefix}run has malformed braces")
        elif placeholders(run) != {"binary"}:
            errors.append(f"{prefix}run must use the {{binary}} placeholder and no other {{…}} fields")

    if data.get("env") is not None:
        env = data["env"]
        if not isinstance(env, dict):
            errors.append(f"{prefix}env must be a mapping of variable names to values")
        else:
            for k, v in env.items():
                if not isinstance(k, str) or not ENV_NAME.fullmatch(k):
                    errors.append(f"{prefix}env name {k!r} is not a valid variable name")
                elif k.startswith("SAGE_"):
                    errors.append(f"{prefix}env name {k!r} uses the reserved SAGE_ prefix")
                if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                    errors.append(f"{prefix}env value for {k!r} must be a string or number")
    return errors


def shell_exports(cfg: TargetConfig, mpi: bool = False) -> str:
    """Bash assignments for the scripts in scripts/ (``eval``-ed there).

    With ``mpi`` the MPI variant's compiler, build, binary, run command and env are used.
    """
    if mpi and cfg.mpi is None:
        raise ConfigError(f"config for {cfg.target} has no mpi section")
    v = cfg.mpi if mpi else cfg
    values = {
        "target": cfg.target, "repo_url": cfg.repo_url, "commit": cfg.commit, "flags": cfg.flags,
        "variant": "mpi" if mpi else "serial", "compiler": v.compiler, "binary": v.binary, "run": v.run,
        "launcher": cfg.mpi.launcher if mpi else "", "ranks": str(cfg.mpi.ranks) if mpi else "",
    }
    q = shlex.quote
    lines = [f"SAGE_{k.upper()}={q(val)}" for k, val in values.items()]
    lines.append("SAGE_BUILD=(" + " ".join(q(c) for c in v.build) + ")")
    env = {**cfg.env, **(cfg.mpi.env if mpi else {})}
    lines.append("SAGE_ENV_NAMES=(" + " ".join(q(k) for k in env) + ")")
    lines += [f"export {k}={q(val)}" for k, val in env.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m sage.config CONFIG TARGET [--mpi]``: validate, then print shell assignments."""
    args = sys.argv[1:] if argv is None else argv
    mpi = "--mpi" in args
    args = [a for a in args if a != "--mpi"]
    if len(args) != 2:
        print("usage: python -m sage.config CONFIG TARGET [--mpi]", file=sys.stderr)
        return 2
    try:
        print(shell_exports(load_config(Path(args[0]), args[1]), mpi=mpi))
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
