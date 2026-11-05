from pathlib import Path

import pytest
import yaml

import subprocess

from sage.config import ConfigError, TargetConfig, load_config, main, shell_exports

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

VALID = {
    "target": "lulesh",
    "repo_url": "https://github.com/LLNL/LULESH.git",
    "commit": "3e01c40b3281aadb7f996525cdd4a3354f6d3801",
    "compiler": "g++",
    "flags": "-O3 -g -fno-omit-frame-pointer",
    "build": ['cmake -S src -B build -DCMAKE_CXX_COMPILER="$SAGE_COMPILER" -DCMAKE_CXX_FLAGS="$SAGE_FLAGS"',
              "cmake --build build"],
    "binary": "build/lulesh2.0",
    "run": "{binary} -s 10",
    "env": {"OMP_NUM_THREADS": 4, "OMP_PROC_BIND": "true"},
}


def write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
    return path


def errors_for(tmp_path, data, target="lulesh"):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, data), target)
    return str(exc.value)


@pytest.mark.parametrize("target,url,binary", [
    ("lulesh", "https://github.com/LLNL/LULESH.git", "build/lulesh2.0"),
    ("comd", "https://github.com/ECP-copa/CoMD.git", "src/bin/CoMD-serial"),
])
def test_shipped_configs_are_valid(target, url, binary):
    cfg = load_config(CONFIGS / f"{target}.yaml", target)
    assert cfg.target == target
    assert cfg.repo_url == url
    assert len(cfg.commit) == 40
    assert cfg.binary == binary
    assert cfg.build and all(isinstance(c, str) for c in cfg.build)
    assert cfg.run_command(Path("/w")).startswith(f"/w/{binary} ")
    assert all(isinstance(v, str) for v in cfg.env.values())
    assert cfg.flags.split()[0].startswith("-O") and "-g" in cfg.flags and "-fno-omit-frame-pointer" in cfg.flags


def test_valid_config_loads_and_normalizes_env(tmp_path):
    cfg = load_config(write(tmp_path, VALID), "lulesh")
    assert cfg == TargetConfig(**{**VALID, "env": {"OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "true"}})
    assert cfg.run_command(Path("/work/lulesh")) == "/work/lulesh/build/lulesh2.0 -s 10"


@pytest.mark.parametrize("env", [None, {}])
def test_env_is_optional(tmp_path, env):
    data = {k: v for k, v in VALID.items() if k != "env"}
    if env is not None:
        data["env"] = env
    assert load_config(write(tmp_path, data), "lulesh").env == {}


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml", "lulesh")


def test_invalid_yaml(tmp_path):
    assert "not valid YAML" in errors_for(tmp_path, "target: [lulesh\n")


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
def test_top_level_must_be_mapping(tmp_path, text):
    assert "top level must be a mapping" in errors_for(tmp_path, text)


def test_all_problems_reported_together(tmp_path):
    msg = errors_for(tmp_path, {"target": "lulesh", "colour": "blue"})
    for key in ("repo_url", "commit", "compiler", "flags", "build", "binary", "run"):
        assert f"missing required key '{key}'" in msg
    assert "unknown key 'colour'" in msg
    assert msg.startswith("invalid config ")


def test_target_must_match_cli(tmp_path):
    assert "config is for target 'lulesh' but --target is 'comd'" in errors_for(tmp_path, VALID, target="comd")


@pytest.mark.parametrize("key,value,message", [
    ("target", "hpcg", "target must be one of comd, lulesh"),
    ("repo_url", "git@github.com:LLNL/LULESH.git", "repo_url must be an https://"),
    ("repo_url", "http://github.com/LLNL/LULESH.git", "repo_url must be an https://"),
    ("repo_url", "https://github.com", "repo_url must be an https://"),
    ("repo_url", 42, "repo_url must be an https://"),
    ("commit", "master", "commit must be a full 40-character"),
    ("commit", "3e01c40", "commit must be a full 40-character"),
    ("commit", "3E01C40B3281AADB7F996525CDD4A3354F6D3801", "commit must be a full 40-character"),
    ("compiler", "", "compiler must be a single command name or path"),
    ("compiler", "gcc -m64", "compiler must be a single command name or path"),
    ("flags", "", "flags must be a non-empty string"),
    ("flags", "-g -fno-omit-frame-pointer", "flags must enable optimization"),
    ("flags", "-O0 -g -fno-omit-frame-pointer", "flags must enable optimization"),
    ("flags", "-O3 -g -fno-omit-frame-pointer -O0", "flags must enable optimization"),
    ("flags", "-O3 -fno-omit-frame-pointer", "flags must include -g so perf"),
    ("flags", "-O2 -g", "flags must include -fno-omit-frame-pointer so perf"),
    ("build", ["make CC=$SAGE_COMPILER"], "build commands must use $SAGE_FLAGS"),
    ("build", ['make OPTFLAGS="${SAGE_FLAGS}"'], "build commands must use $SAGE_COMPILER"),
    ("env", {"SAGE_FLAGS": "-O0"}, "reserved SAGE_ prefix"),
    ("build", [], "build must be a non-empty list"),
    ("build", "make", "build must be a non-empty list"),
    ("build", ["make", ""], "every build command must be a non-empty string"),
    ("build", ["make", 3], "every build command must be a non-empty string"),
    ("binary", "", "binary must be a non-empty path"),
    ("binary", "/usr/bin/lulesh", "binary must be a relative path"),
    ("binary", "../escape/lulesh", "binary must be a relative path"),
    ("run", "", "run must be a non-empty command"),
    ("run", "./lulesh2.0 -s 10", "run must use the {binary} placeholder"),
    ("run", "{binary} -s {size}", "run must use the {binary} placeholder"),
    ("run", "{binary -s 10", "run has malformed braces"),
    ("env", ["OMP_NUM_THREADS=4"], "env must be a mapping"),
    ("env", {"1BAD": "x"}, "is not a valid variable name"),
    ("env", {"DEBUG": True}, "must be a string or number"),
    ("env", {"PATHS": ["a", "b"]}, "must be a string or number"),
])
def test_invalid_field(tmp_path, key, value, message):
    assert message in errors_for(tmp_path, {**VALID, key: value})


@pytest.mark.parametrize("flags", ["-O2 -g -fno-omit-frame-pointer", "-Ofast -march=native -g -fno-omit-frame-pointer",
                                   "-Os -g -fno-omit-frame-pointer"])
def test_accepted_optimization_levels(tmp_path, flags):
    assert load_config(write(tmp_path, {**VALID, "flags": flags}), "lulesh").flags == flags


def test_shell_exports_round_trip_through_bash(tmp_path):
    cfg = load_config(write(tmp_path, {**VALID, "env": {"NOTE": "it's \"quoted\" $x"}}), "lulesh")
    script = shell_exports(cfg) + '''
printf '%s\\n' "$SAGE_TARGET" "$SAGE_COMMIT" "$SAGE_COMPILER" "$SAGE_FLAGS" "$SAGE_BINARY" "${#SAGE_BUILD[@]}" "${SAGE_BUILD[0]}" "$NOTE"
'''
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == ["lulesh", VALID["commit"], "g++", VALID["flags"], "build/lulesh2.0", "2", VALID["build"][0],
                   "it's \"quoted\" $x"]


def test_module_entry_point(tmp_path, capsys):
    assert main([str(write(tmp_path, VALID)), "lulesh"]) == 0
    assert "SAGE_COMMIT=" + VALID["commit"] in capsys.readouterr().out
    assert main([str(write(tmp_path, VALID)), "comd"]) == 2
    assert "but --target is 'comd'" in capsys.readouterr().err
    assert main([]) == 2


MPI = {
    "compiler": "mpicxx",
    "build": ['cmake -S src -B build-mpi -DCMAKE_CXX_COMPILER="$SAGE_COMPILER" -DCMAKE_CXX_FLAGS="$SAGE_FLAGS"'],
    "binary": "build-mpi/lulesh2.0", "launcher": "mpirun -np {ranks}", "ranks": 8,
    "run": "{binary} -s 20", "env": {"OMP_NUM_THREADS": 1},
}


@pytest.mark.parametrize("target", ["lulesh", "comd"])
def test_shipped_configs_have_mpi_variants(target):
    mpi = load_config(CONFIGS / f"{target}.yaml", target).mpi
    assert mpi.compiler.startswith("mpi") and "{ranks}" in mpi.launcher and mpi.ranks >= 1
    assert mpi.binary != load_config(CONFIGS / f"{target}.yaml", target).binary


def test_mpi_section_loads(tmp_path):
    cfg = load_config(write(tmp_path, {**VALID, "mpi": MPI}), "lulesh")
    assert cfg.mpi.ranks == 8 and cfg.mpi.env == {"OMP_NUM_THREADS": "1"} and cfg.mpi.launcher == "mpirun -np {ranks}"
    assert load_config(write(tmp_path, VALID), "lulesh").mpi is None


@pytest.mark.parametrize("change,message", [
    ({"ranks": 0}, "mpi.ranks must be a positive integer"),
    ({"ranks": True}, "mpi.ranks must be a positive integer"),
    ({"ranks": "8"}, "mpi.ranks must be a positive integer"),
    ({"launcher": "mpirun -np 8"}, "mpi.launcher must be a command string using the {ranks} placeholder"),
    ({"launcher": "mpirun -np {n}"}, "mpi.launcher must be a command string using the {ranks} placeholder"),
    ({"binary": "/abs/lulesh"}, "mpi.binary must be a relative path"),
    ({"compiler": "mpicxx -O2"}, "mpi.compiler must be a single command name"),
    ({"build": ["make"]}, "mpi.build commands must use $SAGE_COMPILER"),
    ({"run": "./lulesh"}, "mpi.run must use the {binary} placeholder"),
    ({"env": {"SAGE_X": 1}}, "mpi.env name 'SAGE_X' uses the reserved SAGE_ prefix"),
    ({"extra": 1}, "unknown key 'mpi.extra'"),
])
def test_invalid_mpi_section(tmp_path, change, message):
    assert message in errors_for(tmp_path, {**VALID, "mpi": {**MPI, **change}})


def test_mpi_section_missing_keys_and_type(tmp_path):
    msg = errors_for(tmp_path, {**VALID, "mpi": {"compiler": "mpicc"}})
    for key in ("build", "binary", "launcher", "ranks", "run"):
        assert f"missing required key 'mpi.{key}'" in msg
    assert "mpi must be a mapping" in errors_for(tmp_path, {**VALID, "mpi": ["mpirun"]})


def test_shell_exports_for_mpi_variant(tmp_path):
    cfg = load_config(write(tmp_path, {**VALID, "mpi": MPI}), "lulesh")
    script = shell_exports(cfg, mpi=True) + '\nprintf "%s|" "$SAGE_VARIANT" "$SAGE_COMPILER" "$SAGE_BINARY" ' \
        '"$SAGE_LAUNCHER" "$SAGE_RANKS" "$SAGE_RUN" "$OMP_NUM_THREADS" "$SAGE_FLAGS"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert out == f"mpi|mpicxx|build-mpi/lulesh2.0|mpirun -np {{ranks}}|8|{{binary}} -s 20|1|{VALID['flags']}|"
    serial = subprocess.run(["bash", "-c", shell_exports(cfg) + '\necho "$SAGE_VARIANT $OMP_NUM_THREADS"'],
                            capture_output=True, text=True, check=True).stdout
    assert serial.strip() == "serial 4"
    with pytest.raises(ConfigError, match="no mpi section"):
        shell_exports(load_config(write(tmp_path, VALID), "lulesh"), mpi=True)
    assert main([str(write(tmp_path, VALID)), "lulesh", "--mpi"]) == 2
