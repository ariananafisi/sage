Identify the performance bottlenecks of the {{target}} application ({{variant}} build) with perf.

Run directory (your working directory): {{run_dir}}
Source checkout (read-only): {{src}}
  upstream {{repo_url}} @ {{commit}}
Build record: {{build_info}}
Default run command: {{run_command}}
Environment already set for your commands: {{environment}}

Preflight results (scripts/preflight.sh, {{preflight_path}}):
{{preflight}}

Commands:
  build:              {{build_command}}
  baseline:           {{baseline_command}}
  annotate:           {{annotate_command}}
  experiment:         {{experiment_command}}
  annotate experiment: {{annotate_experiment_command}}
  validate findings:  {{validate_command}}

In the experiment command, `{binary}` is replaced by the built binary, and <name> names the
experiment's directory under experiments/. Use a short name that describes the change, e.g.
size-60. {{mpi_note}}

findings.json schema: {{schema}}
Limits: {{max_experiments}} experiment run(s), {{max_turns}} turns, {{timeout_minutes}} minutes.
{{goal}}
