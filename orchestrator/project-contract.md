# Project Contract

This file records repo-wide invariants shared by every roadmap family and
round. Keep roadmap revisions focused on current coordination; point here for
stable contracts instead of restating them in every role or roadmap file.

## Stable Interfaces

- Event schemas: episode JSONL rows from `isaac_bench/scripts/run_one_episode.py` and `isaac_bench/scripts/run_benchmark.py`; metric summaries from `isaac_bench/metrics/summarize.py`; review every change that adds, removes, or renames result fields.
- Golden logs and fixtures: local generated fixtures live under ignored `data/`; the canonical lightweight smoke fixture path is `data/interioragent_episodes/debug.jsonl` when available, with preprocessing artifacts under `data/interioragent_preprocessed/`.
- Dry-run or command-rendering output: `--detector dry_run`, `--seed-gt-object-memory`, `--allow-gt-goal-fallback`, distance-only/map-only planner behavior, and missing-model stubs are debug/smoke surfaces only and must not be counted as benchmark metrics unless an explicit non-metric/debug run flag marks the output.
- Package and module boundaries: keep reusable behavior in `isaac_bench/` modules; keep CLI orchestration in `isaac_bench/scripts/`; keep environment entrypoints in top-level `scripts/`; keep generated benchmark data and debug media out of version control.
- Public compatibility facades: preserve `./run_isaac_bench.sh`, `scripts/setup_sgnav_isaac_env.sh`, `scripts/activate_sgnav_isaac_env.sh`, `scripts/run_sgnav_isaac_env.sh`, `isaac_bench.scripts.preprocess_interioragent`, `generate_episodes`, `run_one_episode`, `run_benchmark`, and `isaac_bench.metrics.summarize` unless a roadmap revision explicitly authorizes a migration with compatibility shims.

## Alignment Invariants

- Human-approved architecture constraints: the real benchmark path must use RGB-D Isaac/InteriorAgent observations, online occupancy/free-space mapping, YOLO-World plus SAM2 perception, online 3D object memory, SG-Nav-compatible scene graph/subgraph reasoning, LLM or LLM-compatible hierarchical scoring, subgraph-probability-to-frontier interpolation, graph-based re-perception, deterministic A*/FMM-equivalent planning, and STOP only after SG-Nav goal confirmation.
- Compatibility promises: existing smoke tests, map/A* benchmark commands, setup scripts, and README-documented dry-run workflows must keep working while the Isaac/SG-Nav stack is completed.
- Explicit non-goals that should not be reopened without a new roadmap family: do not replace YOLO-World/SAM2 with GT perception for reported metrics; do not silently fall back from model-backed perception to dry-run/GT during metric runs; do not count debug-only fallbacks as benchmark results; do not rewrite the project into an unrelated navigation architecture.

## Verification Anchors

- Invariants every reviewer should consider when touched: benchmark/dry-run mode separation, result schema stability, missing external asset error clarity, SG-Nav paper-mechanism parity, and preservation of map/A* smoke behavior.
- Baseline commands that protect shared contracts: `python -m pytest isaac_bench/tests`; dry-run smoke with `python -m isaac_bench.scripts.run_one_episode --config isaac_bench/configs/isaac_bench.yaml --episode-file data/interioragent_episodes/debug.jsonl --episode-index 0 --planner astar --detector dry_run --sim-backend map --output /tmp/sgnav_isaac_dryrun_results.jsonl --debug-map /tmp/sgnav_isaac_dryrun.png` when the local fixture exists; Isaac smoke through `./scripts/run_sgnav_isaac_env.sh` when Isaac, InteriorAgent, YOLO-World, and SAM2 assets are present.

## Update Rule

Update this file only when the repo-wide invariant itself changes. When a
roadmap temporarily narrows or extends an invariant, record the override in the
active roadmap bundle and keep the durable rule here.
