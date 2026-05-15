# Verification Checks

Roadmap revision: `orchestrator/roadmaps/2026-05-16-00-sgnav-isaac-benchmark-completion/rev-001/`.

This file records repo- and roadmap-specific checks. Universal reviewer duties,
lineage requirements, evidence requirements, and approve/reject format live in
`orchestrator/roles/reviewer.md`.
Repo-wide invariants live in `orchestrator/project-contract.md`; reference them
here only when this roadmap needs a specific check or override.

## Baseline Checks

- Command: `python -m pytest isaac_bench/tests`
  Why: protects existing unit and integration coverage for mapping, A*, episode generation, object memory, scene graph, frontier scoring, metrics, and smoke-compatible SG-Nav flows.
- Command: `git diff --check`
  Why: catches whitespace and patch hygiene issues before review.
- Command: `python -m json.tool orchestrator/state.json >/tmp/sgnav_orchestrator_state.json`
  Why: ensures controller state remains parseable by the orchestrator loop.
- Command: `python -m isaac_bench.scripts.run_one_episode --config isaac_bench/configs/isaac_bench.yaml --episode-file data/interioragent_episodes/debug.jsonl --episode-index 0 --planner astar --detector dry_run --sim-backend map --output /tmp/sgnav_isaac_dryrun_results.jsonl --debug-map /tmp/sgnav_isaac_dryrun.png`
  Why: required non-Isaac dry-run smoke for every runtime-affecting round; if the local episode fixture is missing, reviewers may record a skip only for non-runtime scaffold/docs rounds, otherwise the round must create or document the fixture first.
- Command: `./scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.run_one_episode --config isaac_bench/configs/isaac_bench.yaml --episode-file data/interioragent_episodes/debug.jsonl --episode-index 0 --detector yolo_world --segmenter sam2 --sim-backend isaac --headless true --output /tmp/sgnav_isaac_smoke_results.jsonl --debug-map /tmp/sgnav_isaac_smoke.png`
  Why: required Isaac/YOLO-World/SAM2 smoke when Isaac, InteriorAgent preprocessing, episode fixtures, and model files are available; otherwise the reviewer must record the exact missing import, executable, dataset, or model path and confirm the run failed or skipped clearly.

## Alignment Checks

- Criterion: real benchmark perception uses YOLO-World plus SAM2.
  Check: metric-capable runs do not silently select `dry_run`, seeded GT object memory, GT goal fallback, missing-model stubs, or detector IPC degradation unless a reviewed explicit non-metric/debug flag labels the output as non-metric.
- Criterion: result rows follow the metric validity contract.
  Check: every row includes the fields required by `docs/benchmark_metric_validity_contract.md`; dry-run, seeded GT memory, unmarked deterministic local LLM scoring, and static-map SG-Nav planning are marked `metric_valid=false`.
- Criterion: SG-Nav decision dumps are inspectable.
  Check: `python -m isaac_bench.scripts.dump_sgnav_step ...` writes a JSON artifact with the keys required by `docs/sgnav_decision_dump_contract.md`.
- Criterion: SG-Nav paper mechanism is preserved except for the Isaac/InteriorAgent runtime substrate.
  Check: touched navigation or policy code still supports RGB-D observation, online occupancy/free-space mapping, online 3D scene graph, object/group/room hierarchy where feasible, subgraph text or equivalent representation, hierarchical LLM scoring, subgraph probability to frontier interpolation, frontier selection, graph-based re-perception, deterministic local planning, and SG-Nav STOP confirmation.
- Criterion: benchmark metrics are complete and reproducible.
  Check: result rows and summaries cover SR, SPL, SoftSPL, distance-to-goal, path length, steps, collisions or stuck events, timeout, perception latency, graph latency, and planning latency when the touched path can affect reporting.
- Criterion: existing runnable paths are not degraded.
  Check: existing README smoke commands and A* map commands still parse and run, or any required fixture/model skip is explicit and documented.
- Criterion: external assets fail clearly.
  Check: missing Isaac, InteriorAgent scenes, YOLO-World weights, SAM2 checkpoints/configs, or LLM endpoints produce actionable errors or documented skips rather than quiet behavior changes.

## Task-Specific Checks

- Add focused tests for the touched component before reviewer approval.
- For CLI/config changes, include at least one parser/config regression test or command-level smoke.
- For metric or fallback changes, include a test that proves metric runs reject or label debug-only fallbacks.
- For SG-Nav parity changes, include tests around the specific mechanism touched, such as frontier interpolation, subgraph scoring payloads, candidate credibility accumulation, or STOP verification.

## Manual Checks

- For Isaac-only behavior that cannot run on the reviewer host, record the Isaac availability probe, the exact skipped command, and the missing dependency path or import.
- For visual/debug panel changes, save a representative panel under an ignored `debug/` path and record what it shows.
- For README or setup-script changes, run or dry-run the exact documented command when local external assets permit it.

## Roadmap Overrides

- This roadmap is safety-first and serial by default. Parallel worker fan-out is allowed only when the planner writes `worker-plan.json` with disjoint file ownership and the selected extraction does not touch shared benchmark-mode semantics, result schemas, or orchestrator state.
