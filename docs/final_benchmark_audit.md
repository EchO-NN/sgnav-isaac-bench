# Final Benchmark Audit

Date: 2026-05-16

Overall status: partially complete. The repository now has the strict SG-Nav
Isaac/InteriorAgent benchmark stack implemented and guarded by contracts,
schema tests, asset checks, synthetic/unit coverage, dry-run smoke coverage,
and a short Isaac integration smoke with YOLO-World plus SAM2. In this local
environment the strict Isaac row is intentionally `metric_valid=false` because
it used the deterministic local LLM-compatible scorer, which is a benchmark
fallback unless a named ablation is declared or a real OpenAI-compatible LLM is
enabled and reachable.

## Command Evidence

- `pytest -q`
  - Direct base-shell result: skipped as an environment mismatch because
    `pytest` is not installed in the base shell.
  - Repository environment result: `./run_isaac_bench.sh -m pytest -q`
    passed with `127 passed`.
- `./run_isaac_bench.sh -m isaac_bench.scripts.check_assets --require-yolo-world --require-sam2 --require-rose2-source --require-interioragent --require-isaac`
  must pass before a strict metric run. It reports `MISSING rose2_source_root`
  when `ROSE2_SOURCE_ROOT` is not set to a `goldleaf3i/declutter-reconstruct`
  checkout containing the required source files.
- `./scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.preprocess_interioragent --dataset-root ${INTERIORAGENT_ROOT:-/home/echo/InteriorAgent} --out data/interioragent_preprocessed --resolution 0.05 --scene-id kujiale_0031`
  passed and wrote one preprocessed scene.
- `./scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.generate_episodes --preprocessed-dir data/interioragent_preprocessed --scene-id kujiale_0031 --episodes-per-scene 10 --out data/interioragent_episodes/debug.jsonl`
  passed and wrote ten deterministic episodes.
- Dry-run smoke command with map backend, A*, and `--allow-debug-fallbacks`
  passed. Its result row has `metric_valid=false` and fallback labels
  `dry_run_detector`, `llm_deterministic_local`, and `static_map_planning`.
- Short Isaac strict smoke command with YOLO-World, SAM2, Isaac backend,
  `--strict-benchmark true`, `--max-control-steps 1`, and `--panorama-steps 0`
  passed. Its result row has `detector_backend=yolo_world`,
  `segmenter_backend=sam2`, `sim_backend=isaac`,
  `map_source=depth_ray_online`, `frontier_count=4`,
  `object_memory_count=11`, graph object/group/room counts, latency fields,
  and `metric_valid=false` because `llm_backend=deterministic_local`.
- Missing strict YOLO-World and SAM2 model paths fail before runtime with clear
  status-2 errors. This behavior is covered by
  `isaac_bench/tests/test_benchmark_contracts.py`.
- Batch runner and summarizer smoke passed on dry-run map rows. The summary
  grouped metric-valid versus non-metric rows and counted fallback labels.

## Hard Requirement Audit

| Requirement | Evidence | Status |
| --- | --- | --- |
| Isaac RGB-D observations | `isaac_bench/env/isaac_process.py`, `isaac_bench/env/observation_adapter.py`, strict smoke row `read_depth=true` | implemented, short integration passed |
| YOLO-World detector | `isaac_bench/perception/yolo_world_detector.py`, `isaac_bench/configs/yolo_world.yaml`, strict asset checks | implemented, short integration passed |
| SAM2 segmenter | `isaac_bench/perception/sam2_segmenter.py`, `isaac_bench/tests/test_sam2_segmenter_contract.py`, strict asset checks | implemented, short integration passed |
| Mask/depth backprojection into object memory | `isaac_bench/sensors/depth_backproject.py`, `isaac_bench/perception/fused_instance_registry.py`, `isaac_bench/perception/object_memory.py`, object-memory tests | implemented with synthetic/unit coverage |
| Online occupancy/free-space map | `isaac_bench/mapping/online_mapper.py`, strict smoke row `mapping_source=depth_ray_online` | implemented, short integration passed |
| Reachable frontier extraction from observed-free/unknown boundary | `isaac_bench/mapping/frontier.py`, `isaac_bench/tests/test_frontier.py`, strict smoke row `frontier_unknown_source=observed` | implemented, short integration passed |
| Object/group/room scene graph where feasible | `isaac_bench/graph/paper_scene_graph.py`, `isaac_bench/graph/sgnav_scenegraph_adapter.py`, `isaac_bench/mapping/upstream_rose2_pure_python_adapter.py`, `isaac_bench/tests/test_paper_scene_graph.py` | implemented with online evidence and debug counts |
| SG-Nav-compatible subgraph text/payloads | `isaac_bench/graph/subgraph_builder.py`, `isaac_bench/tests/test_subgraph_hcot.py`, decision dump contract | implemented with tests |
| LLM or declared LLM-compatible scorer | `isaac_bench/graph/hcot_scorer.py`, strict JSON parser/retry, contract labels deterministic local scorer non-metric | implemented; metric-valid real LLM run remains external-service blocked |
| Subgraph probability to frontier interpolation | `isaac_bench/graph/frontier_interpolation.py`, `isaac_bench/tests/test_frontier_score_calibration.py`, strict smoke `paper_frontier_interpolation.mode=paper_subgraph_interpolation` | implemented |
| Candidate re-perception and credibility accumulation | `isaac_bench/graph/reperception.py`, `isaac_bench/graph/decision.py`, `isaac_bench/tests/test_frontier_reperception.py` | implemented with tests |
| STOP only after confirmation | `isaac_bench/graph/decision.py`, `isaac_bench/tests/test_source_aligned_sgnav.py`, result schema `stop_reason` | implemented with tests |
| Deterministic A*/FMM-equivalent local planner | `isaac_bench/navigation/astar.py`, `isaac_bench/navigation/waypoint_follower.py`, `isaac_bench/tests/test_astar.py` | implemented and smoke-tested |
| Metrics SR/SPL/SoftSPL, distance, path, steps, collisions/stuck, timeout, latencies | `isaac_bench/metrics/evaluator.py`, `isaac_bench/metrics/result_schema.py`, result schema tests | implemented |
| Batch runner and summarizer | `isaac_bench/scripts/run_benchmark.py`, `isaac_bench/metrics/summarize.py`, metrics tests and dry-run batch smoke | implemented |
| Visualization/debug panels | `isaac_bench/visualization/debug_episode.py`, `isaac_bench/visualization/sgnav_popup.py`, `isaac_bench/tests/test_sgnav_popup.py` | implemented with tests |
| README exact commands | `README.md` | implemented |
| No hidden benchmark fallbacks | `docs/benchmark_metric_validity_contract.md`, `isaac_bench/metrics/result_schema.py`, fallback tests | implemented |

## Strict Validity Evidence

Strict rows can be `metric_valid=true` only when `strict_benchmark=true`, no
debug fallback label is present, and required strict assets are available. The
following are always non-metric or rejected:

- `detector_backend=dry_run` produces `dry_run_detector` and
  `metric_valid=false`.
- Seeded GT object memory produces `seeded_gt_object_memory` and
  `metric_valid=false`.
- Deterministic local LLM scoring produces `llm_deterministic_local` and
  `metric_valid=false`, unless the run is an explicitly named ablation.
- Static-map planning produces `static_map_planning` and is valid only for
  smoke or baseline rows, not the SG-Nav metric path.
- Missing YOLO-World, SAM2, Isaac, InteriorAgent, or model files fail clearly
  or skip clearly; they never silently downgrade into metric rows.

## Remaining Risks

- A fully metric-valid strict SG-Nav run still needs a reachable real
  OpenAI-compatible LLM endpoint and must be run with `--llm-enabled true`.
- Long Isaac batch execution was not completed in this audit. A short capped
  Isaac integration smoke passed; one earlier long uncapped run reached the live
  control loop but hit Isaac/PhysX instability warnings and timed out.
- Direct base-shell commands lack project Python dependencies. Use
  `source ./scripts/activate_sgnav_isaac_env.sh`,
  `./run_isaac_bench.sh`, or `./scripts/run_sgnav_isaac_env.sh`.
