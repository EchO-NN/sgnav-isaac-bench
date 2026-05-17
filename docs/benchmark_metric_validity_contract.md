# Benchmark Metric Validity Contract

This contract defines when a result row may be treated as a valid benchmark
metric row.

## Required Result Fields

Every episode result row must include:

- `metric_valid`
- `strict_benchmark`
- `fallbacks_used`
- `detector_backend`
- `segmenter_backend`
- `llm_backend`
- `sim_backend`
- `map_source`
- `policy_name`
- `sgnav_decision_mode`
- `sgnav_decision_reason`
- `object_memory_count`
- `goal_candidate_count`
- `frontier_count`
- `selected_frontier`
- `stop_reason`
- `success`
- `spl`
- `softspl`
- `distance_to_goal`
- `path_length`
- `steps`
- `collisions`
- `stuck_events`
- `timeout`
- `perception_latency_ms`
- `mapping_latency_ms`
- `graph_latency_ms`
- `llm_latency_ms`
- `planning_latency_ms`

## Metric Validity Rules

- `strict_benchmark=true` means `metric_valid` can be true only when
  `fallbacks_used` is empty.
- `dry_run` detector implies `metric_valid=false`.
- Seeded ground-truth object memory implies `metric_valid=false`.
- Mock or local deterministic LLM scoring implies `metric_valid=false` unless
  the run is explicitly marked as a named ablation.
- Static-map planning may be used for smoke or baseline results only. It is not
  valid for the SG-Nav metric path.
- `room_map_mode=observed_rooms_json`, `rooms_json`, or `observed` implies
  oracle room maps and is not valid for the strict SG-Nav metric path unless
  the run is explicitly named `oracle_room_ablation`.
- Missing or unreachable room VLM backend is not the same as an evidence-based
  `unknown` room label. Backend failure must fail clearly or mark
  `metric_valid=false` with `room_vlm_unavailable` or
  `room_vlm_invalid_json`.
- The configured strict SG-Nav paper path requires `llm.enabled=true`; an
  unavailable HCoT endpoint may not silently become deterministic local scoring.
- YOLO-World, SAM2, LLM, model, Isaac, and InteriorAgent missing assets must
  fail clearly or skip clearly. They must never silently fall back to another
  benchmark path.

## Fallback Labels

Rows must record all known debug or non-metric substitutions in
`fallbacks_used`. Canonical labels include:

- `dry_run_detector`
- `detector_none`
- `seeded_gt_object_memory`
- `gt_goal_fallback_allowed`
- `gt_goal_fallback_used`
- `sam2_missing_or_disabled`
- `llm_deterministic_local`
- `room_vlm_deterministic_debug`
- `room_vlm_unavailable`
- `room_vlm_invalid_json`
- `oracle_room_map`
- `static_map_planning`
- `static_nearfield_map`
- `frontier_near_fallback`

Additional labels may be added when new debug paths are introduced, but labels
must be stable once result files have been produced.

## Strict Asset Rule

When a strict metric-capable run requests YOLO-World or SAM2 and the configured
model/checkpoint asset is missing, the command must fail before writing a
metric row. The error message must include the missing asset kind and path.

## Ablation Rule

A named ablation may intentionally use a non-paper scorer or alternate
component, but the row must still expose `fallbacks_used`, `policy_name`, and
the ablation name so it cannot be confused with the main SG-Nav metric path.
