# SG-Nav Paper Mechanism Audit

This audit records the current strict SG-Nav paper-path implementation points.

| Requirement | Implementation |
| --- | --- |
| Isaac RGB-D online observations | `run_episode_isaac_closed_loop` forces depth reads and updates `OnlineMapper` from live Isaac observations. |
| YOLO-World + SAM2 metric perception | `ensure_detector_loaded` and `ensure_segmenter_loaded` require YOLO-World and SAM2 assets for strict requested backends. |
| Mask/depth object memory | `FusedInstanceRegistry.update` backprojects detections/masks/depth and `ObjectMemory.update_fused_instances` accumulates instances. |
| Online occupancy/free map | `isaac_bench.mapping.online_mapper.OnlineMapper`. |
| Reachable FBE frontiers | `isaac_bench.mapping.frontier.extract_frontiers`; default `frontier_min_distance_m` is `1.0`. |
| Object scene graph | `PaperSceneGraph.update_from_object_memory`. |
| Group scene graph | `PaperSceneGraph.update_group_nodes`, clustering related nearby objects in the same room context. |
| Room scene graph | `OnlineRoomSegmenter` plus `VLMRoomLabeler`; strict metric path rejects oracle room maps. |
| Object-centered subgraphs | `build_object_centered_subgraphs`, including parent room, parent group, direct neighbors, and edges. |
| HCoT paper scoring | `build_hcot_prior_distance_prompt`, `build_hcot_question_prompt`, `build_hcot_answer_prompt`, `build_hcot_final_distance_prompt`, then `P_sub = 1 / max(distance, min_distance_m)`. |
| Frontier interpolation | `score_frontiers_by_subgraphs` consumes `SubgraphScore.p_sub`; no direct LLM frontier choice in paper mode. |
| Re-perception and STOP | `SGNavDecision` candidate tracks, credibility accumulation, re-perception states, and STOP verification. |
| Deterministic local planning | `GridAStarPlanner` and `HolonomicWaypointFollower`. |
| Metrics and validity | `complete_result_row` records metric validity, fallbacks, latencies, and decision metadata. |
| Visualization evidence | `SGNavPopupVisualizer` saves overlay-layer sidecar JSON and disables GT goal cells by default. |

## Defaults Audited

- `mapping.frontier_min_distance_m: 1.0`
- `sgnav.frontier_distance_weight: 0.2`
- `llm.enabled: true`
- `mapping.room_map_mode: online_geometry_watershed`
- `sgnav.scene_graph.room_nodes.source: online_geometry_watershed_vlm`

## Non-Metric Paths

Dry-run detection, static-map planning, seeded GT object memory, local
deterministic LLM/room labelers, near-frontier fallback, and oracle room maps
are preserved for smoke tests or named ablations only. They add fallback labels
and cannot produce `metric_valid=true` for the main strict SG-Nav path.
