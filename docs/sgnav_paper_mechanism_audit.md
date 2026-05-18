# SG-Nav Paper Mechanism Audit

This audit records the strict SG-Nav paper-path implementation. Status labels
are intentionally conservative: `faithful`, `optimized-equivalent`,
`engineered-faithful`, and `approximate-but-justified`.

| SG-Nav paper mechanism | Current implementation | Status | Evidence |
| --- | --- | --- | --- |
| RGB-D observation | Isaac closed loop reads RGB-D observations before online mapping and perception. | faithful | `isaac_bench/scripts/run_one_episode.py::run_episode_isaac_closed_loop` |
| GLIP open-vocabulary detection | GroundingDINO-B/Swin-B replaces GLIP as the open-vocabulary detector in the metric path. | optimized-equivalent | `isaac_bench/perception/grounding_dino_detector.py`, `ensure_detector_loaded` |
| SAM mask refinement | SAM2 replaces SAM mask refinement and attaches box-prompt masks. | optimized-equivalent | `isaac_bench/perception/sam2_segmenter.py`, `ensure_segmenter_loaded` |
| online occupancy/free map | Live depth is fused into online occupied/free/observed grids. | engineered-faithful | `isaac_bench/mapping/online_mapper.py::OnlineMapper` |
| frontier extraction | Reachable frontiers are extracted at observed-free/unknown boundaries. | engineered-faithful | `isaac_bench/mapping/frontier.py::extract_frontiers` |
| object nodes | Non-edge GroundingDINO-B/Swin-B/SAM2/depth detections accumulate into object memory with track-level category confidence sums before `PaperSceneGraph` object nodes are emitted. | engineered-faithful | `FusedInstanceRegistry.update`, `ObjectMemory.update_fused_instances`, `PaperSceneGraph.update_from_object_memory` |
| group nodes | Related object groups remain cluster-based in the online scene graph. | engineered-faithful | `PaperSceneGraph.update_group_nodes` |
| room nodes | Room nodes are no-ROS upstream ROSE2 pure-Python room masks plus object/VLM semantics, not oracle `rooms.json` in strict mode. | engineered-faithful | `isaac_bench/mapping/upstream_rose2_pure_python_adapter.py`, `isaac_bench/graph/room_semantics.py` |
| object-room edges | Objects are assigned to room masks by centroid or footprint overlap. | engineered-faithful | `assign_objects_to_room_masks`, `PaperSceneGraph.update_affiliation_edges` |
| object-centered subgraphs | Subgraphs include central object, parent room, parent group, direct neighbors, and edges. | faithful | `isaac_bench/graph/subgraph_builder.py::build_object_centered_subgraphs` |
| HCoT stage 1 | LLM predicts prior object-goal distance. | faithful | `build_hcot_prior_distance_prompt` |
| HCoT stage 2 | LLM asks distance-relevant object-goal questions. | faithful | `build_hcot_question_prompt` |
| HCoT stage 3 | LLM answers questions from subgraph context. | faithful | `build_hcot_answer_prompt` |
| HCoT stage 4 | LLM summarizes final subgraph-goal distance. | faithful | `build_hcot_final_distance_prompt` |
| P_sub | `P_sub = 1 / max(stage4_final_distance_m, min_distance_m)`. | faithful | `score_subgraph_with_hcot` |
| frontier interpolation | Frontier scores use `sum(P_sub / D(frontier, subgraph_center))`. | faithful | `isaac_bench/graph/frontier_interpolation.py::score_frontiers_by_subgraphs` |
| distance bias | Distance bias is an engineered calibration term with default `frontier_distance_weight=0.2`. | engineered-faithful | `isaac_bench/graph/decision.py::SGNavDecision.choose_frontier` |
| candidate re-perception | Candidate credibility accumulates detector confidence and graph/subgraph support; acceptance wins when threshold is reached, including at `n_max`. | engineered-faithful | `isaac_bench/graph/reperception.py::GraphReperceptionManager.update` |
| STOP | STOP is allowed only after SG-Nav candidate confirmation / stop verification. | engineered-faithful | `SGNavDecision.choose_navigation_target`, `build_stop_state_payload` |

## T0 Room Merge And Object Memory Rules

Upstream ROSE2 physical room proposals are not final semantic rooms. Before
frontier scoring, premerge proposals are labeled only for open-plan merge
decisions; final merged room masks are then labeled again for SG-Nav room
nodes. Verified structural boundaries preserve splits. Weak/open boundaries
preserve a functional split only when both proposal labels are reliable,
non-unknown, and different. Object lists such as sink/fridge/sofa/TV are
evidence to the room recognizer only, not authoritative split rules.

GroundingDINO detections at or below `0.45` confidence are rejected before bbox drawing,
SAM2/depth fusion, object memory, candidate-goal logic, STOP, policy graph
objects, and first-version GNN policy features. Edge-touching GroundingDINO/SAM2
detections are not hard-discarded: they remain raw evidence, may associate to a
stable full track by mask/footprint overlap, and otherwise stay tentative and
ineligible for policy/room/goal/STOP. Object tracks expose the accumulated
winner category plus `mean_confidence`, `detection_count`, and
`winner_detection_count`.

## T0 Runtime Scheduling

Room segmentation and room VLM labeling are scoring-gated. The strict call order
for a new SG-Nav frontier decision is:

1. frontier extraction
2. `prepare_room_context_for_frontier_scoring`
3. `scenegraph.update_from_frame(..., room_masks=..., room_semantic_labels=...)`
4. HCoT subgraph scoring
5. `P_sub / D` frontier interpolation
6. frontier selection

The room context hook is not called for mapper-only updates, perception-only
updates, committed-frontier local replans, or candidate re-perception/STOP
confirmation. Candidate re-perception uses the latest cached graph context.

## Defaults Audited

- `mapping.frontier_min_distance_m: 1.0`
- `sgnav.frontier_distance_weight: 0.2`
- `llm.enabled: true`
- `mapping.room_map_mode: upstream_rose2_vertical_or_free`
- `mapping.room_segmentation.algorithm: upstream_rose2_vertical_or_free`
- `mapping.room_segmentation.source_mode: declutter_reconstruct_mit`
- `mapping.room_segmentation.require_upstream_source_for_strict: true`
- `sgnav.scene_graph.room_nodes.source: upstream_rose2_vertical_or_free_vlm`

## Benchmark Validity Notes

Dry-run detection, static-map planning, seeded GT object memory, deterministic
local LLM/room labelers, near-frontier fallback, and oracle room maps are
preserved for smoke tests or named ablations only. They add fallback labels and
cannot produce `metric_valid=true` for the main strict SG-Nav path.

VLM room-label `confidence` is stored as `vlm_self_confidence`, a self-reported
ordinal signal. Room node `confidence` uses evidence-derived
`label_reliability`; evidence-based `unknown` remains metric-valid, while
missing/unavailable VLM backend remains non-metric or a strict failure.
