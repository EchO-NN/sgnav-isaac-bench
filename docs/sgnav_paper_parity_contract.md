# SG-Nav Paper Parity Contract

This contract defines the required SG-Nav mechanism for metric-path
Isaac/InteriorAgent benchmark runs. It is intentionally stricter than smoke
tests and debug tools.

## Metric-Path Mechanism

Metric-path SG-Nav runs must use the following pipeline:

1. Isaac RGB-D observations are the only online perception and mapping input.
2. GroundingDINO-B/Swin-B detections provide metric-path open-vocabulary object boxes.
3. SAM2 masks are prompted from GroundingDINO boxes for metric-path segmentation.
4. A GroundingDINO/SAM detection is valid only when `confidence > 0.45`. Detections at
   or below 0.45 must not draw RGB bboxes, enter mask/depth fusion, create
   object-memory nodes, create scene-graph object nodes, seed candidate goals,
   or support STOP confirmation.
5. GroundingDINO boxes or SAM2 masks that touch the image edge are partial raw
   evidence, not hard-discarded detections. They are logged with
   `visibility_status=partial_edge`, associated to prior full/stable tracks by
   mask/footprint overlap when possible, and otherwise kept as tentative
   tracks that cannot create goal candidates, support STOP, or become SG-Nav
   policy graph nodes until stable.
6. SAM2 mask pixels and RGB-D depth are backprojected into online 3D object
   memory. Object tracks maintain accumulated `class_conf_sums`/`class_hits`;
   the current category is the accumulated winner, not the most recent frame.
7. Online occupancy and free-space maps are updated from live RGB-D evidence.
8. Reachable frontiers are extracted from the observed-free and unknown-space
   boundary.
9. The online scene graph represents object nodes, clustering-based group
   nodes, and online geometry room-mask nodes where feasible. Metric room nodes
   must not use `rooms.json` labels or oracle room masks.
10. Room masks are prepared lazily only for SG-Nav frontier scoring. After
   reachable frontier extraction, `prepare_room_context_for_frontier_scoring`
   runs room segmentation/labeling immediately before scene graph update and
   HCoT/subgraph frontier interpolation. Mapper-only updates, perception-only
   updates, committed-frontier local replans, and candidate re-perception/STOP
   confirmation must use cached context and must not trigger room VLM calls.
11. Strict room masks come from `upstream_rose2_vertical_or_free`: a no-ROS
   pure-Python adapter for the MIT `goldleaf3i/declutter-reconstruct` ROSE2
   paper code. It operates on the online 2D occupancy grid, estimates DFT/FFT
   dominant wall directions, filters robust structure, extracts Hough/clustered
   wall lines, creates room masks/polygons, preserves stable IDs, and runs
   two-stage room recognition immediately before frontier scoring. Strict runs
   must fail clearly if `ROSE2_SOURCE_ROOT` does not point to a checkout
   containing `code/FFT_MQ.py`, `code/minibatch.py`, and `code/parameters.py`.
   Local ROSE2-lite and the old watershed segmenter are debug/ablation only.
   Premerge room labels can preserve weak/open-plan functional splits only
   when both sides are reliable, non-unknown, and different; final room masks
   are labeled again for SG-Nav room nodes. `unknown` is a valid first-class
   room category when evidence is weak or ambiguous, but it does not create
   functional splits without a structural boundary.
   The ROSE2 path is vertical-profile-assisted: per-cell `low`, `robot_body`,
   `mid`, and `upper` height-band counts build a `vertical_carved_map` and
   `wall_confidence_map`. Vertical free evidence may suppress furniture or
   clutter, but it must not create a room connection. Only floor-level
   traversability plus wall-line support can create a verified doorway; window,
   curtain, glass, high-band, and exterior/perimeter gaps are closed as
   wall-like for room segmentation.
12. VLM room-label `confidence` is only `vlm_self_confidence`, not calibrated
   probability. Room node confidence must use evidence-derived
   `label_reliability`; evidence-gated `unknown` remains metric-valid.
13. SG-Nav-compatible object-centered subgraph text or payloads are generated
   from the scene graph, including central object, parent room mask/label,
   parent group, direct object neighbors, and edges.
14. Subgraphs are scored by the SG-Nav paper HCoT sequence: prior
   object-goal distance, distance-prediction questions, subgraph-grounded
   answers, final distance summary, then `P_sub = 1 / max(distance,
   min_distance_m)`.
15. Subgraph probabilities are interpolated onto reachable frontiers. The paper
   path must not ask an LLM to directly choose frontiers.
16. Candidate goals are re-perceived and their credibility is accumulated from
    graph and detector evidence.
17. STOP is allowed only after SG-Nav goal confirmation rules are satisfied.
18. Local motion uses a deterministic A* planner or an FMM-equivalent grid
    planner.

## Forbidden Metric-Path Substitutions

- Ground-truth object memory must not replace GroundingDINO-B/Swin-B plus SAM2 perception.
- Static map knowledge must not replace online RGB-D occupancy/free-space
  mapping for SG-Nav exploration decisions.
- Nearest-frontier, distance-only, or category-only heuristics must not replace
  SG-Nav subgraph probability to frontier interpolation.
- Candidate proximity alone must not trigger STOP without the configured
  re-perception and confirmation state.
- Missing model, dataset, simulator, or LLM assets must not silently switch the
  run into a debug fallback.
- Dataset room labels or preprocessed oracle room masks must not create strict
  metric room nodes. `oracle_room_ablation` is the only allowed named ablation.
- Frontier selection must ignore frontiers closer than `1.0 m` unless an
  explicit near-frontier fallback is enabled, which makes the row non-metric.

## Allowed Non-Metric Uses

The following are allowed only when the output is marked non-metric/debug or as
a named ablation:

- `dry_run` detector smoke tests.
- Seeded ground-truth object memory smoke tests.
- Static-map A* smoke or baseline commands.
- Missing-model stubs.
- Deterministic local scorer in place of the declared LLM path.

## Review Rule

Any round touching perception, mapping, scene graph reasoning, frontier
selection, re-perception, STOP logic, planner/controller behavior, benchmark
runner output, or README benchmark commands must cite this contract in its
review evidence.
