# SG-Nav Paper Parity Contract

This contract defines the required SG-Nav mechanism for metric-path
Isaac/InteriorAgent benchmark runs. It is intentionally stricter than smoke
tests and debug tools.

## Metric-Path Mechanism

Metric-path SG-Nav runs must use the following pipeline:

1. Isaac RGB-D observations are the only online perception and mapping input.
2. YOLO-World detections provide metric-path open-vocabulary object boxes.
3. SAM2 masks are prompted from YOLO-World boxes for metric-path segmentation.
4. SAM2 mask pixels and RGB-D depth are backprojected into online 3D object
   memory.
5. Online occupancy and free-space maps are updated from live RGB-D evidence.
6. Reachable frontiers are extracted from the observed-free and unknown-space
   boundary.
7. The online scene graph represents objects, groups, and rooms where feasible.
8. SG-Nav-compatible subgraph text or payloads are generated from the scene
   graph.
9. Subgraphs are scored by an LLM or by an explicitly declared
   LLM-compatible scorer.
10. Subgraph probabilities are interpolated onto reachable frontiers.
11. Candidate goals are re-perceived and their credibility is accumulated from
    graph and detector evidence.
12. STOP is allowed only after SG-Nav goal confirmation rules are satisfied.
13. Local motion uses a deterministic A* planner or an FMM-equivalent grid
    planner.

## Forbidden Metric-Path Substitutions

- Ground-truth object memory must not replace YOLO-World plus SAM2 perception.
- Static map knowledge must not replace online RGB-D occupancy/free-space
  mapping for SG-Nav exploration decisions.
- Nearest-frontier, distance-only, or category-only heuristics must not replace
  SG-Nav subgraph probability to frontier interpolation.
- Candidate proximity alone must not trigger STOP without the configured
  re-perception and confirmation state.
- Missing model, dataset, simulator, or LLM assets must not silently switch the
  run into a debug fallback.

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
