# Roadmap

## Status Legend

- `pending`
- `in-progress`
- `done`

## Goal

Turn `EchO-NN/sgnav-isaac-bench` into a complete, reproducible, runnable
SG-Nav Isaac/InteriorAgent benchmark project whose reported metric path uses
Isaac RGB-D observations, YOLO-World plus SAM2 perception, online mapping,
SG-Nav-compatible scene graph reasoning, deterministic local planning, explicit
metrics, batch execution, visualization, tests, and README commands.

## Alignment Summary

- Approved thesis: Complete the benchmark by staged, review-gated hardening of
  the existing `isaac_bench/` package, preserving the already-runnable smoke and
  A* paths while making metric eligibility, SG-Nav paper parity, and external
  asset requirements explicit.
- Success criteria: A user can follow README commands to set up the environment,
  preprocess InteriorAgent assets, generate episodes, run dry-run non-Isaac
  smoke, run Isaac closed-loop SG-Nav paper mode with YOLO-World plus SAM2 when
  assets exist, run batches, summarize SR/SPL/SoftSPL and latency metrics, view
  debug panels, and see clear errors or skips for missing Isaac, InteriorAgent,
  model, or LLM assets.
- Non-goals: This roadmap does not replace YOLO-World/SAM2 with GT perception
  for metric results, does not silently make debug fallbacks count as benchmark
  metrics, does not switch away from the SG-Nav mechanism except for the
  Isaac/InteriorAgent runtime substrate, and does not rewrite the package into
  an unrelated navigation architecture.
- Chosen strategy: Use a safety-first staged completion plan. First make
  benchmark-mode contracts and regression protection explicit, then advance
  environment/data reproducibility, Isaac closed-loop execution, perception,
  SG-Nav reasoning parity, metrics/batch/visualization, and final docs/tests.
  This is chosen over a metric-first shortcut because the project already has
  broad modules and must avoid hidden fallbacks.
- Deferred alternatives: A monolithic "finish everything" patch is deferred as
  too hard to review; a metric-first runner polish strategy is deferred because
  it could certify incomplete SG-Nav behavior; a parallel-heavy strategy is
  deferred until milestone interfaces and fallback semantics are proven stable.

## Outcome Boundaries

- In scope: `isaac_bench/` package completion, top-level setup/run scripts,
  Isaac and InteriorAgent preprocessing/episode workflows, YOLO-World detector,
  SAM2 segmentation, RGB-D fusion and object memory, online mapping/frontiers,
  SG-Nav scene graph and subgraph reasoning, LLM scoring adapters, candidate
  re-perception and STOP logic, deterministic planner/controller, metrics,
  batch runner, summarizer, visualization/debug panels, tests, and README
  commands.
- Out of scope: new simulator substrates beyond Isaac/InteriorAgent, replacing
  YOLO-World/SAM2 in the real benchmark path, silently depending on private
  local paths without documented configuration, reporting metrics from fallback
  debug modes, or changing completed orchestrator history outside the active
  roadmap update process.
- Repo-wide invariants: see `orchestrator/project-contract.md`; this roadmap
  should record only roadmap-specific overrides.
- Completed history: keep compact notes in
  `orchestrator/roadmaps/2026-05-16-00-sgnav-isaac-benchmark-completion/roadmap-history.md` instead of
  copying completed item bodies into new active revisions.

## Global Sequencing Rules

- Keep controller execution serial by default. The first accepted rounds should
  make metric/debug mode separation testable before feature-completion rounds
  depend on it.
- Do not extract work that changes reported metrics, benchmark fallbacks, or
  result schemas in parallel with other shared runner/policy changes unless a
  reviewed `worker-plan.json` gives disjoint ownership and integration order.
- Any round touching runtime behavior must end with pytest where possible, one
  dry-run non-Isaac smoke command, and one Isaac command or clear skip evidence
  for missing Isaac/InteriorAgent/model assets.
- Preserve existing smoke tests and A* benchmark commands. If a round needs to
  change a CLI, it must keep a compatibility alias or update docs and tests in
  the same round.
- Missing external assets must fail clearly or skip clearly. A round may add
  debug stubs only when the CLI/config labels outputs as non-metric/debug.

## Parallel Lanes

- `lane-contract`: benchmark-mode safety, CLI compatibility, result schemas,
  configuration, and README command contracts. Serial with every other lane
  until milestone 1 is complete.
- `lane-data-runtime`: environment scripts, InteriorAgent preprocessing,
  episode generation, Isaac closed-loop execution, mapping, planning, and
  controller behavior.
- `lane-perception-graph`: YOLO-World, SAM2, depth fusion, object memory,
  scene graph, LLM/subgraph scoring, frontier interpolation, re-perception, and
  STOP confirmation.
- `lane-reporting-viz`: metrics, batch runner, summarizer, debug panels,
  artifacts, documentation, and regression tests.

## Milestones

### [pending] Milestone 1: Benchmark Contract And Safety Gates
Milestone id: milestone-001-contract-safety
Depends on:
Intent: Make the project contract enforceable before implementation rounds can
claim metric completeness.
Completion signal: Metric and debug modes are distinguishable in CLI/config,
hidden fallback risks have regression coverage, existing smoke/A* commands
still work, and roadmap verification can reject benchmark-unsafe changes.
Milestone acceptance gates:
- `docs/sgnav_paper_parity_contract.md` defines the exact metric-path SG-Nav
  mechanism.
- `docs/benchmark_metric_validity_contract.md` defines strict benchmark
  validity, fallback labels, and missing-asset behavior.
- `python -m isaac_bench.scripts.dump_sgnav_step ...` writes the required
  SG-Nav decision dump JSON artifact shape.
- Pytest passes for mode contract, result schema, fallback policy, and config
  defaults.
- A dry-run smoke result is marked `metric_valid=false`.
- A strict benchmark run with missing YOLO-World or SAM2 fails clearly before
  silently falling back.
- The existing A* smoke command still works or has a compatibility alias.
Parallel lane: lane-contract
Coordination notes: Start here. Keep changes small and reviewable because later
milestones rely on these invariants.

Candidate directions:
- Direction id: direction-001-mode-contract
  Summary: Add or harden explicit metric-vs-debug run semantics across config,
    CLI, output rows, and tests.
  Why it matters now: The user's hardest constraint is that fallbacks stay
    benchmark-safe and never silently contaminate reported metrics.
  Preconditions: none
  Parallel hints: Serial; this defines shared runner semantics.
  Boundary notes: Do not change detector quality or policy logic except as
    needed to label or reject unsafe modes.
  Extraction notes: Acceptance should include tests proving YOLO-World/SAM2 is
    required for metric runs and dry-run/GT/stub modes are labeled non-metric or
    rejected.

- Direction id: direction-002-regression-baseline
  Summary: Establish baseline regression commands and fixtures for pytest,
    map/A* dry-run smoke, and clear Isaac skip behavior.
  Why it matters now: Every later round needs reproducible verification instead
    of ad hoc local screenshots.
  Preconditions: none
  Parallel hints: Can run after or with direction-001 only if it avoids shared
    benchmark-mode fields.
  Boundary notes: Keep generated data ignored; version only minimal fixtures or
    fixture builders when necessary.
  Extraction notes: Acceptance should show current smoke commands still parse
    and either run or skip with exact missing-asset evidence.

- Direction id: direction-003-config-audit
  Summary: Audit and align `isaac_bench/configs/` defaults for paper-mode
    benchmark safety.
  Why it matters now: Config defaults determine whether README commands perform
    real SG-Nav benchmark work or a debug shortcut.
  Preconditions: direction-001-mode-contract preferred
  Parallel hints: Serial with runner changes; may co-run with docs-only review
    after mode semantics settle.
  Boundary notes: Do not tune performance heuristics beyond safety and clarity.
  Extraction notes: Acceptance should include config tests or snapshot checks
    for detector, segmenter, mapping, SG-Nav policy, metrics, and fallback flags.

### [pending] Milestone 2: Reproducible Environment, Assets, And Episodes
Milestone id: milestone-002-reproducible-data
Depends on: milestone-001-contract-safety
Intent: Make setup, asset discovery, InteriorAgent preprocessing, and episode
generation reproducible for a new user while preserving current local paths.
Completion signal: README setup commands, environment scripts, preprocessing,
episode generation, and asset/model checks produce deterministic outputs or
clear missing-asset errors.
Parallel lane: lane-data-runtime
Coordination notes: Keep external assets out of git; make path overrides and
skip/error behavior explicit.

Candidate directions:
- Direction id: direction-004-env-scripts
  Summary: Harden `scripts/setup_sgnav_isaac_env.sh`,
    `scripts/activate_sgnav_isaac_env.sh`, and
    `scripts/run_sgnav_isaac_env.sh` for Isaac 5.1, Python environment
    activation, and model dependency checks.
  Why it matters now: Users need reliable setup before Isaac and perception
    work can be validated.
  Preconditions: milestone-001-contract-safety
  Parallel hints: May co-run with episode docs only if script ownership is
    separate.
  Boundary notes: Do not vendor Isaac, InteriorAgent, YOLO-World, SAM2, or LLM
    model files.
  Extraction notes: Acceptance should include dry-runable shell checks and clear
    failures for missing Isaac root or model paths.

- Direction id: direction-005-preprocess-contract
  Summary: Complete InteriorAgent preprocessing contracts for USD semantics,
    room/object extraction, occupancy/navigability artifacts, passable openings,
    and debug maps.
  Why it matters now: Online SG-Nav evaluation needs consistent scene geometry,
    goal objects, room hierarchy seeds, and traversability baselines.
  Preconditions: milestone-001-contract-safety
  Parallel hints: Can co-run with environment work after CLI contracts are
    stable.
  Boundary notes: Keep preprocessing deterministic and avoid metric-only ground
    truth leakage into online perception.
  Extraction notes: Acceptance should include tests for generated artifact names
    and a smoke preprocess skip path when InteriorAgent assets are absent.

- Direction id: direction-006-episode-generation
  Summary: Harden episode generation for object categories, start/goal clearance,
    shortest-path metadata, success distance, and reproducible seeds.
  Why it matters now: Benchmark metrics are only meaningful with valid,
    reproducible episodes.
  Preconditions: direction-005-preprocess-contract
  Parallel hints: Serial after preprocessing artifact contracts.
  Boundary notes: Do not rely on seeded GT memory during metric execution; GT
    data may define episodes and evaluation goals only.
  Extraction notes: Acceptance should include deterministic generation tests and
    README command verification.

### [pending] Milestone 3: Isaac Closed Loop, Online Mapping, And Planning
Milestone id: milestone-003-isaac-online-runtime
Depends on: milestone-002-reproducible-data
Intent: Complete the Isaac runtime substrate while keeping navigation mechanics
SG-Nav-compatible and deterministic.
Completion signal: Isaac closed-loop episodes consume live RGB-D observations,
build online occupancy/free-space maps, extract reachable frontiers, plan with
A*/FMM-equivalent deterministic grids, control Kaya, and report collisions,
stuck events, timeouts, path length, and steps.
Parallel lane: lane-data-runtime
Coordination notes: Runtime work may expose simulator-specific issues, but it
must not replace online SG-Nav mechanisms with static-map metric shortcuts.

Candidate directions:
- Direction id: direction-007-isaac-observation-loop
  Summary: Harden Isaac process startup, reset, camera RGB-D capture, robot
    pose synchronization, and headless/headed behavior.
  Why it matters now: Perception and mapping correctness depends on live Isaac
    observations being reliable.
  Preconditions: milestone-002-reproducible-data
  Parallel hints: Serial with controller changes; may co-run with visualization
    panel work after observation payloads are stable.
  Boundary notes: Keep map-only A* smoke path working.
  Extraction notes: Acceptance should include Isaac availability skip behavior
    and a dry-run non-Isaac smoke command.

- Direction id: direction-008-online-mapping-frontiers
  Summary: Complete RGB-D online occupancy/free-space mapping and SG-Nav-style
    frontier extraction from observed unknown boundaries.
  Why it matters now: The paper mechanism requires online maps rather than
    static omniscient traversability for exploration decisions.
  Preconditions: direction-007-isaac-observation-loop preferred
  Parallel hints: Can co-run with perception only if shared observation types
    are stable.
  Boundary notes: Static near-field or map-fill helpers must remain explicit
    debug-only behavior.
  Extraction notes: Acceptance should include frontier/mapping unit tests and
    a smoke command that records online map debug stats.

- Direction id: direction-009-planner-controller
  Summary: Harden deterministic grid planning, waypoint following, collision
    handling, no-progress/stuck detection, timeout handling, and path-length
    accounting.
  Why it matters now: SPL, SoftSPL, collisions, and STOP validity depend on
    deterministic local execution.
  Preconditions: direction-008-online-mapping-frontiers preferred
  Parallel hints: Serial with metrics schema changes.
  Boundary notes: Distance-only planner behavior is allowed only for smoke/debug
    and must be labeled non-metric when used as a fallback.
  Extraction notes: Acceptance should include A* regression tests and result-row
    checks for steps, path length, timeout, collisions, and stuck events.

### [pending] Milestone 4: YOLO-World, SAM2, And Object Memory
Milestone id: milestone-004-perception-memory
Depends on: milestone-002-reproducible-data
Intent: Make the real benchmark perception path model-backed, explicit, and
integrated with online 3D object memory.
Completion signal: YOLO-World detections, SAM2 masks, depth backprojection,
instance fusion, confidence/hit accumulation, and missing-model behavior are
tested and cannot silently downgrade metric runs.
Parallel lane: lane-perception-graph
Coordination notes: Perception can advance alongside runtime only after shared
observation payload and benchmark-mode contracts are stable.

Candidate directions:
- Direction id: direction-010-yolo-world-loader
  Summary: Harden YOLO-World model loading, vocabulary prompts, confidence/NMS
    config, CUDA/CPU handling, IPC worker boundaries, and error messages.
  Why it matters now: YOLO-World is a hard requirement for the real benchmark
    path.
  Preconditions: milestone-001-contract-safety
  Parallel hints: May co-run with SAM2 work if detector and segmenter files are
    disjoint and integration is planned.
  Boundary notes: IPC may be supported, but it must not hide missing model or
    dependency failures in metric runs.
  Extraction notes: Acceptance should include tests/mocks for model absence,
    vocabulary setting, and non-metric fallback labeling.

- Direction id: direction-011-sam2-segmentation
  Summary: Harden SAM2 box-prompt segmentation, checkpoint/config discovery,
    mask attachment, and required-vs-auto behavior.
  Why it matters now: The real path must project mask pixels, not only YOLO
    boxes, into object memory when SAM2 is required.
  Preconditions: milestone-001-contract-safety
  Parallel hints: Can co-run with YOLO loader if integration tests are assigned.
  Boundary notes: `segmenter: auto` may be useful for smoke, but metric runs
    requiring SAM2 must fail clearly when SAM2 is unavailable.
  Extraction notes: Acceptance should include missing-checkpoint tests and mask
    propagation tests.

- Direction id: direction-012-object-memory-fusion
  Summary: Complete depth fusion, 3D detection localization, object-memory merge
    policy, hit/age tracking, and candidate credibility inputs.
  Why it matters now: SG-Nav re-perception and STOP confirmation depend on
    accumulated object evidence rather than one-frame detections.
  Preconditions: direction-010-yolo-world-loader and direction-011-sam2-segmentation
  Parallel hints: Serial with candidate re-perception changes.
  Boundary notes: Seeded GT memory remains smoke-only and must be labeled
    non-metric.
  Extraction notes: Acceptance should include object-memory and fused-instance
    tests using synthetic detections and masks.

### [pending] Milestone 5: SG-Nav Scene Graph, LLM Reasoning, Re-Perception, And STOP
Milestone id: milestone-005-sgnav-policy-parity
Depends on: milestone-003-isaac-online-runtime, milestone-004-perception-memory
Intent: Reproduce SG-Nav decision mechanics over the Isaac/InteriorAgent runtime
without replacing the paper algorithm with unrelated heuristics.
Completion signal: Runtime builds object/group/room hierarchy where feasible,
serializes SG-Nav-compatible subgraphs, scores subgraphs through LLM or
LLM-compatible adapters, interpolates subgraph probabilities to frontiers,
selects frontiers/candidates, accumulates graph-based re-perception
credibility, and calls STOP only under SG-Nav confirmation rules.
Parallel lane: lane-perception-graph
Coordination notes: This milestone is policy-critical. Prefer one mechanism per
round and keep reviewer evidence close to the SG-Nav paper requirement it
protects.

Candidate directions:
- Direction id: direction-013-scene-graph-hierarchy
  Summary: Complete online scene graph construction with object, group, room,
    spatial relationship, edge proposal, and pruning behavior.
  Why it matters now: Hierarchical reasoning needs structured graph context,
    not only flat detection lists.
  Preconditions: direction-012-object-memory-fusion
  Parallel hints: May co-run with LLM adapter tests only when graph data
    contracts are stable.
  Boundary notes: Room/group hierarchy should be implemented where feasible and
    documented where InteriorAgent/online evidence is insufficient.
  Extraction notes: Acceptance should include graph unit tests and debug dumps
    showing object, group, room, and edge counts.

- Direction id: direction-014-subgraph-llm-scoring
  Summary: Complete SG-Nav-compatible subgraph text/payload construction,
    hierarchical LLM scoring, strict JSON parsing/retry, and deterministic
    local scorer fallback policy.
  Why it matters now: Frontier probabilities must come from SG-Nav-style
    subgraph reasoning, not arbitrary distance or category-only heuristics.
  Preconditions: direction-013-scene-graph-hierarchy
  Parallel hints: Serial with frontier interpolation if score schema is still
    changing.
  Boundary notes: Deterministic local scoring is allowed for tests and debug; it
    must be labeled or configured so metric claims are honest when no LLM is
    used.
  Extraction notes: Acceptance should include strict parser tests and score
    payload snapshots.

- Direction id: direction-015-frontier-interpolation
  Summary: Complete subgraph probability to frontier interpolation and frontier
    selection with SG-Nav-compatible `P_sub / D` behavior and calibration tests.
  Why it matters now: Exploration policy depends on mapping scene graph scores
    onto reachable frontiers.
  Preconditions: direction-014-subgraph-llm-scoring and direction-008-online-mapping-frontiers
  Parallel hints: Serial with planner selection changes.
  Boundary notes: Do not reintroduce nearest-frontier-only metric behavior.
  Extraction notes: Acceptance should include frontier score calibration tests
    and debug payload evidence.

- Direction id: direction-016-reperception-stop
  Summary: Complete candidate-goal credibility accumulation, graph-based
    re-perception, candidate rejection TTL, standoff selection, and STOP
    verification.
  Why it matters now: SG-Nav should stop only when the goal is confirmed, not
    simply when a category is nearby or a static goal region is reached.
  Preconditions: direction-012-object-memory-fusion and direction-015-frontier-interpolation
  Parallel hints: Serial with metrics success criteria.
  Boundary notes: GT goal fallback is debug-only and must never be silently used
    for metric STOP confirmation.
  Extraction notes: Acceptance should include tests for credibility accumulation,
    re-perception thresholds, and STOP rejection when confirmation fails.

### [pending] Milestone 6: Metrics, Batch Runner, Summaries, And Visualization
Milestone id: milestone-006-reporting-visualization
Depends on: milestone-003-isaac-online-runtime, milestone-005-sgnav-policy-parity
Intent: Make benchmark execution, reporting, and debugging complete enough for
repeatable development and reported results.
Completion signal: Batch runner supports map and Isaac modes as appropriate,
summaries compute required metrics, latency measurements are recorded, debug
panels expose RGB detections/object memory/map/frontiers/path/graph scores, and
artifacts remain reproducible and ignored when generated.
Parallel lane: lane-reporting-viz
Coordination notes: Metrics schema changes are shared-contract changes and must
be reviewed with compatibility in mind.

Candidate directions:
- Direction id: direction-017-metric-schema
  Summary: Complete per-episode result rows for SR, SPL, SoftSPL,
    distance-to-goal, path length, steps, collisions/stuck events, timeout,
    perception latency, graph latency, and planning latency.
  Why it matters now: The benchmark is not complete until reported metrics are
    explicit and reproducible.
  Preconditions: direction-009-planner-controller preferred
  Parallel hints: Serial with runner and summarizer changes.
  Boundary notes: Preserve existing fields or provide compatibility aliases.
  Extraction notes: Acceptance should include metrics tests and sample result
    row validation.

- Direction id: direction-018-batch-runner-summarizer
  Summary: Complete batch benchmark runner and summarizer for grouped results,
    failure reasons, non-metric labels, and reproducible output paths.
  Why it matters now: Single-episode smoke commands are not enough for benchmark
    runs.
  Preconditions: direction-017-metric-schema
  Parallel hints: Can co-run with visualization after metrics schema is stable.
  Boundary notes: Do not make `run_benchmark` silently force map-only behavior
    when Isaac mode is requested.
  Extraction notes: Acceptance should include CLI tests and summary tests over
    synthetic JSONL rows.

- Direction id: direction-019-debug-panels
  Summary: Complete visualization/debug panels for RGB detections, object
    memory, observed map, frontiers, selected candidate/frontier, A* path, and
    scene-graph score state.
  Why it matters now: Debuggability is required to diagnose Isaac perception and
    policy failures without weakening the metric path.
  Preconditions: direction-013-scene-graph-hierarchy and direction-008-online-mapping-frontiers preferred
  Parallel hints: Can co-run with docs or batch work if it avoids metric schema
    ownership.
  Boundary notes: Generated images/videos stay under ignored `debug/` or
    configured output directories.
  Extraction notes: Acceptance should include a saved-panel smoke or skip
    evidence on headless hosts.

### [pending] Milestone 7: Documentation, Final Regression, And Release Readiness
Milestone id: milestone-007-release-readiness
Depends on: milestone-006-reporting-visualization
Intent: Freeze the reproducible user path and verify no hidden fallback or
documentation drift remains.
Completion signal: README has exact commands, configs are complete, tests cover
critical contracts, setup/asset errors are clear, smoke/A*/Isaac commands are
verified or explicitly skipped with reasons, and the repository is ready for a
user to run the benchmark from documented steps.
Parallel lane: lane-reporting-viz
Coordination notes: This milestone should mostly integrate evidence from prior
rounds and close gaps discovered by review.

Candidate directions:
- Direction id: direction-020-readme-commands
  Summary: Finalize README commands for setup, activation, preprocessing,
    episode generation, dry-run smoke, Isaac SG-Nav, batch runs, summaries, and
    visualization.
  Why it matters now: The final deliverable is user-runnable, not only
    internally testable.
  Preconditions: milestone-006-reporting-visualization
  Parallel hints: Can co-run with tests if docs do not change CLI semantics.
  Boundary notes: Document local path overrides and exact missing-asset behavior.
  Extraction notes: Acceptance should include command copy/paste checks where
    assets exist and clear skip notes where they do not.

- Direction id: direction-021-regression-suite
  Summary: Fill regression coverage for benchmark-mode safety, perception
    contracts, mapping/frontiers, graph scoring, re-perception/STOP, metrics,
    batch summaries, and script entrypoints.
  Why it matters now: The project must remain runnable after future changes.
  Preconditions: milestone-006-reporting-visualization
  Parallel hints: Can co-run with docs only if tests are isolated and do not
    require doc command edits.
  Boundary notes: Prefer deterministic synthetic tests for heavy external
    dependencies and explicit integration skips for Isaac/model assets.
  Extraction notes: Acceptance should include full pytest and smoke command
    evidence.

- Direction id: direction-022-final-benchmark-audit
  Summary: Audit the complete stack against the user hard requirements and
    record final pass/skip evidence.
  Why it matters now: The roadmap is complete only when the repository can be
    evaluated against the original SG-Nav Isaac/InteriorAgent benchmark goal.
  Preconditions: direction-020-readme-commands and direction-021-regression-suite
  Parallel hints: Serial final audit.
  Boundary notes: Do not implement new feature work here unless the audit finds
    a small blocking defect; larger gaps should create a new roadmap revision.
  Extraction notes: Acceptance should include a checklist mapping every hard
    requirement to tests, docs, commands, or explicit external-asset skips.
