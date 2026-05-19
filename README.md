# SG-Nav Isaac / InteriorAgent Benchmark

This repository wraps SG-Nav ObjectNav evaluation on Isaac Sim and
InteriorAgent scenes. The metric path is intentionally strict: Isaac RGB-D,
GroundingDINO-B/Swin-B detections, SAM2 masks, mask/depth object memory, online mapping,
scene-graph subgraph reasoning, frontier interpolation, candidate
re-perception, STOP confirmation, and deterministic A*/FMM-equivalent local
planning.

See also:

- `docs/sgnav_paper_parity_contract.md`
- `docs/benchmark_metric_validity_contract.md`
- `docs/sgnav_decision_dump_contract.md`
- `docs/external_assets.md`
- `docs/room_node_online_geometry_vlm.md`
- `docs/vertical_profile_roomseg.md`
- `docs/sgnav_paper_mechanism_audit.md`

## Setup

External assets are not vendored. By default the project expects:

- Isaac Sim: `/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64`
- InteriorAgent: `/home/echo/InteriorAgent`
- GroundingDINO-B/Swin-B: `data/models/groundingdino_swinb_cogcoor.pth`
- SAM2: `data/models/sam2.1_hiera_small.pt`
- Optional ROSE2 source for debug/ablation only: `ROSE2_SOURCE_ROOT` pointing
  at `goldleaf3i/declutter-reconstruct`
- Optional OpenAI-compatible LLM endpoint: `http://127.0.0.1:8000/v1`

Override paths with `ISAAC_SIM_ROOT`, `INTERIORAGENT_ROOT`,
`GROUNDING_DINO_CHECKPOINT`, `GROUNDING_DINO_CONFIG`, `GROUNDING_DINO_ROOT`,
`SAM2_CHECKPOINT`, `SAM2_MODEL_CFG`, optional `ROSE2_SOURCE_ROOT`, and
`LLM_BASE_URL`.
`ISAAC_ROOT` remains a legacy alias for `ISAAC_SIM_ROOT`.

```bash
./scripts/setup_sgnav_isaac_env.sh
source ./scripts/activate_sgnav_isaac_env.sh
```

Check required assets:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.download_grounding_dino

python -m isaac_bench.scripts.check_assets \
  --require-grounding-dino \
  --require-sam2 \
  --require-interioragent \
  --require-isaac
```

Add `--require-rose2-source` only for upstream ROSE2 debug/ablation runs; the
strict default `vertical_free_gap_closure_v1` room segmentation does not need
that source checkout.

## Preprocess And Episodes

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.preprocess_interioragent \
  --dataset-root ${INTERIORAGENT_ROOT:-/home/echo/InteriorAgent} \
  --out data/interioragent_preprocessed \
  --resolution 0.05 \
  --scene-id kujiale_0031
```

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.generate_episodes \
  --preprocessed-dir data/interioragent_preprocessed \
  --scene-id kujiale_0031 \
  --episodes-per-scene 10 \
  --out data/interioragent_episodes/debug.jsonl
```

## Smoke Run

This command is for smoke/debug only. It must produce `metric_valid=false`.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector dry_run \
  --sim-backend map \
  --allow-debug-fallbacks \
  --output data/isaac_bench_runs/dry_run_smoke/results.jsonl
```

## Strict Isaac SG-Nav

Strict SG-Nav requires GroundingDINO-B/Swin-B and SAM2. A row is metric-valid only when no
debug fallback is used. Local deterministic LLM scoring is non-metric unless a
named ablation is explicitly declared; configure a real OpenAI-compatible LLM
for metric SG-Nav scoring. The default config uses `llm.enabled: true`,
`mapping.room_map_mode: vertical_free_gap_closure_v1_vlm`,
`mapping.room_segmentation.algorithm: vertical_free_gap_closure_v1`,
`mapping.room_segmentation.backend: vertical_free_gap_closure_v1`,
`mapping.room_segmentation.require_upstream_source_for_strict: false`,
`sgnav.scene_graph.room_nodes.source: vertical_free_gap_closure_v1_vlm`,
`mapping.frontier_min_distance_m: 1.0`, and
`sgnav.frontier_distance_weight: 0.2`.

Room nodes in strict SG-Nav come from pure Python vertical-free gap closure over
the online 0.2--2.0 m vertical-free roomseg map: short wall gaps are closed only
as virtual room boundaries after endpoint, side-support, and topology checks,
then rooms are connected components of `vertical_free & ~virtual_boundary`.
VLM labels are applied over objects inside those masks. This path does not require
`ROSE2_SOURCE_ROOT`, does not call upstream ROSE2, and does not fall back to
local ROSE2-lite, watershed, or `rooms.json`. Virtual room boundaries are not
written into the planner obstacle map. `unknown` is the correct room
label when the evidence is insufficient. `rooms.json` labels are rejected in
strict metric mode except for a named `oracle_room_ablation`. Upstream ROSE2,
local ROSE2-lite, and old watershed room segmentation are kept only for debug
or explicit ablations.
Premerge proposal labels are used only to decide weak/open-plan functional
splits. Verified structural walls preserve splits regardless of room labels.
Open-plan proposal boundaries preserve a functional split only when both
premerge labels are reliable, non-unknown, and different; same, unknown, or
unreliable labels merge. Final merged room masks are labeled again before they
become SG-Nav room nodes. Objects such as sinks, fridges, sofas, or TVs are
room-recognition evidence, not direct hardcoded split rules.

Room segmentation and room recognition are lazy and scoring-gated: they run
only immediately before a new SG-Nav frontier-scoring decision, after reachable
frontiers are extracted and before scene-graph/HCoT scoring. They do not run
for mapper-only updates, perception-only updates, committed-frontier local A*
replans, or candidate re-perception/STOP confirmation. Result rows expose
`room_update_invoked_for_frontier_scoring`, `room_segmentation_called_for`,
`room_segmentation_algorithm`, `room_segmentation_ran`, `room_labeling_ran`,
`room_context_cache_hit`, and `room_call_order_trace`.

Room VLM `confidence` is stored as `vlm_self_confidence`, a self-reported weak
signal rather than a calibrated probability. Room node confidence uses
evidence-derived `label_reliability`; weak, partial, ambiguous, contradictory,
or non-diagnostic evidence is labeled `unknown`.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --llm-enabled true \
  --llm-base-url ${LLM_BASE_URL:-http://127.0.0.1:8000/v1} \
  --room-map-mode vertical_free_gap_closure_v1_vlm \
  --output data/isaac_bench_runs/final_strict_smoke/results.jsonl \
  --debug-map debug/final_strict_smoke.png
```

For a headed Isaac window with saved SG-Nav visualization panels:

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless false \
  --strict-benchmark true \
  --llm-enabled true \
  --llm-base-url ${LLM_BASE_URL:-http://127.0.0.1:8000/v1} \
  --room-map-mode vertical_free_gap_closure_v1_vlm \
  --sgnav-viz \
  --sgnav-viz-save-dir debug/full_llm_episode_viz \
  --debug-graph-dump \
  --debug-graph-dump-dir debug/full_llm_episode_graphs \
  --output data/isaac_bench_runs/full_llm_episode/results.jsonl
```

For a short integration check on machines with Isaac/model assets:

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --room-map-mode vertical_free_gap_closure_v1_vlm \
  --max-control-steps 1 \
  --panorama-steps 0 \
  --output data/isaac_bench_runs/short_isaac_smoke/results.jsonl
```

## Batch And Summary

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_benchmark \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --output-dir data/isaac_bench_runs/sgnav_grounding_dino_sam2_debug
```

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.metrics.summarize \
  --input data/isaac_bench_runs/sgnav_grounding_dino_sam2_debug/results.jsonl \
  --out data/isaac_bench_runs/sgnav_grounding_dino_sam2_debug/summary.json
```

## Debug Artifacts

Enable saved panels and graph dumps:

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --sgnav-viz-save-dir debug/sgnav_panels \
  --debug-graph-dump \
  --debug-graph-dump-dir debug/graphs \
  --output data/isaac_bench_runs/debug_panels/results.jsonl
```

Each saved SG-Nav panel can also write
`sgnav_step_XXXXXX.layers.json`. That sidecar records primitive counts for
frontier cells, online room masks/boundaries/labels, object nodes, accepted
candidates, GT goal cells, and other overlay layers. Room masks are drawn from
online geometry room segmentation and VLM room labels by default; use
`--no-show-room-masks` or `--no-show-room-labels` only when debugging panel
clutter. GT goal cells are disabled by default so oracle goal markers do not
appear in strict visualization.

GroundingDINO/SAM detections are treated as valid only when `confidence > 0.45`. Lower
or equal detections are filtered before bbox rendering, mask/depth fusion,
object memory insertion, scene-graph object nodes, and goal-candidate logic.
GroundingDINO boxes or SAM2 masks touching an image edge are no longer discarded. They
are logged as `visibility_status=partial_edge`, associated to prior full/stable
tracks by SAM2 mask or 3D footprint overlap when possible, and otherwise kept
as tentative raw evidence. Tentative partial tracks do not enter room evidence,
goal candidates, STOP, or SG-Nav policy graph objects until they become stable.
Object tracks use accumulated `class_conf_sums` and `class_hits`; the stable
category is the accumulated winner and graph/debug dumps expose
`mean_confidence`, `detection_count`, `winner_detection_count`, visibility
counts, geometry confidence, and parent/child track ids.

Convert one graph step into the SG-Nav decision dump contract:

```bash
python -m isaac_bench.scripts.dump_sgnav_step \
  --output debug/sgnav_step.json \
  --graph-debug-dump debug/graphs/graph_step_000000.json \
  --result-row data/isaac_bench_runs/debug_panels/results.jsonl \
  --pretty
```

## Tests

After activating the environment:

```bash
pytest -q
```

From the repo-local lightweight environment:

```bash
./run_isaac_bench.sh -m pytest -q
```

Generated data, logs, maps, videos, and benchmark outputs belong under ignored
`data/` or `debug/` paths.
