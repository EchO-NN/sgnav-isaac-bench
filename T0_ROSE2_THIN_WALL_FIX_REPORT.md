# T0 ROSE2 Thin Wall Fix Report

## 1. Changed Files

- `isaac_bench/mapping/rose2_source_form.py`
- `isaac_bench/mapping/rose2_separator_detection.py`
- `isaac_bench/mapping/rose2_partition_graph.py`
- `isaac_bench/mapping/rose2_source_external_runner.py`
- `isaac_bench/mapping/upstream_rose2_pure_python_adapter.py`
- `isaac_bench/mapping/room_segmentation_debug.py`
- `isaac_bench/scripts/replay_roomseg_from_debug.py`
- `isaac_bench/scripts/run_one_episode.py`
- `isaac_bench/scripts/run_benchmark.py`
- `isaac_bench/metrics/result_schema.py`
- `isaac_bench/graph/room_context.py`
- `isaac_bench/configs/isaac_bench.yaml`
- `isaac_bench/env/isaac_process.py`
- `isaac_bench/tests/test_isaac_kinematic_pose.py`
- `isaac_bench/tests/test_rose2_thin_wall_split.py`
- Existing roomseg/config/schema tests updated for `rose2_source_form_v2`.

## 2. Vertical-Free Contract

The 0.2--2.0 m vertical-free input logic is unchanged:

- `vertical_or_free_z_min_m = 0.20`
- `vertical_or_free_z_max_m = 2.00`
- `vertical_or_free_min_free_rays = 1`
- `vertical_or_free_min_observed_rays = 1`

Strict room segmentation still uses the vertical profile only. Navigation free is not overlaid into strict room segmentation:

- `roomseg_input_source = vertical_profile_only`
- `navigation_free_added_to_strict_roomseg_cells = 0`

The adapter now asserts this invariant at runtime.

## 3. Step 000496 Before/After

Before, the saved source-form debug for the full Isaac run reported one room at `debug/t0_rose2_source_full_episode/rose2_source/rose2_source_step_000496.summary.json`.

After replaying the same NPZ through `rose2_source_form_v2`:

- `source_room_count = 2`
- `proposal_room_count = 2`
- `accepted_topology_separator_count = 2`
- `rejected_separator_count = 2`
- `legacy_connected_component_rooms_used = false`

Replay output:

- `debug/replay_step_000496_v2_final/rose2_source_step_000000.summary.json`
- `debug/replay_step_000496_v2_final/rose2_source_step_000000.room_labels.png`
- `debug/replay_step_000496_v2_final/rose2_source_step_000000.partition_boundary.png`
- `debug/replay_step_000496_v2_final/rose2_source_step_000000.accepted_separators.png`
- `debug/replay_step_000496_v2_final/rose2_source_step_000000.input_masks.npz`

PNG-only replay, useful when only a vertical-free screenshot is available:

- `source_room_count = 5`
- `accepted_topology_separator_count = 5`
- Output: `debug/replay_step_000496_v2_png_after_axis/`

## 4. Thin Wall Separators

New module: `isaac_bench/mapping/rose2_separator_detection.py`.

Toy regression confirms thin-wall promotion:

- 1-cell interior wall:
  - `thin_wall_separator_count >= 1`
  - `accepted_topology_separator_count >= 1`
  - `source_room_count >= 2`
- Screenshot-like synthetic layout:
  - `thin_wall_separator_count >= 1`
  - `accepted_topology_separator_count >= 1`
  - `proposal_room_count >= 2`
  - `final_room_count_before_policy_merge >= 2`

For the old NPZ step 000496, the split is recovered by ROSE/Hough representative wall evidence rather than the thin-wall promoter, because the saved masks already expose enough wall structure. The PNG-only replay recovers multiple topology separators from the image-derived wall structure.

## 5. Doorway Partition Cuts

New module logic: `generate_doorway_partition_cuts(...)` in `isaac_bench/mapping/rose2_partition_graph.py`.

Regression behavior:

- Navigation free remains connected through the doorway.
- Room labels split across the doorway.
- Doorway cuts are virtual room boundaries only; they are not written into the navigation obstacle map.

## 6. Proposal And Final Room Counts

Strict default backend is now:

- `room_segmentation.backend = rose2_source_form_v2`
- `room_map_mode = rose2_source_form_v2_vlm`

Default finalization remains guarded:

- `finalization_mode = no_merge_until_source_backend_verified`
- `merge_guard_enabled = true`

The default path does not merge separated proposals back into one room.

## 7. Legacy / Source-Form-v2 / External

- `legacy_rose2_style_debug` remains debug/ablation-only and is rejected in strict mode unless explicitly configured as an ablation/debug path.
- `rose2_source_form_v2` is the strict no-ROS default.
- `rose2_source_external_runner` now creates a real wrapper bundle and subprocess invocation instead of only exporting and raising.

External runner evidence:

- Command produced `debug/rose2_external_step_000496/external_summary.json`
- It exported `external_source_input.metric_map.png`, `external_source_input.npz`, `run_external_rose2.py`, `stdout.txt`, and `stderr.txt`.
- Current local upstream run fails reproducibly because the upstream environment is missing Python package `png`; this is recorded in `stderr.txt`.
- No source-form fallback is used by the external runner.

## 8. Toy Tests

Added `isaac_bench/tests/test_rose2_thin_wall_split.py`:

- thin 1-cell wall splits two rooms
- wall with doorway keeps navigation free connected but room labels split
- open plan remains one room
- furniture-like blob does not split room
- screenshot-like thin-wall layout produces multiple rooms
- external runner writes reproducible failure report

## 9. Verification

Commands run:

```bash
./run_isaac_bench.sh -m pytest -q
```

Result:

```text
309 passed in 3.41s
```

```bash
./run_isaac_bench.sh -m isaac_bench.mapping.rose2_source_tests \
  --backend rose2_source_form_v2 \
  --out-dir debug/rose2_source_v2_t0_tests
```

Result:

```text
PASS two_rooms_door rooms=2
PASS three_rooms_corridor rooms=4
PASS open_plan_clutter rooms=1
```

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.replay_roomseg_from_debug \
  --input debug/t0_rose2_source_full_episode/rose2_source/rose2_source_step_000496.npz \
  --out-dir debug/replay_step_000496_v2_final \
  --backend rose2_source_form_v2 \
  --dump-layers
```

Result:

```text
source_room_count=2
proposal_room_count=2
accepted_topology_separator_count=2
```

## 10. Visible Isaac Episode And Kaya Kinematic Fix

A visible Isaac run was started with:

```bash
ROSE2_SOURCE_ROOT=/home/echo/declutter-reconstruct ./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector grounding_dino \
  --grounding-dino-checkpoint data/models/groundingdino_swinb_cogcoor.pth \
  --grounding-dino-config data/models/GroundingDINO_SwinB.cfg.py \
  --grounding-dino-text-threshold 0.25 \
  --detector-conf 0.45 \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless false \
  --strict-benchmark \
  --llm-enabled \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model qwen3-vl-8b-instruct \
  --room-map-mode rose2_source_form_v2_vlm \
  --roomseg-backend rose2_source_form_v2 \
  --roomseg-finalization-mode no_merge_until_source_backend_verified \
  --debug-rose2-source \
  --rose2-source-work-dir debug/t0_thin_wall_full_episode/rose2_source \
  --debug-roomseg-layers \
  --debug-roomseg-dir debug/t0_thin_wall_full_episode/roomseg_layers \
  --sgnav-viz \
  --sgnav-viz-save-dir debug/t0_thin_wall_full_episode/panels \
  --sgnav-viz-save-every-steps 5 \
  --output data/isaac_bench_runs/t0_thin_wall_full_episode/results.jsonl \
  --debug-map debug/t0_thin_wall_full_episode/final_map.png
```

The first run produced visible panel snapshots and ROSE2 source dumps before
Isaac physics became unstable:

- `debug/t0_thin_wall_full_episode/panels/` contains visible SG-Nav panel frames.
- `debug/t0_thin_wall_full_episode/rose2_source/rose2_source_step_000104.summary.json`
  reports `backend=rose2_source_form_v2`, `source_room_count=2`,
  `proposal_room_count=2`, `accepted_topology_separator_count=1`, and
  `legacy_connected_component_rooms_used=false`.
- The episode did not write a final `results.jsonl` because Isaac emitted
  repeated `Invalid PhysX transform detected for /World/Kaya/base_link` and
  `Illegal BroadPhaseUpdateData` errors.

This was fixed by changing kinematic closed-loop stepping so it no longer calls
`set_world_pose` on the PhysX Kaya articulation every frame. Kinematic stepping
now updates the internal SG-Nav pose and camera poses while leaving the Kaya
articulation at its reset pose, preventing PhysX broadphase corruption.

Regression evidence:

```bash
./run_isaac_bench.sh -m pytest -q isaac_bench/tests/test_isaac_kinematic_pose.py isaac_bench/tests/test_waypoint_follower.py
```

Result:

```text
3 passed
```

Isaac kinematic server smoke:

- 180 kinematic steps on `kujiale_0031`
- Log: `debug/isaac_kinematic_guard/smoke.log`
- Check result: no `Invalid PhysX transform`, no `Illegal BroadPhaseUpdateData`

`run_one_episode` closed-loop smoke:

- Command log: `debug/isaac_kinematic_guard/run_one_episode_smoke.log`
- Result row: `data/isaac_bench_runs/kinematic_guard_smoke/results.jsonl`
- Check result: no `Invalid PhysX transform`, no `Illegal BroadPhaseUpdateData`,
  no traceback.

## 11. Remaining External Case

The only remaining issue is external upstream execution, not the strict default backend:

- `rose2_source_external_runner` launches the wrapper and records all inputs/logs.
- Local upstream dependency `png` is missing, so exact external rooms are not produced in this environment.
- Strict benchmark default does not depend on this external runner.
