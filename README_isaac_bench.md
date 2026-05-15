# SG-Nav Isaac/InteriorAgent Benchmark

This branch implements SG-Nav paper mode in Isaac Sim. The original online
open-vocabulary 3D instance segmentation module is replaced by YOLO-World +
SAM2 + RGB-D depth fusion; downstream navigation follows SG-Nav's hierarchical
scene graph, dense edge generation and pruning, object-centered H-CoT subgraph
scoring, frontier interpolation, and graph-based re-perception.

## Setup

Recommended for Isaac 5.1 + YOLO-World + SAM2: use one Python 3.11 mamba env
and source Isaac's conda bridge. The old `/home/echo/SG-Nav/.mamba/envs/sg-nav`
env is Python 3.9, so keep it for legacy Habitat SG-Nav runs; do not use it for
Isaac 5.1.

```bash
./scripts/setup_sgnav_isaac_env.sh
source scripts/activate_sgnav_isaac_env.sh
```

After activation, `python` can import Isaac Sim, YOLO-World, SAM2, and this
benchmark package from the same process. Run commands through:

```bash
./scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.run_one_episode ...
```

The setup script uses Isaac's prebundled `torch 2.7.0+cu128` from
`/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64/setup_conda_env.sh` instead
of installing a second PyTorch wheel into the mamba env.

Legacy SG-Nav Python 3.9 setup is still supported for map-only commands:

```bash
chmod +x run_isaac_bench.sh
./run_isaac_bench.sh -m pip install -r requirements-isaac-bench.txt
```

YOLO-World uses `data/models/yolov8l-worldv2.pt` by default. If the file is not
present, Ultralytics will try to resolve `yolov8l-worldv2.pt` on first load.
SAM2 uses the local checkpoint `data/models/sam2.1_hiera_small.pt` and packaged
config `configs/sam2.1/sam2.1_hiera_s.yaml` by default.

## Preprocess InteriorAgent

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.preprocess_interioragent \
  --dataset-root /home/echo/InteriorAgent \
  --out data/interioragent_preprocessed \
  --resolution 0.05 \
  --inflation-radius-m 0.0 \
  --scene-id kujiale_0031
```

The preprocessing command automatically re-execs with Isaac Kit Python for USD
parsing and writes:

`objects.json`, `objects_all.json`, `rooms.json`, `occupancy.npy`,
`navigable.npy`, `inflated_obstacles.npy`, `map_info.json`, `nav2_map.png`,
`nav2_map.yaml`, and `debug_topdown.png`.

`robot_radius_m` is the only obstacle inflation used for navigability. The
default is now `0.14 m`, i.e. half of the configured `0.28 m` Kaya footprint
width. The legacy `--inflation-radius-m` argument is accepted for old commands
but no longer adds an extra boundary band. Structural doorway categories such as
`doorsill` and `doorway` are treated as passable openings; `door`/`door_panel`
meshes are only skipped when their bbox looks open instead of covering the
opening.

## Generate Episodes

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.generate_episodes \
  --preprocessed-dir data/interioragent_preprocessed \
  --scene-id kujiale_0031 \
  --episodes-per-scene 100 \
  --min-start-object-clearance-m 0.80 \
  --min-start-goal-bbox-distance-m 1.5 \
  --out data/interioragent_episodes/debug.jsonl
```

Episode generation now rejects starts that are too close to InteriorAgent object
bboxes, including walls, doors, windows, and furniture from `objects_all.json`.
Raise `--min-start-object-clearance-m` if Kaya still spawns too near geometry;
lower it only if a very cluttered scene cannot produce enough starts.

## Run A* Benchmark Smoke

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.run_one_episode \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector dry_run \
  --output data/isaac_bench_runs/debug/results.jsonl \
  --debug-map debug/episode.png
```

## Select Longest Instance A*

This scans every object instance independently, finds the valid start with the
largest A* distance to that instance's goal region, then writes a one-episode
JSONL for stress-testing A*.

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.select_longest_instance_astar \
  --preprocessed-dir data/interioragent_preprocessed \
  --scene-id kujiale_0031 \
  --out data/interioragent_episodes/longest_instance_astar.jsonl \
  --report-out data/interioragent_episodes/longest_instance_astar_report.json \
  --debug-map debug/longest_instance_astar.png \
  --min-planning-clearance-m 0.80 \
  --min-goal-clearance-m 0.80
```

## Run Isaac SG-Nav Closed Loop

Run this inside the `sgnav-isaac` Python 3.11 env. The default config runs
`sgnav.mode: paper`: RGB-D observation, online depth occupancy/free-space
mapping, YOLO-World + SAM2 + depth fusion, hierarchical scene graph updates,
object-centered H-CoT subgraph scoring, `sum(P_sub / D)` frontier interpolation,
graph-based re-perception, deterministic A* replanning, and holonomic Kaya
control.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --detector yolo_world \
  --sim-backend isaac \
  --headless true \
  --output data/isaac_bench_runs/sgnav_paper_mode/results.jsonl \
  --debug-map debug/sgnav_paper_mode.png
```

Use `--headless true` for non-GUI runs. A*, frontiers, collision guarding,
YOLO/SAM2 object projection, and the popup map use the live Isaac RGB-D stream.
The online occupancy map now uses depth ray casting: every sampled depth pixel
forms a 2D grid ray from the camera center to the depth endpoint, cells crossed
by that ray are marked free, endpoints inside the robot-height obstacle band are
marked occupied, and cells never crossed by a ray remain unknown. Rays whose
endpoints are above the configured robot obstacle slice are ignored for 2D free
clearing, so ceiling and high-wall pixels do not smear free space across the
map. The old
benchmark-only static near-field fill is disabled by default; enable it only
explicitly with `--static-nearfield-map` if you need a temporary simulator blind
spot patch. Frontiers are reachable free-boundary cells next to unknown space.
Tune this with
`mapping.depth_max_m`,
`mapping.depth_stride_px`, `mapping.obstacle_min_height_m`,
`mapping.obstacle_max_height_m`, `mapping.free_min_height_m`,
`mapping.free_max_height_m`, and
`mapping.map_size_m` in
`isaac_bench/configs/isaac_bench.yaml`. Obstacle inflation uses
`robot.footprint_radius_m` only; the default `0.14 m` is half the robot width,
which is about 3 cells at 0.05 m resolution. Frontier extraction now follows
the original `SG_Nav.py::fbe()` construction: free cells are marked as 1,
obstacles are dilated with a disk radius of 4 cells and marked as 3, unknown
space is dilated with a disk radius of 1 cell, and frontiers are free cells on
that unknown boundary. It does not cluster or project frontier cells to nearby
targets. Tune this with `mapping.frontier_min_distance_m`,
`mapping.frontier_max_count`, `mapping.frontier_obstacle_dilation_radius_cells`,
and `mapping.frontier_unknown_dilation_radius_cells`.

At reset, Kaya performs an SG-Nav-style opening panorama before normal
planning. Each panorama view updates the depth map, YOLO/SAM2 detections,
object memory, and scene graph. Tune it with `sgnav.panorama_steps`,
`sgnav.panorama_wz_radps`, or CLI flags `--panorama-steps` and
`--panorama-wz-radps`. Runtime graph rows now include object, room, group, and
edge counts; the adapter builds object-room, related-object, and group
relationships from Isaac RGB-D detections and the observed online map.

Frontier scoring defaults to the local graph fallback, but you can route it to
an OpenAI-compatible vLLM server. Run vLLM in a separate terminal and its own
`sg-nav-vllm` environment:

```bash
cd /home/echo/sgnav
./scripts/run_vllm_server.sh
```

Then run Isaac/YOLO/SAM2 from the `sgnav-isaac` terminal:

```bash
./scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector yolo_world \
  --sim-backend isaac \
  --vllm-frontier-scoring \
  --vllm-base-url http://127.0.0.1:8000/v1
```

If vLLM is unreachable or returns malformed scores, the run records
`vllm_disabled_reason` and falls back to deterministic graph scoring instead of
crashing. Because the server API accepts images as HTTP payloads, this path
uses a CPU JPEG copy of the latest Isaac camera frame by default. Disable image
context with `--no-vllm-image-scoring`, or tune transfer size with
`--vllm-image-max-width` and `--vllm-image-jpeg-quality`.
When vLLM frontier scoring is enabled, frontiers are scored before the
candidate shortcut as well, so the terminal should print `[sgnav-vllm] POST ...`
even if the high-level policy later decides to move to a detected object
candidate. Disable that probe with `--no-score-frontiers-before-candidate`.
The result JSONL records `vllm_num_requests`, `vllm_last_skip_reason`, and
`vllm_disabled_reason` so it is clear whether the policy entered frontier
scoring.

The Isaac backend currently uses
kinematic holonomic control to keep Kaya from falling through InteriorAgent
scenes whose USD physics hierarchy has nested rigid-body warnings. If Isaac
Python cannot import Ultralytics, `--detector yolo_world` can still fall back to
a small external detector worker, but the recommended `sgnav-isaac` env avoids
that IPC split.
In the recommended unified env, Isaac RGB is read through the CUDA camera
annotator and passed to YOLO-World as a CUDA tensor, so perception frames avoid
the expensive GPU-texture-to-CPU-image transfer. If the detector falls back to
the external IPC worker or the popup needs a display frame, that specific frame
uses a CPU image. Override with `--camera-annotator-device cpu|cuda`.
For a no-model smoke test of the SG-Nav candidate branch, use `--detector
dry_run --seed-gt-object-memory --candidate-min-detector-hits 1`; do not use
that seeded mode for benchmark metrics.
When running with `--headless false`, the active viewport is bound to Kaya's
camera prim `/World/Kaya/camera_rgbd` by default. The default camera is a wider
110 degree HFOV view mounted at 1.35 m over the base center, with explicit
USD camera clipping `near=0.02 m`, `far=80.0 m`; override with
`--camera-hfov-deg`, `--camera-mast-height-m`, `--camera-forward-offset-m`,
`--camera-near-m`, and `--camera-far-m` for debugging.
The SG-Nav debug popup opens automatically when `--headless false` is used. It
runs in a separate SG-Nav-env UI worker so OpenCV GUI calls do not run inside
Isaac's `SimulationApp` process. It shows RGB detections, object memory,
zoomed observed map, frontiers, selected candidate/frontier, A* path, and the
latest scene-graph scores. The default popup is `1440x900`, with most of the
width allocated to the map. Force it with `--sgnav-viz`, disable it with
`--no-sgnav-viz`, and optionally save panels with
`--sgnav-viz-save-dir debug/sgnav_viz`. The popup now refreshes every 20 control
steps by default and saves panels every 10 popup updates to avoid dragging down
Isaac; override with `--sgnav-viz-every-steps` and
`--sgnav-viz-save-every-steps`.
The default Kaya command limits are now intentionally slower:
`max_vx=max_vy=0.15 m/s`, `max_wz=0.35 rad/s`; override them with
`--max-vx-mps`, `--max-vy-mps`, and `--max-wz-radps`.
YOLO detections below confidence `0.7` are filtered before depth backprojection
and object-memory updates; override with `--detector-conf` if needed. When
`perception.segmenter: auto|sam2` is enabled, SAM2 is prompted with YOLO boxes
and the mask pixels, not the whole box, are backprojected into object memory.
YOLO-World runs every 5 control steps by default; raise/lower the rate with
`--perception-every-steps` (`3` is more responsive, `1` is heaviest).

For YOLO-World dependency loading:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.run_one_episode \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector yolo_world \
  --output data/isaac_bench_runs/yolo/results.jsonl
```

The result JSONL records SG-Nav integration fields such as
`object_memory_count`, `goal_candidate_count`, `sgnav_decision_mode`,
`sgnav_decision_reason`, and `scenegraph_backend`.

## Summarize

```bash
./run_isaac_bench.sh -m isaac_bench.metrics.summarize \
  --input data/isaac_bench_runs/debug/results.jsonl
```

## Isaac Smoke Test

```bash
/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  -m isaac_bench.env.isaac_process \
  --scene-usd /home/echo/InteriorAgent/kujiale_0031/kujiale_0031.usda \
  --spawn 2.775 1.125 0.05 -2.4799373265 \
  --headless true \
  --save-frame debug/isaac_frame.png
```

Nav2 is intentionally a placeholder in this first pass. `planner=nav2` raises a
clear ROS2/Nav2 unavailable error until ROS2 and Nav2 are installed.

The original SG-Nav `SceneGraph` can be exercised with
`--use-original-scenegraph`. The default path keeps this disabled so the
benchmark remains runnable without loading GLIP/GroundingDINO/VLLM dependencies;
the adapter synchronizes Isaac object memory into the same scene-graph node
shape and falls back cleanly if the original stack is unavailable.
