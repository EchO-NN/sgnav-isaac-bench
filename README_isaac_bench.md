# SG-Nav Isaac/InteriorAgent Benchmark

This directory is a standalone first-pass Isaac benchmark wrapper. It references
the original SG-Nav checkout at `/home/echo/SG-Nav` without modifying it.

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

YOLO-World uses `data/models/yolov8s-worldv2.pt` by default. If the file is not
present, Ultralytics will try to resolve `yolov8s-worldv2.pt` on first load.

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

`robot_radius_m` is the only default obstacle inflation used for navigability;
the extra `--inflation-radius-m` defaults to `0.0`. Door-like categories
(`door`, `doorsill`, `door_handle`) are treated as passable for occupancy and
planning clearance so doorway cells remain usable.

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

Run this inside the `sgnav-isaac` Python 3.11 env. This is the full Isaac SG-Nav
loop: RGB-D observation, online depth occupancy/free-space mapping,
YOLO-World/dry detector, optional SAM2 masks, object memory, SG-Nav scene graph
scoring, candidate/frontier/STOP policy, A* replanning, and holonomic Kaya
control.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector yolo_world \
  --sim-backend isaac \
  --headless false \
  --output data/isaac_bench_runs/isaac_closed_loop/results.jsonl \
  --debug-map debug/isaac_closed_loop.png
```

Use `--headless true` for non-GUI runs. The Isaac backend now keeps the
preprocessed static map only for benchmark GT distance/SPL reporting. A*,
frontiers, collision guarding, YOLO/SAM2 object projection, and the popup map use
the live Isaac depth image instead: depth points mark obstacles by height, rays
mark observed free cells, and frontiers are reachable observed-free boundary
cells next to unknown space. Tune this with `mapping.depth_max_m`,
`mapping.depth_stride_px`, `mapping.obstacle_min_height_m`,
`mapping.obstacle_max_height_m`, and `mapping.map_size_m` in
`isaac_bench/configs/isaac_bench.yaml`.

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
observed map, frontiers, selected candidate/frontier, A* path, and the latest
scene-graph scores. Force it with `--sgnav-viz`, disable it with
`--no-sgnav-viz`, and optionally save panels with
`--sgnav-viz-save-dir debug/sgnav_viz`. The popup now refreshes every 20 control
steps by default and saves panels every 10 popup updates to avoid dragging down
Isaac; override with `--sgnav-viz-every-steps` and
`--sgnav-viz-save-every-steps`.
The default Kaya command limits are now intentionally slower:
`max_vx=max_vy=0.15 m/s`, `max_wz=0.35 rad/s`; override them with
`--max-vx-mps`, `--max-vy-mps`, and `--max-wz-radps`.
YOLO detections below confidence `0.5` are filtered before depth backprojection
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
