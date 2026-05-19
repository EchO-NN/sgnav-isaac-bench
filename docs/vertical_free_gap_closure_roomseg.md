# Vertical-Free Gap-Closure Room Segmentation

`vertical_free_gap_closure_v1` is the strict default room geometry backend.
It consumes the online 0.2-2.0 m vertical-free roomseg map plus the non-free
wall map produced from RGB-D mapping. It does not call upstream ROSE2,
source-form ROSE2, watershed, `rooms.json`, or VLMs for geometry.

The algorithm closes only short, verified wall gaps with virtual room
boundaries:

- candidate endpoints come from wall skeleton endpoints and high-curvature
  wall boundary points;
- candidates must be within `close_max_gap_m` and aligned with local wall
  tangents;
- the line between endpoints must be mostly vertical-free and low-unknown;
- both sides of the gap must have free-space support;
- topology verification keeps a closure only when it separates two meaningful
  free-space components.

Final room masks are connected components of:

```text
vertical_free_room_domain & ~virtual_room_boundary
```

The virtual boundary is used only for room labeling. It is not written into the
planner obstacle map, so navigation and frontier extraction continue to use the
normal occupancy/free-space maps.

Replay a saved roomseg/source NPZ:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.replay_vertical_free_gap_closure_roomseg \
  --input debug/roomseg_layers/roomseg_step_000000.npz \
  --out-dir debug/vertical_free_gap_closure/replay
```

Inspect a replay or live debug artifact:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.inspect_vertical_free_gap_closure_roomseg \
  --input debug/vertical_free_gap_closure/replay/vfgc_step_000000.npz \
  --print-summary
```
