# Vertical-Free Geodesic RoomSeg

`vertical_free_geodesic_watershed_v1` is the previous vertical-free
room-segmentation backend. The strict default is now
`vertical_free_gap_closure_v1`; this geodesic watershed backend remains
available only as a debug/ablation path.

It segments rooms directly from the dedicated 0.2-2.0 m vertical-free map:

- cells with vertical-free evidence form the only room-label domain;
- observed non-free cells are treated as wall-like for the distance transform;
- unknown cells are never labeled;
- virtual doorway/bottleneck boundaries split room labels only and are not
  written back into the navigation obstacle map.

The implementation is pure Python/Numpy/Scipy in
`isaac_bench/mapping/vertical_free_roomseg.py`. It does not call ROSE2,
`goldleaf3i/declutter-reconstruct`, ROS, VLM labels, or `rooms.json` for room
geometry. Older ROSE2/source-form paths remain available only by explicitly
selecting their backend as debug/ablation paths.

Replay a saved room-segmentation input:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.replay_vertical_free_roomseg \
  --input debug/rose2_source/rose2_source_step_000496.input_masks.npz \
  --out-dir debug/vfgw_step_000496 \
  --config isaac_bench/configs/isaac_bench.yaml
```

Inspect an output artifact:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.inspect_vertical_free_roomseg \
  --input debug/vfgw_step_000496/vertical_free_step_000496.npz \
  --print-summary
```
