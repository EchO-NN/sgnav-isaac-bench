# No-ROS Upstream ROSE2 Room Segmentation

Strict SG-Nav room segmentation uses
`isaac_bench.mapping.upstream_rose2_pure_python_adapter.UpstreamROSE2PurePythonSegmenter`.
It is a pure-Python, no-ROS adapter for the MIT
`goldleaf3i/declutter-reconstruct` source contract. It must never import or
require `rospy`, `nav_msgs`, `jsk_recognition_msgs`, `catkin`, or `roslaunch`.

## Strict Contract

The strict default config is:

```yaml
mapping:
  room_map_mode: upstream_rose2_vertical_or_free
  strict_no_oracle_rooms: true
  room_segmentation:
    algorithm: upstream_rose2_vertical_or_free
    vertical_or_free:
      enabled: true
      z_min_m: 0.20
      z_max_m: 2.00
      min_free_rays: 1
      min_observed_rays: 1
    source_mode: declutter_reconstruct_mit
    legacy_watershed_allowed: debug_only
    local_rose2_lite_allowed: debug_only
    require_upstream_source_for_strict: true
    upstream_repo_env: ROSE2_SOURCE_ROOT
    run_only_before_frontier_scoring: true
```

Set `ROSE2_SOURCE_ROOT` to a checkout containing:

- `code/FFT_MQ.py`
- `code/minibatch.py`
- `code/parameters.py`

If this source root is missing in strict benchmark mode, the command fails
clearly before frontier scoring. It must not fall back to local ROSE2-lite,
watershed, `rooms.json`, or oracle rooms.

## Runtime Placement

Room segmentation runs only inside
`prepare_room_context_for_frontier_scoring(...)`, after reachable frontiers are
extracted and before scene graph/HCoT frontier scoring. Mapper updates,
perception-only updates, committed A* replans, and candidate re-perception/STOP
checks reuse cached context and do not invoke segmentation or room VLM.

## Debug Artifacts

When room debug dumping is enabled, artifacts are written as:

```text
debug/roomseg/<episode>/<step>_rose2_rooms.png
debug/roomseg/<episode>/<step>_rose2_layers.json
```

The JSON includes `vertical_or_free_map`, `repaired_roomseg_occupied`,
`repaired_window_gaps`, `verified_doorway_gaps`, `algorithm`, `source_mode`,
`num_rooms`, `num_wall_lines`, `main_directions`, `room_masks`,
`room_labels`, `source_repository`, `source_provenance`, and
`strict_fallback_used`. `source_provenance` records the required upstream
source files and SHA256 hashes from the configured source root.
