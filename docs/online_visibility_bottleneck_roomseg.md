# Online Visibility-Bottleneck Room Segmentation

This repository now includes a CPU-compatible online room segmentation backend:

```python
from src.room_segmentation import GridSpec, OnlineRoomSegmenter

segmenter = OnlineRoomSegmenter(grid_spec=GridSpec(0.05, (0.0, 0.0), width, height))
output = segmenter.update_from_masks(observed_free_mask, obstacle_mask, unknown_mask, frame_id=1)
```

The depth API is also available as `OnlineRoomSegmenter.update(depth, K, T_wc, frame_id, ...)`.

For Isaac integration use:

```bash
--room-map-mode online_visibility_bottleneck_roomseg_v1_vlm \
--roomseg-backend online_visibility_bottleneck_roomseg_v1
```

The backend outputs stable room IDs, confidence maps, soft masks, corridor and open-space masks, functional-zone labels, room graph edges, separator candidates, and debug layers. Unknown cells are never used as hard room separators; they only affect distance, visibility stopping, and confidence.

Run the focused tests with:

```bash
./run_isaac_bench.sh -m pytest isaac_bench/tests/room_segmentation
```

