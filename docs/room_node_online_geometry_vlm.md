# Online Geometry Room Nodes And VLM Labels

Strict SG-Nav room nodes are built online. They do not use InteriorAgent
`rooms.json`, dataset room labels, or oracle room masks.

## Geometry Stage

`isaac_bench.mapping.room_segmentation.OnlineRoomSegmenter` consumes only the
online RGB-D occupancy/free-space state:

- `observed_free_mask`
- `obstacle_mask`
- `unknown_mask`

It builds a structural free-space mask from observed free cells, suppresses
small clutter only for room segmentation, applies close/open morphology,
segments connected components with distance-transform seeds plus watershed or
deterministic seeded region growing, refines splits with doorway-width
metadata, merges tiny clutter fragments, and preserves stable room IDs by
mask IoU/centroid matching.

Each `RoomMask` records `source=online_geometry_watershed`, area, centroid,
observed cells, boundary unknown fraction, doorway edges, confidence, partial
state, and stable room id.

## Object Affiliation

`assign_objects_to_room_masks` assigns online object-memory nodes to room masks
by centroid containment first, then projected point-cloud or bbox footprint
overlap. Edges carry:

- `edge_type=contains`
- `source=online_geometry_mask_overlap`
- `centroid_inside`
- `mask_overlap_ratio`
- `assignment_confidence`
- `ambiguous`

## Semantic Stage

`isaac_bench.graph.room_semantics.VLMRoomLabeler` asks an OpenAI-compatible
LLM/VLM backend to choose one category from the configured allowed set:

`kitchen, bedroom, bathroom, living_room, dining_room, office, hallway,
corridor, laundry_room, storage_room, entryway, balcony, unknown`.

The prompt uses only room geometry, objects inside the room mask, and optional
online visual evidence. It explicitly requires `unknown` when evidence is weak,
partial, ambiguous, contradictory, or not diagnostic.

Post-processing forces `unknown` for invalid categories, low confidence, weak
evidence, partial weak evidence, ambiguous ranked alternatives, or contradictory
objects. A VLM returning `unknown` from insufficient evidence is valid; a
missing/unreachable VLM backend is not valid for strict metrics.

## Config

The default strict path uses:

- `mapping.room_map_mode: online_geometry_watershed`
- `sgnav.scene_graph.room_nodes.source: online_geometry_watershed_vlm`
- `sgnav.scene_graph.room_nodes.strict_no_oracle_rooms: true`

`rooms.json` remains allowed for episode generation, sanity checks, debug
visualization, and explicitly named `oracle_room_ablation` runs only.
