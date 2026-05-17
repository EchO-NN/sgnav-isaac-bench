# Online Geometry Room Nodes And VLM Labels

Strict SG-Nav room nodes are built online. They do not use InteriorAgent
`rooms.json`, dataset room labels, or oracle room masks.

Room context is lazy and frontier-scoring gated. The runtime calls
`prepare_room_context_for_frontier_scoring` only after reachable frontiers have
been extracted and immediately before SG-Nav scene-graph/HCoT/subgraph frontier
scoring. Mapper updates, perception updates, committed-frontier A* replans, and
candidate re-perception/STOP confirmation reuse cached graph context and do not
trigger room segmentation or VLM room labeling.

## Geometry Stage

`isaac_bench.mapping.room_segmentation.OnlineRoomSegmenter` consumes only the
online RGB-D occupancy/free-space state:

- `observed_free_mask`
- `obstacle_mask`
- `unknown_mask`

It first builds a structural obstacle/free-space mask from observed free cells.
Furniture and movable-object clutter are suppressed before room segmentation:
compact depth blobs and object-memory detections such as chairs, sofas, tables,
beds, TVs, plants, pictures, lamps, and cabinets are not allowed to split
rooms. Unknown cells are not treated as structural wall support.

Distance-transform watershed or deterministic seeded region growing is used
only to create premerge room proposals. Final room masks come from a
doorway-constrained merge pass:

- verified structural walls, doorways, or gateways preserve a physical split
  regardless of room type;
- weak/open-plan proposal boundaries preserve a functional split only when
  premerge room recognition returns reliable, non-unknown, different room
  categories on both sides;
- same-category proposals, unknown proposals, unreliable labels, furniture
  boundaries, and unknown-supported boundaries merge into one physical
  room/zone mask.

Objects such as sinks, fridges, sofas, or TVs are only evidence for room
recognition. They are not direct hardcoded split rules. Premerge labels are used
only for merge/split decisions; final room masks are labeled again before they
become SG-Nav room nodes. Stable room IDs are then preserved by mask
IoU/centroid matching.

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
partial, ambiguous, contradictory, or not diagnostic. The VLM-returned
`confidence` is recorded as `vlm_self_confidence`; it is a self-reported ordinal
signal, not a calibrated probability.

Post-processing forces `unknown` for invalid categories, low confidence, weak
evidence, partial weak evidence, ambiguous ranked alternatives, or contradictory
objects. A VLM returning `unknown` from insufficient evidence is valid; a
missing/unreachable VLM backend is not valid for strict metrics.

Accepted non-unknown room labels also expose `label_reliability` and
`reliability_factors`, derived from diagnostic object hits, reliable object
count, visual evidence, partial-room state, boundary unknown fraction, ambiguity
margin, mask confidence, and contradictory evidence checks. `RoomNode.confidence`
uses this evidence reliability rather than raw VLM confidence.

For open-plan functional split decisions, the default reliability gate is
`room_semantics.min_label_reliability_for_functional_split: 0.65`. `unknown`
never creates a functional split unless a structural boundary already preserves
the split.

## Config

The default strict path uses:

- `mapping.room_map_mode: online_geometry_watershed`
- `room_semantics.use_premerge_labels_for_open_plan_merge: true`
- `room_semantics.min_label_reliability_for_functional_split: 0.65`
- `room_semantics.unknown_allows_functional_split: false`
- `room_semantics.final_label_after_merge: true`
- `sgnav.scene_graph.room_nodes.source: online_geometry_watershed_vlm`
- `sgnav.scene_graph.room_nodes.strict_no_oracle_rooms: true`

`rooms.json` remains allowed for episode generation, sanity checks, debug
visualization, and explicitly named `oracle_room_ablation` runs only.
