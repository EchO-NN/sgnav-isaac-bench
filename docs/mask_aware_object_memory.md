# Mask-Aware YOLO-World Object Memory

Metric perception keeps YOLO-World plus SAM2 as the real path. Detections with
`confidence <= 0.55` are not valid detections: they must not draw RGB bboxes,
enter SAM2/depth fusion, create object-memory nodes, create scene-graph object
nodes, seed candidate goals, or support STOP.

Edge-touching or partially visible detections are different. They are raw
evidence, not automatic rejects. The object-memory registry logs them with
visibility metadata and then uses mask/footprint overlap to decide whether they
support an existing stable track.

## Track Rules

- Raw detections are always logged for debugging and GNN data extraction.
- Track category is accumulated with `class_conf_sums` and `class_hits`; the
  stable category is the confidence-sum winner.
- A partial/edge detection that overlaps a prior stable full-mask track
  inherits stable geometry and cannot overwrite that center with the visible
  fragment.
- A partial/edge detection without stable association remains tentative and is
  not eligible for policy graph, room labels, goal candidates, or STOP.
- Small masks contained inside larger masks with different categories create a
  child relation such as `pillow supported_by sofa`; they do not overwrite the
  parent category.

## Exported Fields

Object-memory debug/GNN snapshots expose raw detections and object tracks with:

- `raw_category`, `confidence`, `bbox_xyxy`, `bbox_touches_edge`,
  `mask_touches_edge`, `visibility_status`, and `associated_track_id`
- `category`, `stable_category`, `mean_confidence`, `detection_count`,
  `winner_detection_count`, `valid_detection_count`
- `used_for_policy_graph`, `used_for_room_label`,
  `used_for_goal_candidate`, and `used_for_stop`
- parent/child relations for containment cases

