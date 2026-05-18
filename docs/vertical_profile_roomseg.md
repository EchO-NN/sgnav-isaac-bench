# Vertical-Profile-Assisted ROSE2 Room Segmentation

Strict room segmentation uses no-ROS upstream ROSE2 plus a vertical profile
filter before structure extraction. The key rule is that vertical free evidence
between `0.10 m` and `2.00 m` can suppress furniture/clutter, but it cannot by
itself create a room connection or doorway.

## Height Bands

The online mapper maintains per-cell counts for:

- `low`: `0.10-0.35 m`
- `robot_body`: `0.35-0.80 m`
- `mid`: `0.80-1.30 m`
- `upper`: `1.30-2.00 m`

Each band records `occupied_count`, `free_ray_count`, `observed_count`, and
`unknown_count`.

## Structural Maps

Before ROSE2 extraction, the adapter builds:

- `vertical_carved_map`: a furniture-suppressed candidate structural map.
- `wall_confidence_map`: a float confidence map used to select structural
  walls.
- `structural_wall_mask`: cells above `wall_confidence_threshold`.

Positive wall evidence includes vertical occupancy continuity, long/thin line
support, persistence, and wall-network/perimeter support. Negative evidence
includes furniture object overlap, bulky isolated geometry, surrounding free
space, low observations, and vertical free evidence inside the column.

Furniture such as cabinets, shelves, curtains, wardrobes, refrigerators, sofas,
beds, and tables may remain planner obstacles and object nodes, but they are
not structural room boundaries unless wall confidence and wall-network support
are strong.

## Window And Door Repair

After structural line extraction, gaps are classified:

- Doorway/portal: floor traversability, side free-space support, wall support
  on both sides, doorway-width bounds, low unknown ratio, and non-exterior
  placement. Doorway gaps are recorded as verified portals and used as virtual
  room boundaries without making windows into adjacency.
- Window/non-traversable gap: high-band gap, depth absence, exterior/perimeter
  placement, floor not traversable, or window/curtain/glass object overlap.
  These gaps are closed as wall-like for room segmentation.

Windows can become scene-graph object nodes, but they do not create room
adjacency or open-plan room merges unless floor-level traversability proves a
doorway.

## Debug Evidence

Room segmentation debug artifacts include:

- `vertical_profile_bands`
- `vertical_carved_map`
- `wall_confidence_map`
- detected structural wall lines
- `repaired_window_gaps`
- `verified_doorway_gaps`
- final room masks

