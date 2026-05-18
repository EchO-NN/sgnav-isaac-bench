import math

import numpy as np
import pytest

from isaac_bench.mapping.structure_extraction import (
    StructureExtractionConfig,
    cluster_wall_segments,
    detect_line_segments,
    dominant_directions_from_fft,
    face_adjacency_edges,
    faces_from_boundary_map,
    remove_isolated_clutter,
    split_faces_by_topology,
)


def _angle_close(value, target, tol=0.20):
    diff = abs((float(value) % math.pi) - (float(target) % math.pi))
    return min(diff, math.pi - diff) <= tol


def test_rose2_detects_dominant_wall_directions():
    occupied = np.zeros((64, 64), dtype=bool)
    occupied[8, 8:56] = True
    occupied[55, 8:56] = True
    occupied[8:56, 8] = True
    occupied[8:56, 55] = True

    directions = dominant_directions_from_fft(occupied, count=2)

    assert any(_angle_close(v, 0.0) for v in directions)
    assert any(_angle_close(v, math.pi / 2.0) for v in directions)


def test_rose2_removes_clutter_from_occupied_map():
    config = StructureExtractionConfig(resolution_m=0.10, clutter_component_max_area_m2=0.20)
    occupied = np.zeros((40, 40), dtype=bool)
    occupied[20, 4:34] = True
    occupied[5:8, 5:8] = True
    free = np.ones_like(occupied, dtype=bool)
    free[occupied] = False

    cleaned = remove_isolated_clutter(occupied, free, config)

    assert np.count_nonzero(cleaned[20, 4:34]) >= 25
    assert not np.any(cleaned[5:8, 5:8])


def test_rose2_wall_clusters_merge_collinear_segments():
    config = StructureExtractionConfig(resolution_m=0.10, wall_cluster_gap_m=1.0, wall_cluster_distance_m=0.2)
    segments = [
        {"segment_id": 1, "p0": [12, 2], "p1": [12, 11], "angle_rad": 0.0, "length_px": 9.0, "support_ratio": 1.0},
        {"segment_id": 2, "p0": [12, 14], "p1": [12, 25], "angle_rad": 0.0, "length_px": 11.0, "support_ratio": 1.0},
    ]

    clusters = cluster_wall_segments(segments, [0.0, math.pi / 2.0], config)

    assert len(clusters) == 1
    assert clusters[0]["support_length_m"] > 1.5


def test_rose2_faces_cluster_into_rooms():
    config = StructureExtractionConfig(resolution_m=0.10, min_room_area_m2=0.5)
    free = np.ones((30, 30), dtype=bool)
    boundary = np.zeros_like(free)
    boundary[:, 15] = True
    free[:, 15] = False

    labels, faces = faces_from_boundary_map(free, boundary, config)
    edges = face_adjacency_edges(labels, boundary)

    assert len(faces) == 2
    assert sorted(int(v) for v in np.unique(labels) if v > 0) == [1, 2]
    assert edges == []


def test_rose2_partial_map_topology_split_when_needed():
    config = StructureExtractionConfig(resolution_m=0.10, min_room_area_m2=0.2)
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[3:27, 3:27] = 1
    labels[13:17, 3:27] = 0
    labels[14:16, 14:16] = 1
    free = labels > 0

    split = split_faces_by_topology(labels, free, config)

    assert len([v for v in np.unique(split) if int(v) > 0]) >= 2
