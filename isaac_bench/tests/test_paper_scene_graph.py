import numpy as np

from isaac_bench.graph.paper_scene_graph import ObjectEdge, PaperSceneGraph
from isaac_bench.perception.detection_types import FusedInstance


def _instance(instance_id: str, category: str, center, node_type: str = "object") -> FusedInstance:
    center_arr = np.asarray(center, dtype=np.float32)
    points = np.asarray(
        [
            center_arr + np.asarray([-0.1, -0.1, 0.0], dtype=np.float32),
            center_arr + np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    return FusedInstance(
        instance_id=instance_id,
        category=category,
        node_type=node_type,
        confidence=0.9,
        point_cloud_world=points,
        bbox_world=np.stack([np.min(points, axis=0), np.max(points, axis=0)], axis=0),
        center_world=center_arr,
        last_mask=None,
        last_seen_step=1,
        observed_count=1,
    )


def test_paper_scene_graph_registers_nodes_and_unknown_room_affiliation():
    graph = PaperSceneGraph()
    graph.update_object_and_room_nodes([_instance("0", "chair", (1.0, 1.0, 0.5))])

    assert "object:0" in graph.object_nodes
    assert "room:unknown_room" in graph.room_nodes
    assert graph.object_nodes["object:0"].room_id == "room:unknown_room"
    assert graph.affiliation_edges[0].relation == "belongs_to"


def test_paper_scene_graph_assigns_object_to_projected_room():
    graph = PaperSceneGraph()
    room = _instance("room0", "living room", (1.0, 1.0, 0.5), node_type="room")
    room.point_cloud_world = np.asarray([[0.0, 0.0, 0.0], [3.0, 3.0, 2.0]], dtype=np.float32)
    room.bbox_world = np.asarray([[0.0, 0.0, 0.0], [3.0, 3.0, 2.0]], dtype=np.float32)

    graph.update_object_and_room_nodes([room, _instance("obj0", "table", (1.0, 1.0, 0.5))])

    assert graph.object_nodes["object:obj0"].room_id == "room:room0"
    assert graph.room_nodes["room:room0"].contained_object_ids == ["object:obj0"]


def test_paper_scene_graph_builds_related_group_from_edges():
    graph = PaperSceneGraph()
    graph.update_object_and_room_nodes(
        [
            _instance("table", "table", (1.0, 1.0, 0.5)),
            _instance("chair", "chair", (1.5, 1.0, 0.5)),
        ]
    )
    graph.object_edges.append(ObjectEdge("object:table", "object:chair", "next to", confidence=0.8))

    graph.update_group_nodes()

    assert len(graph.group_nodes) == 1
    group = next(iter(graph.group_nodes.values()))
    assert group.object_ids == ["object:chair", "object:table"]
    assert group.room_id == "room:unknown_room"
    assert any(edge.src_id == group.id and edge.dst_id == "room:unknown_room" for edge in graph.affiliation_edges)
