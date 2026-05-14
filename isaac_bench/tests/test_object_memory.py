from isaac_bench.perception.detection_types import Detection3D
from isaac_bench.perception.object_memory import ObjectMemory


def test_object_memory_merge():
    mem = ObjectMemory(merge_radius_m=0.5)
    d1 = Detection3D("chair", "chair", 0.8, (1.0, 2.0, 0.5), (0, 0, 10, 10))
    d2 = Detection3D("chair", "chair", 0.7, (1.1, 2.0, 0.5), (0, 0, 10, 10))
    mem.update([d1], step_id=1)
    mem.update([d2], step_id=2)
    assert len(mem.nodes) == 1
    assert mem.nodes[0].observed_count == 2
    assert mem.nodes[0].confidence == 0.8


def test_object_memory_merges_normalized_alias_categories():
    mem = ObjectMemory(merge_radius_m=0.5)
    d1 = Detection3D("sofa", "sofa", 0.8, (1.0, 2.0, 0.5), (0, 0, 10, 10))
    d2 = Detection3D("couch", "couch", 0.7, (1.2, 2.0, 0.8), (0, 0, 10, 10))

    mem.update([d1], step_id=1)
    mem.update([d2], step_id=2)

    assert len(mem.nodes) == 1
    assert mem.nodes[0].category == "sofa"
    assert mem.nodes[0].observed_count == 2


def test_object_memory_dedupes_existing_close_nodes():
    mem = ObjectMemory(merge_radius_m=0.5)
    mem.update(
        [
            Detection3D("tv", "tv", 0.8, (1.0, 2.0, 0.5), (0, 0, 10, 10)),
            Detection3D("television", "television", 0.9, (1.2, 2.1, 1.0), (0, 0, 10, 10)),
        ],
        step_id=1,
    )

    assert len(mem.nodes) == 1
    assert mem.nodes[0].category == "tv"
    assert mem.nodes[0].observed_count == 2
    assert mem.nodes[0].confidence == 0.9
