import json

from isaac_bench.scripts.dump_sgnav_step import main as dump_sgnav_step_main


def test_dump_sgnav_step_includes_room_segmentation_and_semantics(tmp_path):
    graph_debug = tmp_path / "graph_step_000001.json"
    out = tmp_path / "step.json"
    graph_debug.write_text(
        json.dumps(
            {
                "objects": [],
                "groups": [],
                "rooms": [{"id": "room:room_0001", "caption": "unknown"}],
                "edges": [],
                "room_segmentation": {
                    "source": "rose2_structure",
                    "algorithm": "rose2_structure",
                    "room_count": 1,
                    "rooms": [{"room_id": "room_0001", "area_m2": 3.0}],
                },
                "room_semantics": {
                    "backend": "vlm",
                    "allowed_categories": ["bathroom", "unknown"],
                    "labels": [{"room_id": "room_0001", "category": "unknown", "confidence": 0.4}],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert dump_sgnav_step_main(["--output", str(out), "--graph-debug-dump", str(graph_debug)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["room_segmentation"]["room_count"] == 1
    assert payload["room_semantics"]["backend"] == "vlm"
    assert payload["rooms"][0]["id"] == "room:room_0001"
