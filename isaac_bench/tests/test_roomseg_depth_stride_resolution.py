from isaac_bench.scripts.run_one_episode import _resolve_roomseg_depth_stride_px


def test_roomseg_depth_stride_uses_densest_roomseg_request():
    cfg = {
        "online_roomseg": {"depth": {"roomseg_depth_stride_px": 4}},
        "online_watershed_roomseg": {"depth": {"roomseg_depth_stride_px": 2}},
    }

    assert _resolve_roomseg_depth_stride_px(cfg, 8) == 2


def test_roomseg_depth_stride_accepts_top_level_depth_block():
    cfg = {"depth": {"roomseg_depth_stride_px": 3}}

    assert _resolve_roomseg_depth_stride_px(cfg, 8) == 3


def test_roomseg_depth_stride_ignores_invalid_values():
    cfg = {
        "depth": {"roomseg_depth_stride_px": 0},
        "online_roomseg": {"depth": {"roomseg_depth_stride_px": "bad"}},
    }

    assert _resolve_roomseg_depth_stride_px(cfg, 8) == 8
