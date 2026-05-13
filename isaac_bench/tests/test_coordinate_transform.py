from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, world_xy_to_grid


def test_world_grid_roundtrip():
    info = MapInfo(resolution_m=0.05, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0, width=400, height=400)
    x, y = 1.23, -4.56
    r, c = world_xy_to_grid(x, y, info)
    x2, y2 = grid_to_world_xy(r, c, info)
    assert abs(x - x2) <= info.resolution_m
    assert abs(y - y2) <= info.resolution_m

