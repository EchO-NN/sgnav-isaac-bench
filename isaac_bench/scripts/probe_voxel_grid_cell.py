from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_CONFLICT, VOXEL_UNKNOWN


STATE_NAMES = {
    int(VOXEL_UNKNOWN): "UNKNOWN",
    int(VOXEL_FREE): "FREE",
    int(VOXEL_OCCUPIED): "OCCUPIED",
    int(VOXEL_CONFLICT): "CONFLICT",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one xy cell from a saved voxel roomseg npz.")
    parser.add_argument("--debug-npz", required=True, help="Path to roomseg debug/snapshot npz")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--col", type=int, required=True)
    args = parser.parse_args()

    path = Path(args.debug_npz)
    data = np.load(path, allow_pickle=True)
    row, col = int(args.row), int(args.col)
    print("cell row=%d col=%d" % (row, col))

    state = _first_present(data, ("voxel_state_zyx", "state"))
    log_odds = _first_present(data, ("voxel_log_odds_zyx", "log_odds"))
    sensor_range_zyx = _first_present(data, ("voxel_sensor_range_count_zyx", "sensor_range_count"))
    z_min = _scalar(data, "voxel_z_min_m", _scalar(data, "z_min_m", 0.0))
    z_res = _scalar(data, "voxel_z_resolution_m", _scalar(data, "z_resolution_m", 0.05))
    active_min = _scalar(data, "voxel_active_z_min_m", _scalar(data, "active_z_min_m", 0.10))
    active_max = _scalar(data, "voxel_active_z_max_m", _scalar(data, "active_z_max_m", np.nan))
    ceiling = _scalar(data, "voxel_ceiling_height_m", np.nan)
    print("active_z_min=%.2f active_z_max=%s ceiling=%s" % (active_min, _fmt(active_max), _fmt(ceiling)))

    if state is not None:
        state = np.asarray(state)
        if state.ndim != 3:
            raise SystemExit("voxel state must be 3D zyx")
        if row < 0 or col < 0 or row >= state.shape[1] or col >= state.shape[2]:
            raise SystemExit("row/col outside voxel grid shape %s" % (state.shape,))
        log = None if log_odds is None else np.asarray(log_odds)
        sensor = None if sensor_range_zyx is None else np.asarray(sensor_range_zyx)
        for z_idx in range(state.shape[0]):
            z = float(z_min) + (z_idx + 0.5) * float(z_res)
            value = int(state[z_idx, row, col])
            sensor_value = None if sensor is None or sensor.shape != state.shape else int(sensor[z_idx, row, col])
            if log is not None and log.shape == state.shape:
                if sensor_value is not None:
                    print("z=%.2f state=%s log=%d sensor_range=%d" % (z, STATE_NAMES.get(value, str(value)), int(log[z_idx, row, col]), sensor_value))
                else:
                    print("z=%.2f state=%s log=%d" % (z, STATE_NAMES.get(value, str(value)), int(log[z_idx, row, col])))
            else:
                if sensor_value is not None:
                    print("z=%.2f state=%s sensor_range=%d" % (z, STATE_NAMES.get(value, str(value)), sensor_value))
                else:
                    print("z=%.2f state=%s" % (z, STATE_NAMES.get(value, str(value))))
    else:
        print("voxel_state_zyx missing in npz; showing 2D classification arrays only")

    free_count = _cell(data, "voxel_active_free_count_xy", row, col)
    occ_count = _cell(data, "voxel_active_occupied_count_xy", row, col)
    unk_count = _cell(data, "voxel_active_unknown_count_xy", row, col)
    observed_count = _cell(data, "voxel_active_observed_count_xy", row, col)
    sensor_range_count = _cell(data, "voxel_sensor_range_count_xy", row, col)
    in_range_unknown_count = _cell(data, "voxel_sensor_in_range_unknown_count_xy", row, col)
    outside_range_unknown_count = _cell(data, "voxel_sensor_outside_range_unknown_count_xy", row, col)
    generalized_occupied_count = _cell(data, "voxel_generalized_occupied_count_xy", row, col)
    ratio = _cell(data, "voxel_occupied_ratio_active_xy", row, col)
    unknown_ratio = _cell(data, "voxel_unknown_ratio_active_xy", row, col)
    observed_ratio = _cell(data, "voxel_observed_ratio_active_xy", row, col)
    sensor_range_ratio = _cell(data, "voxel_sensor_range_ratio_xy", row, col)
    in_range_unknown_ratio = _cell(data, "voxel_in_range_unknown_ratio_xy", row, col)
    outside_unknown_ratio = _cell(data, "voxel_outside_unknown_ratio_xy", row, col)
    generalized_occupied_ratio = _cell(data, "voxel_generalized_occupied_ratio_xy", row, col)
    nav_free = _bool_cell(data, "voxel_nav_free_xy", row, col)
    nav_occ = _bool_cell(data, "voxel_nav_occupied_xy", row, col)
    nav_unknown = _bool_cell(data, "voxel_nav_unknown_xy", row, col)
    wall = _bool_cell(data, "voxel_wall_xy", row, col)
    vertical_free = _bool_cell(data, "voxel_vertical_free_xy", row, col)
    unknown = _bool_cell(data, "voxel_unknown_xy", row, col)
    occupied_any = _bool_cell(data, "voxel_occupied_any_xy", row, col)
    wall_generalized_raw = _bool_cell(data, "voxel_wall_generalized_raw_xy", row, col)
    wall_actual_ratio_raw = _bool_cell(data, "voxel_wall_actual_ratio_raw_xy", row, col)
    wall_actual_occupied_requirement = _bool_cell(data, "voxel_wall_actual_occupied_requirement_xy", row, col)
    wall_from_in_range_unknown = _bool_cell(data, "voxel_wall_from_in_range_unknown_xy", row, col)
    wall_rejected_by_outside_unknown = _bool_cell(data, "voxel_wall_rejected_by_outside_unknown_xy", row, col)
    outside_unknown_dominant = _bool_cell(data, "voxel_outside_unknown_dominant_xy", row, col)
    wall_line_support_raw = _bool_cell(data, "voxel_wall_line_support_raw_xy", row, col)
    strong_structural_support = _bool_cell(data, "voxel_strong_structural_support_xy", row, col)
    bridge_only_support = _bool_cell(data, "voxel_bridge_only_support_xy", row, col)
    forbidden_frontier_residual = _bool_cell(data, "voxel_forbidden_frontier_residual_support_xy", row, col)
    forbidden_unknown_boundary = _bool_cell(data, "voxel_forbidden_unknown_boundary_support_xy", row, col)
    free_conflict_support = _bool_cell(data, "voxel_free_conflict_support_xy", row, col)
    protected_structural_wall_band = _bool_cell(data, "voxel_protected_structural_wall_band_xy", row, col)
    support_seed = _bool_cell(data, "voxel_support_seed_for_projection_xy", row, col)
    support_bridge = _bool_cell(data, "voxel_support_bridge_for_projection_xy", row, col)
    projection_display = _bool_cell(data, "voxel_support_for_projection_display_xy", row, col)
    projected_wall = _bool_cell(data, "voxel_projected_structural_wall_map", row, col)
    projection_reject_code = _cell(data, "voxel_wall_projection_reject_reason_map", row, col)
    step2_target = _bool_cell(data, "voxel_step2_target_wall_map", row, col)
    step2_source = _bool_cell(data, "voxel_step2_source_line_map", row, col)
    step2_source_id = _cell(data, "voxel_step2_target_source_map", row, col)
    door_seed = _bool_cell(data, "voxel_door_seed_mask", row, col)
    reject_code = _cell(data, "voxel_door_seed_reject_reason_map", row, col)
    print("")
    print("navigation:")
    print("navigation_free=%s" % nav_free)
    print("navigation_occ=%s" % nav_occ)
    print("navigation_unknown=%s" % nav_unknown)
    print("")
    print("2D classification:")
    print("free_count=%s" % _fmt(free_count))
    print("occupied_count=%s" % _fmt(occ_count))
    print("unknown_count=%s" % _fmt(unk_count))
    print("observed_count=%s" % _fmt(observed_count))
    print("sensor_range_count=%s" % _fmt(sensor_range_count))
    print("in_range_unknown_count=%s" % _fmt(in_range_unknown_count))
    print("outside_range_unknown_count=%s" % _fmt(outside_range_unknown_count))
    print("generalized_occupied_count=%s" % _fmt(generalized_occupied_count))
    print("occupied_ratio=%s" % _fmt(ratio))
    print("unknown_ratio=%s" % _fmt(unknown_ratio))
    print("observed_ratio=%s" % _fmt(observed_ratio))
    print("sensor_range_ratio=%s" % _fmt(sensor_range_ratio))
    print("in_range_unknown_ratio=%s" % _fmt(in_range_unknown_ratio))
    print("outside_unknown_ratio=%s" % _fmt(outside_unknown_ratio))
    print("generalized_occupied_ratio=%s" % _fmt(generalized_occupied_ratio))
    print("wall=%s" % wall)
    print("vertical_free=%s" % vertical_free)
    print("unknown=%s" % unknown)
    print("")
    print("v24 generalized wall:")
    print("wall_generalized_raw=%s" % wall_generalized_raw)
    print("wall_actual_ratio_raw=%s" % wall_actual_ratio_raw)
    print("wall_actual_occupied_requirement=%s" % wall_actual_occupied_requirement)
    print("outside_unknown_dominant=%s" % outside_unknown_dominant)
    print("wall_from_in_range_unknown=%s" % wall_from_in_range_unknown)
    print("wall_rejected_by_outside_unknown=%s" % wall_rejected_by_outside_unknown)
    print("")
    print("v23/v24 wall support:")
    print("occupied_any=%s" % occupied_any)
    print("wall_line_support_raw=%s" % wall_line_support_raw)
    print("strong_structural_support=%s" % strong_structural_support)
    print("bridge_only_support=%s" % bridge_only_support)
    print("free_conflict_support=%s" % free_conflict_support)
    print("forbidden_frontier_residual_support=%s" % forbidden_frontier_residual)
    print("forbidden_unknown_boundary_support=%s" % forbidden_unknown_boundary)
    print("protected_structural_wall_band=%s" % protected_structural_wall_band)
    print("support_seed_for_projection=%s" % support_seed)
    print("support_bridge_for_projection=%s" % support_bridge)
    print("support_for_projection_display=%s" % projection_display)
    print("")
    print("projection / step2:")
    print("projected_wall=%s" % projected_wall)
    print("projection_reject_code=%s" % _fmt(projection_reject_code))
    print("step2_target_wall=%s" % step2_target)
    print("step2_source_line=%s" % step2_source)
    print("step2_target_source_id=%s" % _fmt(step2_source_id))
    print("")
    print("door:")
    print("door_seed=%s" % door_seed)
    print("seed_reject_code=%s" % _fmt(reject_code))
    return 0


def _first_present(data, keys: tuple[str, ...]):
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def _scalar(data, key: str, default):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def _cell(data, key: str, row: int, col: int):
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 2 or row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
        return None
    value = arr[row, col]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _bool_cell(data, key: str, row: int, col: int) -> bool | None:
    value = _cell(data, key, row, col)
    if value is None:
        return None
    return bool(value)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return "%.3f" % float(value)
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
