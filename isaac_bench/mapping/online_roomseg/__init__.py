from __future__ import annotations

from isaac_bench.mapping.online_roomseg.online_room_segmenter import (
    HEIGHT_PROFILE_CORRIDOR_V7_BACKEND,
    HEIGHT_PROFILE_CORRIDOR_V7_CONTEXT,
    HEIGHT_PROFILE_DOOR_WALL_V6_2_BACKEND,
    HEIGHT_PROFILE_DOOR_WALL_V6_2_CONTEXT,
    ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
    ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
    ONLINE_LINE_EXTEND_ROOMSEG_V4_BACKEND,
    ONLINE_LINE_EXTEND_ROOMSEG_V4_CONTEXT,
    ONLINE_ROSE_STYLE_BACKEND,
    ONLINE_ROSE_STYLE_CONTEXT,
    ROOMSEG_EVIDENCE_LINE_CLOSURE_V3_BACKEND,
    ROOMSEG_EVIDENCE_LINE_CLOSURE_V3_CONTEXT,
    OnlineRoseStyleConfig,
    OnlineRoseStyleRoomSegmenter,
    run_online_rose_style_roomseg,
)

HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND = "height_profile_door_wall_v8"
HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT = "height_profile_door_wall_v8_vlm"
VOXEL_OCCUPANCY_ROOMSEG_BACKEND = "voxel_occupancy_door_wall_v9"
VOXEL_OCCUPANCY_ROOMSEG_CONTEXT = "voxel_occupancy_door_wall_v9_vlm"


def __getattr__(name: str):
    if name in {
        "HeightProfileDoorWallRoomSegConfig",
        "HeightProfileDoorWallRoomSegmenter",
        "run_height_profile_door_wall_roomseg",
    }:
        from isaac_bench.mapping import height_profile_door_wall_roomseg as module

        return getattr(module, name)
    if name in {
        "VoxelOccupancyDoorWallRoomSegConfig",
        "VoxelOccupancyDoorWallRoomSegmenter",
        "run_voxel_occupancy_door_wall_roomseg",
    }:
        from isaac_bench.mapping import voxel_occupancy_door_wall_roomseg as module

        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "HEIGHT_PROFILE_DOOR_WALL_V8_BACKEND",
    "HEIGHT_PROFILE_DOOR_WALL_V8_CONTEXT",
    "VOXEL_OCCUPANCY_ROOMSEG_BACKEND",
    "VOXEL_OCCUPANCY_ROOMSEG_CONTEXT",
    "HEIGHT_PROFILE_CORRIDOR_V7_BACKEND",
    "HEIGHT_PROFILE_CORRIDOR_V7_CONTEXT",
    "HEIGHT_PROFILE_DOOR_WALL_V6_2_BACKEND",
    "HEIGHT_PROFILE_DOOR_WALL_V6_2_CONTEXT",
    "ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND",
    "ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT",
    "ONLINE_LINE_EXTEND_ROOMSEG_V4_BACKEND",
    "ONLINE_LINE_EXTEND_ROOMSEG_V4_CONTEXT",
    "ONLINE_ROSE_STYLE_BACKEND",
    "ONLINE_ROSE_STYLE_CONTEXT",
    "ROOMSEG_EVIDENCE_LINE_CLOSURE_V3_BACKEND",
    "ROOMSEG_EVIDENCE_LINE_CLOSURE_V3_CONTEXT",
    "HeightProfileDoorWallRoomSegConfig",
    "HeightProfileDoorWallRoomSegmenter",
    "VoxelOccupancyDoorWallRoomSegConfig",
    "VoxelOccupancyDoorWallRoomSegmenter",
    "OnlineRoseStyleConfig",
    "OnlineRoseStyleRoomSegmenter",
    "run_online_rose_style_roomseg",
    "run_height_profile_door_wall_roomseg",
    "run_voxel_occupancy_door_wall_roomseg",
]
