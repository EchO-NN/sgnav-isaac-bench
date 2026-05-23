from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw


V6_2_COLORS: dict[str, tuple[int, int, int]] = {
    "free": (224, 224, 238),
    "unknown": (42, 42, 46),
    "wall": (18, 18, 20),
    "door": (255, 176, 36),
    "corridor_separator": (226, 47, 47),
    "frontier": (0, 210, 230),
    "agent": (220, 36, 36),
}

V6_2_LEGEND_ITEMS: tuple[str, ...] = (
    "room free",
    "wall",
    "door/header",
    "corridor separator",
    "unknown",
    "frontier",
    "agent",
)


def build_v6_2_debug_layers(
    *,
    wall_endpoint_ratio: np.ndarray,
    best_door_lower_free_ratio: np.ndarray,
    best_door_upper_occ_ratio: np.ndarray,
    best_door_transition_z: np.ndarray,
    wall_mask_raw: np.ndarray,
    door_mask_raw: np.ndarray,
    wall_mask_repaired: np.ndarray,
    door_mask_repaired: np.ndarray,
    wall_line_support_mask: np.ndarray,
    corridor_l_corner_candidates: np.ndarray,
    accepted_corridor_separator_mask: np.ndarray,
    room_cut_mask: np.ndarray,
    final_room_labels: np.ndarray,
    free_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "height_profile_wall_endpoint_ratio": np.asarray(wall_endpoint_ratio, dtype=np.float32),
        "height_profile_best_door_lower_free_ratio": np.asarray(best_door_lower_free_ratio, dtype=np.float32),
        "height_profile_best_door_upper_occ_ratio": np.asarray(best_door_upper_occ_ratio, dtype=np.float32),
        "height_profile_best_door_transition_z": np.asarray(best_door_transition_z, dtype=np.float32),
        "wall_mask_raw": np.asarray(wall_mask_raw, dtype=bool),
        "door_mask_raw": np.asarray(door_mask_raw, dtype=bool),
        "wall_mask_repaired": np.asarray(wall_mask_repaired, dtype=bool),
        "door_mask_repaired": np.asarray(door_mask_repaired, dtype=bool),
        "wall_line_support_mask": np.asarray(wall_line_support_mask, dtype=bool),
        "corridor_l_corner_candidates": np.asarray(corridor_l_corner_candidates, dtype=bool),
        "accepted_corridor_separator_mask": np.asarray(accepted_corridor_separator_mask, dtype=bool),
        "room_cut_mask": np.asarray(room_cut_mask, dtype=bool),
        "final_room_labels": np.asarray(final_room_labels, dtype=np.int32),
        "v6_2_free_mask": np.asarray(free_mask, dtype=bool),
        "v6_2_unknown_mask": np.asarray(unknown_mask, dtype=bool),
    }


def render_v6_2_debug_panel(
    layers: Mapping[str, np.ndarray],
    *,
    debug: Mapping[str, object] | None = None,
    scale: int = 4,
) -> Image.Image:
    labels = np.asarray(layers.get("final_room_labels"), dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError("final_room_labels layer is required")
    free = np.asarray(layers.get("v6_2_free_mask", labels > 0), dtype=bool)
    unknown = np.asarray(layers.get("v6_2_unknown_mask", labels == -1), dtype=bool)
    wall = np.asarray(layers.get("wall_mask_repaired", np.zeros_like(free)), dtype=bool)
    door = np.asarray(layers.get("door_mask_repaired", np.zeros_like(free)), dtype=bool)
    corridor = np.asarray(layers.get("accepted_corridor_separator_mask", np.zeros_like(free)), dtype=bool)

    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    arr[:, :] = V6_2_COLORS["unknown"]
    arr[free] = V6_2_COLORS["free"]
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        color = _label_color(label)
        arr[labels == label] = _blend(arr[labels == label], color, 0.42)
    arr[unknown] = V6_2_COLORS["unknown"]
    arr[wall] = V6_2_COLORS["wall"]
    arr[door] = V6_2_COLORS["door"]
    arr[corridor] = V6_2_COLORS["corridor_separator"]

    img = Image.fromarray(arr)
    scale = max(1, int(scale))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    footer_h = 64
    legend_w = 210
    panel = Image.new("RGB", (img.width + legend_w, img.height + footer_h), (24, 24, 26))
    panel.paste(img, (0, 0))
    draw = ImageDraw.Draw(panel)
    _draw_legend(draw, img.width + 12, 12)
    text = _footer_text(layers, debug or {})
    draw.text((8, img.height + 8), text[0], fill=(235, 235, 235))
    draw.text((8, img.height + 30), text[1], fill=(218, 218, 218))
    return panel


def save_v6_2_debug_panel(
    *,
    out_dir: str | Path,
    layers: Mapping[str, np.ndarray],
    debug: Mapping[str, object] | None = None,
) -> str:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "roomseg_v6_2_panel.png"
    render_v6_2_debug_panel(layers, debug=debug).save(path)
    return str(path)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = (
        ("room free", V6_2_COLORS["free"]),
        ("wall", V6_2_COLORS["wall"]),
        ("door/header", V6_2_COLORS["door"]),
        ("corridor separator", V6_2_COLORS["corridor_separator"]),
        ("unknown", V6_2_COLORS["unknown"]),
        ("frontier", V6_2_COLORS["frontier"]),
        ("agent", V6_2_COLORS["agent"]),
    )
    draw.text((x, y), "Legend", fill=(240, 240, 240))
    yy = y + 18
    for label, color in items:
        draw.rectangle((x, yy, x + 12, yy + 12), fill=color)
        draw.text((x + 18, yy - 1), label, fill=(225, 225, 225))
        yy += 18


def _footer_text(layers: Mapping[str, np.ndarray], debug: Mapping[str, object]) -> tuple[str, str]:
    wall = int(np.count_nonzero(np.asarray(layers.get("wall_mask_repaired"), dtype=bool)))
    door = int(np.count_nonzero(np.asarray(layers.get("door_mask_repaired"), dtype=bool)))
    free = int(np.count_nonzero(np.asarray(layers.get("v6_2_free_mask"), dtype=bool)))
    unknown = int(np.count_nonzero(np.asarray(layers.get("v6_2_unknown_mask"), dtype=bool)))
    corridor = int(np.count_nonzero(np.asarray(layers.get("accepted_corridor_separator_mask"), dtype=bool)))
    labels = np.asarray(layers.get("final_room_labels"), dtype=np.int32)
    rooms = int(len([v for v in np.unique(labels) if int(v) > 0]))
    wall_thr = float(debug.get("wall_endpoint_ratio_min", 0.95))
    low_thr = float(debug.get("door_lower_free_ratio_min", 0.95))
    up_thr = float(debug.get("door_upper_occupied_ratio_min", 0.95))
    transition = float(debug.get("door_transition_min_height_m", 1.80))
    return (
        "RoomSeg v6.2 | wall=%d door=%d free=%d unknown=%d corridor_sep=%d rooms=%d"
        % (wall, door, free, unknown, corridor, rooms),
        "wall_thr=%.2f door_low_free_thr=%.2f door_up_occ_thr=%.2f transition_min=%.2fm"
        % (wall_thr, low_thr, up_thr, transition),
    )


def _label_color(label: int) -> tuple[int, int, int]:
    value = int(label) * 2654435761
    return (
        80 + ((value >> 0) & 0x7F),
        80 + ((value >> 8) & 0x7F),
        80 + ((value >> 16) & 0x7F),
    )


def _blend(base: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    arr = np.asarray(base, dtype=np.float32)
    overlay = np.asarray(color, dtype=np.float32).reshape(1, 3)
    return np.clip(arr * (1.0 - float(alpha)) + overlay * float(alpha), 0, 255).astype(np.uint8)
