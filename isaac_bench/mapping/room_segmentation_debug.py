from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaac_bench.mapping.rose2_room_segmentation import rose2_debug_json_ready, write_rose2_layers_json


def save_rose2_roomseg_debug(
    *,
    out_dir: str | Path,
    episode_id: str,
    step: int,
    occupancy: np.ndarray,
    room_masks: Sequence[object],
    debug: Mapping[str, object],
    room_labels: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "%s_%06d_rose2" % (str(episode_id), int(step))
    png_path = out / ("%s.png" % stem)
    json_path = out / ("%s.layers.json" % stem)
    image = render_rose2_roomseg_panel(occupancy, room_masks, debug, room_labels or {})
    image.save(png_path)
    layers = rose2_layers_payload(room_masks, debug, room_labels or {})
    write_rose2_layers_json(json_path, layers)
    return png_path, json_path


def render_rose2_roomseg_panel(
    occupancy: np.ndarray,
    room_masks: Sequence[object],
    debug: Mapping[str, object],
    room_labels: Mapping[str, object],
) -> Image.Image:
    occ = np.asarray(occupancy, dtype=bool)
    h, w = occ.shape
    scale = max(2, min(6, int(720 / max(h, w, 1))))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (26, 28, 32)
    canvas[occ] = (230, 230, 230)
    clean = np.asarray(debug.get("clean_structure_map", np.zeros_like(occ)), dtype=bool)
    canvas[clean] = (255, 190, 70)
    for idx, room in enumerate(room_masks):
        mask = np.asarray(getattr(room, "mask", None), dtype=bool)
        if mask.shape != occ.shape or not np.any(mask):
            continue
        color = np.asarray(_room_color(idx), dtype=np.float32)
        canvas[mask] = np.clip(canvas[mask].astype(np.float32) * 0.50 + color[None, :] * 0.50, 0, 255).astype(np.uint8)
        boundary = _boundary(mask)
        canvas[boundary] = np.asarray(np.clip(color * 1.15, 0, 255), dtype=np.uint8)
    img = Image.fromarray(canvas).resize((w * scale, h * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def xy(cell):
        return int((float(cell[1]) + 0.5) * scale), int((float(cell[0]) + 0.5) * scale)

    for seg in list(debug.get("hough_segments") or [])[:512]:
        draw.line([xy(seg.get("p0", (0, 0))), xy(seg.get("p1", (0, 0)))], fill=(255, 80, 60), width=max(1, scale // 2))
    for line in list(debug.get("representative_lines") or [])[:256]:
        draw.line([xy(line.get("p0", (0, 0))), xy(line.get("p1", (0, 0)))], fill=(60, 180, 255), width=max(2, scale))
    for idx, room in enumerate(room_masks):
        label = room_labels.get(getattr(room, "room_id", ""))
        category = str(getattr(label, "category", "unknown"))
        reliability = float(getattr(label, "label_reliability", 0.0) or 0.0)
        rr, cc = np.nonzero(np.asarray(getattr(room, "mask", None), dtype=bool))
        if rr.size:
            x, y = xy((float(np.mean(rr)), float(np.mean(cc))))
            text = "%s | %s | %.2f" % (getattr(room, "room_id", "room"), category, reliability)
            draw.rectangle((x, y, x + 7 * len(text), y + 12), fill=(0, 0, 0))
            draw.text((x + 2, y + 1), text, fill=(250, 250, 255), font=font)
    legend = "ROSE2 structure | dirs=%s | walls=%d | rooms=%d" % (
        ",".join("%.2f" % float(v) for v in list(debug.get("dominant_directions_rad") or [])[:4]),
        int(debug.get("num_representative_lines", len(debug.get("representative_lines") or [])) or 0),
        len(room_masks),
    )
    draw.rectangle((4, 4, 8 + 7 * len(legend), 20), fill=(0, 0, 0))
    draw.text((8, 7), legend, fill=(255, 255, 255), font=font)
    return img


def rose2_layers_payload(room_masks: Sequence[object], debug: Mapping[str, object], room_labels: Mapping[str, object]) -> dict:
    rooms = []
    for room in room_masks:
        label = room_labels.get(getattr(room, "room_id", ""))
        rooms.append(
            {
                "room_id": getattr(room, "room_id", ""),
                "category": str(getattr(label, "category", "unknown")),
                "label_reliability": float(getattr(label, "label_reliability", 0.0) or 0.0),
                "area_m2": float(getattr(room, "area_m2", 0.0)),
                "source": "rose2_structure",
                "functional_split": bool((getattr(room, "metadata", {}) or {}).get("functional_split", False)),
                "unknown_reason": getattr(label, "unknown_reason", None),
            }
        )
    return {
        "algorithm": "rose2_structure",
        "dominant_directions_rad": [float(v) for v in list(debug.get("dominant_directions_rad") or [])],
        "num_hough_segments": int(debug.get("num_hough_segments", len(debug.get("hough_segments") or [])) or 0),
        "num_wall_clusters": int(debug.get("num_wall_clusters", len(debug.get("wall_clusters") or [])) or 0),
        "num_representative_lines": int(debug.get("num_representative_lines", len(debug.get("representative_lines") or [])) or 0),
        "num_physical_rooms": int(debug.get("num_physical_rooms", 0) or 0),
        "num_final_rooms": int(debug.get("num_final_rooms", len(room_masks)) or 0),
        "rooms": rooms,
        "merge_split_decisions": rose2_debug_json_ready(list(debug.get("merge_split_decisions") or debug.get("adjacency_decisions") or [])),
        "layers": [
            {"name": "input_occupancy", "enabled": True, "primitive_count": int(np.count_nonzero(np.asarray(debug.get("clean_structure_map", []))))},
            {"name": "clean_structure_map", "enabled": True, "primitive_count": int(np.count_nonzero(np.asarray(debug.get("clean_structure_map", []))))},
            {"name": "hough_segments", "enabled": True, "primitive_count": int(debug.get("num_hough_segments", 0) or 0)},
            {"name": "wall_clusters", "enabled": True, "primitive_count": int(debug.get("num_wall_clusters", 0) or 0)},
            {"name": "representative_wall_lines", "enabled": True, "primitive_count": int(debug.get("num_representative_lines", 0) or 0)},
            {"name": "physical_room_masks", "enabled": True, "primitive_count": int(debug.get("num_physical_rooms", 0) or 0)},
            {"name": "final_room_masks", "enabled": True, "primitive_count": len(room_masks)},
            {"name": "room_labels", "enabled": True, "primitive_count": len(room_labels)},
        ],
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    neighbors = padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]
    return arr & ~neighbors


def _room_color(index: int) -> tuple[int, int, int]:
    palette = [
        (110, 170, 255),
        (255, 150, 95),
        (130, 220, 145),
        (210, 145, 255),
        (245, 210, 90),
        (95, 220, 220),
        (255, 120, 180),
    ]
    return palette[int(index) % len(palette)]
