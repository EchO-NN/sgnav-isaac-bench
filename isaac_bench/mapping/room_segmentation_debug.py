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
    algorithm = str(debug.get("algorithm", "rose2_structure"))
    if algorithm in {
        "rose2_source_form",
        "rose2_source_form_v2",
        "rose2_source_faithful_v1",
        "rose2_source_external_runner",
        "upstream_rose2_pure_python",
        "upstream_rose2_vertical_or_free",
    }:
        out = out / str(episode_id)
        out.mkdir(parents=True, exist_ok=True)
        stem = "%06d_rose2" % int(step)
        png_path = out / ("%s_rooms.png" % stem)
        json_path = out / ("%s_layers.json" % stem)
    else:
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
    vertical_carved = np.asarray(debug.get("vertical_carved_map", np.zeros_like(occ)), dtype=bool)
    rejected = np.asarray(debug.get("structural_component_rejected_mask", np.zeros_like(occ)), dtype=bool)
    clutter = np.asarray(debug.get("interior_clutter_suppression_mask", np.zeros_like(occ)), dtype=bool)
    furniture = np.asarray(debug.get("furniture_suppression_mask", np.zeros_like(occ)), dtype=bool)
    suppressed = rejected | clutter | furniture
    canvas[suppressed] = (58, 82, 132)
    canvas[vertical_carved] = (115, 105, 88)
    wall_conf = np.asarray(debug.get("wall_confidence_map", np.zeros_like(occ, dtype=np.float32)), dtype=np.float32)
    if wall_conf.shape == occ.shape and np.any(wall_conf > 0):
        hot = wall_conf >= float(debug.get("wall_confidence_threshold", 0.55) or 0.55)
        canvas[hot] = (255, 105, 75)
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
    for gap in list(debug.get("repaired_window_gaps") or [])[:128]:
        _draw_gap_marker(draw, xy, gap, (255, 80, 130), scale)
    for gap in list(debug.get("verified_doorway_gaps") or [])[:128]:
        _draw_gap_marker(draw, xy, gap, (80, 255, 130), scale)
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
    legend = "ROSE2 profile | dirs=%s | walls=%d | rooms=%d win=%d door=%d" % (
        ",".join("%.2f" % float(v) for v in list(debug.get("dominant_directions_rad") or [])[:4]),
        int(debug.get("num_representative_lines", len(debug.get("representative_lines") or [])) or 0),
        len(room_masks),
        len(list(debug.get("repaired_window_gaps") or [])),
        len(list(debug.get("verified_doorway_gaps") or [])),
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
                "source": str((getattr(room, "metadata", {}) or {}).get("segmentation_source", debug.get("algorithm", "rose2_structure"))),
                "functional_split": bool((getattr(room, "metadata", {}) or {}).get("functional_split", False)),
                "unknown_reason": getattr(label, "unknown_reason", None),
            }
        )
    algorithm = str(debug.get("algorithm", "rose2_structure"))
    source_mode = str(
        debug.get(
            "source_mode",
            "declutter_reconstruct_external"
            if algorithm in {"rose2_source_external_runner", "upstream_rose2_pure_python", "upstream_rose2_vertical_or_free"}
            else ("source_form_no_ros" if algorithm in {"rose2_source_form", "rose2_source_form_v2", "rose2_source_faithful_v1"} else "local_rose2_lite"),
        )
    )
    return {
        "algorithm": algorithm,
        "source_mode": source_mode,
        "num_rooms": int(debug.get("num_rooms", len(room_masks)) or 0),
        "num_wall_lines": int(debug.get("num_wall_lines", len(debug.get("wall_lines") or debug.get("hough_segments") or [])) or 0),
        "main_directions": [float(v) for v in list(debug.get("main_directions") or debug.get("dominant_directions_rad") or [])],
        "room_masks": rose2_debug_json_ready(list(debug.get("room_masks") or [])),
        "room_labels": [
            {
                "room_id": str(room_id),
                "category": str(getattr(label, "category", "unknown")),
                "label_reliability": float(getattr(label, "label_reliability", 0.0) or 0.0),
            }
            for room_id, label in sorted(room_labels.items())
        ],
        "strict_fallback_used": bool(debug.get("strict_fallback_used", False)),
        "vertical_profile_bands": rose2_debug_json_ready(debug.get("vertical_profile_bands", {})),
        "vertical_cell_evidence_summary": rose2_debug_json_ready(debug.get("vertical_cell_evidence_summary", {})),
        "vertical_or_free_z_min_m": float(debug.get("vertical_or_free_z_min_m", 0.20) or 0.20),
        "vertical_or_free_z_max_m": float(debug.get("vertical_or_free_z_max_m", 2.00) or 2.00),
        "vertical_free_overrides_occupied": bool(debug.get("vertical_free_overrides_occupied", False)),
        "vertical_free_overridden_occupied_cells": int(debug.get("vertical_free_overridden_occupied_cells", 0) or 0),
        "roomseg_input_source": str(debug.get("roomseg_input_source", "")),
        "vertical_observed_map": rose2_debug_json_ready(np.asarray(debug.get("vertical_observed_map", []), dtype=np.uint8).tolist()),
        "vertical_observed_cells": int(debug.get("vertical_observed_cells", 0) or 0),
        "vertical_or_free_map": rose2_debug_json_ready(np.asarray(debug.get("vertical_or_free_map", []), dtype=np.uint8).tolist()),
        "vertical_or_free_cells": int(debug.get("vertical_or_free_cells", 0) or 0),
        "vertical_free_added_to_roomseg_cells": int(debug.get("vertical_free_added_to_roomseg_cells", 0) or 0),
        "navigation_free_added_to_roomseg_cells": int(debug.get("navigation_free_added_to_roomseg_cells", 0) or 0),
        "navigation_free_not_added_to_roomseg_cells": int(debug.get("navigation_free_not_added_to_roomseg_cells", 0) or 0),
        "initial_roomseg_free": rose2_debug_json_ready(np.asarray(debug.get("initial_roomseg_free", []), dtype=np.uint8).tolist()),
        "initial_roomseg_occupied": rose2_debug_json_ready(np.asarray(debug.get("initial_roomseg_occupied", []), dtype=np.uint8).tolist()),
        "repaired_roomseg_free": rose2_debug_json_ready(np.asarray(debug.get("repaired_roomseg_free", []), dtype=np.uint8).tolist()),
        "repaired_roomseg_occupied": rose2_debug_json_ready(np.asarray(debug.get("repaired_roomseg_occupied", []), dtype=np.uint8).tolist()),
        "vertical_carved_map": rose2_debug_json_ready(np.asarray(debug.get("vertical_carved_map", []), dtype=np.uint8).tolist()),
        "wall_confidence_map": rose2_debug_json_ready(np.asarray(debug.get("wall_confidence_map", []), dtype=np.float32).round(3).tolist()),
        "wall_confidence_threshold": float(debug.get("wall_confidence_threshold", 0.55) or 0.55),
        "structural_wall_cells": int(debug.get("structural_wall_cells", 0) or 0),
        "vertical_carved_cells": int(debug.get("vertical_carved_cells", 0) or 0),
        "furniture_suppressed_cells": int(debug.get("furniture_suppressed_cells", 0) or 0),
        "interior_clutter_suppressed_cells": int(debug.get("interior_clutter_suppressed_cells", 0) or 0),
        "structural_component_rejected_cells": int(debug.get("structural_component_rejected_cells", 0) or 0),
        "structural_component_rejected_mask": rose2_debug_json_ready(np.asarray(debug.get("structural_component_rejected_mask", []), dtype=np.uint8).tolist()),
        "interior_clutter_suppression_mask": rose2_debug_json_ready(np.asarray(debug.get("interior_clutter_suppression_mask", []), dtype=np.uint8).tolist()),
        "furniture_suppression_mask": rose2_debug_json_ready(np.asarray(debug.get("furniture_suppression_mask", []), dtype=np.uint8).tolist()),
        "repaired_window_gaps": rose2_debug_json_ready(list(debug.get("repaired_window_gaps") or [])),
        "verified_doorway_gaps": rose2_debug_json_ready(list(debug.get("verified_doorway_gaps") or [])),
        "source_repository": debug.get("source_repository"),
        "source_provenance": rose2_debug_json_ready(debug.get("source_provenance", {})),
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
            {"name": "vertical_observed_map", "enabled": True, "primitive_count": int(debug.get("vertical_observed_cells", 0) or 0)},
            {"name": "vertical_or_free_map", "enabled": True, "primitive_count": int(debug.get("vertical_or_free_cells", 0) or 0)},
            {
                "name": "vertical_free_overridden_occupied_cells",
                "enabled": True,
                "primitive_count": int(debug.get("vertical_free_overridden_occupied_cells", 0) or 0),
            },
            {"name": "vertical_carved_map", "enabled": True, "primitive_count": int(debug.get("vertical_carved_cells", 0) or 0)},
            {"name": "wall_confidence_map", "enabled": True, "primitive_count": int(debug.get("structural_wall_cells", 0) or 0)},
            {"name": "structural_component_rejected_mask", "enabled": True, "primitive_count": int(debug.get("structural_component_rejected_cells", 0) or 0)},
            {"name": "interior_clutter_suppression_mask", "enabled": True, "primitive_count": int(debug.get("interior_clutter_suppressed_cells", 0) or 0)},
            {"name": "clean_structure_map", "enabled": True, "primitive_count": int(np.count_nonzero(np.asarray(debug.get("clean_structure_map", []))))},
            {"name": "hough_segments", "enabled": True, "primitive_count": int(debug.get("num_hough_segments", 0) or 0)},
            {"name": "wall_clusters", "enabled": True, "primitive_count": int(debug.get("num_wall_clusters", 0) or 0)},
            {"name": "representative_wall_lines", "enabled": True, "primitive_count": int(debug.get("num_representative_lines", 0) or 0)},
            {"name": "physical_room_masks", "enabled": True, "primitive_count": int(debug.get("num_physical_rooms", 0) or 0)},
            {"name": "final_room_masks", "enabled": True, "primitive_count": len(room_masks)},
            {"name": "room_labels", "enabled": True, "primitive_count": len(room_labels)},
            {"name": "repaired_window_gaps", "enabled": True, "primitive_count": len(list(debug.get("repaired_window_gaps") or []))},
            {"name": "verified_doorway_gaps", "enabled": True, "primitive_count": len(list(debug.get("verified_doorway_gaps") or []))},
            {
                "name": "rose2_source_external",
                "enabled": bool(debug.get("actual_backend") == "rose2_source_external_runner" or debug.get("source_backend") == "rose2_source_external_runner"),
                "primitive_count": int(debug.get("source_room_count", debug.get("num_physical_rooms", 0)) or 0),
                "backend": str(debug.get("actual_backend", debug.get("source_backend", ""))),
                "work_dir": str((debug.get("external_runner_summary") or {}).get("work_dir", "")) if isinstance(debug.get("external_runner_summary"), dict) else "",
                "metric_map_png": str(debug.get("metric_map_path", "")),
                "source_output_png": str(debug.get("source_output_png", debug.get("source_output_path", ""))),
                "parsed_labels_png": str(debug.get("parsed_labels_png", "")),
                "summary_json": str(debug.get("summary_json", "")),
                "room_count": int(debug.get("source_room_count", debug.get("num_physical_rooms", 0)) or 0),
                "source_form_used_for_final": bool(debug.get("source_form_used_for_final", False)),
            },
        ],
    }


def _draw_gap_marker(draw: ImageDraw.ImageDraw, xy, gap: Mapping[str, object], color: tuple[int, int, int], scale: int) -> None:
    axis = str(gap.get("axis", "vertical"))
    index = int(gap.get("index", 0) or 0)
    start = int(gap.get("start", 0) or 0)
    end = int(gap.get("end", start) or start)
    p0 = (start, index) if axis == "vertical" else (index, start)
    p1 = (end, index) if axis == "vertical" else (index, end)
    draw.line([xy(p0), xy(p1)], fill=color, width=max(2, int(scale)))


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
