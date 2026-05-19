from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


ROSE2_ENCODING_CANDIDATES = (
    "source_black_wall_white_free",
    "source_white_wall_black_free",
    "source_metric_binary_with_unknown_border",
)


@dataclass
class ROSE2ExportBundle:
    work_dir: Path
    metric_map_path: Path
    orebro_input_path: Path
    free_png_path: Path
    occupied_png_path: Path
    unknown_png_path: Path
    overlay_png_path: Path
    npz_path: Path
    summary_json_path: Path
    chosen_encoding: str
    shape: tuple[int, int]
    encoding_candidates: list[dict]
    input_summary: dict


def export_rose2_metric_map(
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    out_dir: str | Path,
    stem: str,
    encoding: str,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free, occupied, unknown_arr, _ = normalize_rose2_masks(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
    )
    metric = _metric_array_for_encoding(free=free, occupied=occupied, unknown=unknown_arr, encoding=encoding)
    metric_path = out / ("%s.metric_map.png" % stem)
    orebro_path = out / ("%s.orebro_input.png" % stem)
    Image.fromarray(metric, mode="L").save(metric_path)
    Image.fromarray(metric, mode="L").save(orebro_path)
    return {"metric_map": metric_path, "orebro_input": orebro_path}


def export_rose2_upstream_input(
    *,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    resolution_m: float,
    out_dir: str | Path,
    stem: str,
    encoding: str = "auto",
    vertical_observed: np.ndarray | None = None,
    vertical_free: np.ndarray | None = None,
    wall_confidence_map: np.ndarray | None = None,
) -> ROSE2ExportBundle:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    free, occupied, unknown_arr, corrections = normalize_rose2_masks(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
    )
    chosen_encoding, candidates = choose_best_rose2_encoding(
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown_arr,
        requested_encoding=str(encoding or "auto"),
    )
    metric_paths = export_rose2_metric_map(
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown_arr,
        out_dir=out,
        stem=stem,
        encoding=chosen_encoding,
    )
    vertical_observed_arr = (
        np.asarray(vertical_observed, dtype=bool)
        if vertical_observed is not None
        else (free | occupied)
    )
    vertical_free_arr = np.asarray(vertical_free, dtype=bool) if vertical_free is not None else free
    if vertical_observed_arr.shape != free.shape or vertical_free_arr.shape != free.shape:
        raise ValueError("vertical_observed and vertical_free must match ROSE2 input shape")
    wall_conf = (
        np.asarray(wall_confidence_map, dtype=np.float32)
        if wall_confidence_map is not None
        else occupied.astype(np.float32)
    )
    if wall_conf.shape != free.shape:
        raise ValueError("wall_confidence_map must match ROSE2 input shape")

    free_png = out / ("%s.vertical_free.png" % stem)
    occupied_png = out / ("%s.vertical_occupied.png" % stem)
    unknown_png = out / ("%s.unknown.png" % stem)
    overlay_png = out / ("%s.overlay.png" % stem)
    npz_path = out / ("%s.input_masks.npz" % stem)
    summary_path = out / ("%s.input_summary.json" % stem)
    _bool_image(free, true_color=(255, 255, 255)).save(free_png)
    _bool_image(occupied, true_color=(255, 255, 255)).save(occupied_png)
    _bool_image(unknown_arr, true_color=(160, 160, 160)).save(unknown_png)
    _input_overlay(free=free, occupied=occupied, unknown=unknown_arr).save(overlay_png)
    np.savez_compressed(
        npz_path,
        observed_free=free.astype(np.uint8),
        observed_occupied=occupied.astype(np.uint8),
        unknown=unknown_arr.astype(np.uint8),
        vertical_observed=vertical_observed_arr.astype(np.uint8),
        vertical_free=vertical_free_arr.astype(np.uint8),
        vertical_free_room_domain=vertical_free_arr.astype(np.uint8),
        wall_confidence_map=wall_conf.astype(np.float32),
        resolution_m=np.asarray(float(resolution_m), dtype=np.float32),
    )
    summary = {
        "shape": [int(free.shape[0]), int(free.shape[1])],
        "free_cells": int(np.count_nonzero(free)),
        "occupied_cells": int(np.count_nonzero(occupied)),
        "unknown_cells": int(np.count_nonzero(unknown_arr)),
        "vertical_observed_cells": int(np.count_nonzero(vertical_observed_arr)),
        "vertical_free_cells": int(np.count_nonzero(vertical_free_arr)),
        "free_occupied_overlap": int(corrections["free_occupied_overlap"]),
        "free_unknown_overlap": int(corrections["free_unknown_overlap"]),
        "occupied_unknown_overlap": int(corrections["occupied_unknown_overlap"]),
        "corrected_free_cells": int(corrections["corrected_free_cells"]),
        "corrected_unknown_cells": int(corrections["corrected_unknown_cells"]),
        "resolution_m": float(resolution_m),
        "chosen_encoding": chosen_encoding,
        "encoding_candidates": candidates,
        "metric_map_path": str(metric_paths["metric_map"]),
        "orebro_input_path": str(metric_paths["orebro_input"]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ROSE2ExportBundle(
        work_dir=out,
        metric_map_path=metric_paths["metric_map"],
        orebro_input_path=metric_paths["orebro_input"],
        free_png_path=free_png,
        occupied_png_path=occupied_png,
        unknown_png_path=unknown_png,
        overlay_png_path=overlay_png,
        npz_path=npz_path,
        summary_json_path=summary_path,
        chosen_encoding=chosen_encoding,
        shape=(int(free.shape[0]), int(free.shape[1])),
        encoding_candidates=candidates,
        input_summary=summary,
    )


def choose_best_rose2_encoding(
    *,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    requested_encoding: str = "auto",
) -> tuple[str, list[dict]]:
    requested = str(requested_encoding or "auto").strip()
    candidates = list(ROSE2_ENCODING_CANDIDATES) if requested == "auto" else [requested]
    free, occupied, unknown_arr, _ = normalize_rose2_masks(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
    )
    rows = []
    for name in candidates:
        if name not in ROSE2_ENCODING_CANDIDATES:
            raise ValueError("unsupported ROSE2 upstream encoding: %s" % name)
        image = _metric_array_for_encoding(free=free, occupied=occupied, unknown=unknown_arr, encoding=name)
        precheck = _encoding_precheck(image=image, occupied=occupied, free=free)
        precheck["name"] = name
        rows.append(precheck)
    rows = sorted(rows, key=lambda item: (float(item.get("score", 0.0)), int(item.get("line_count", 0))), reverse=True)
    if not rows or float(rows[0].get("score", 0.0)) <= 0.0:
        raise RuntimeError("all ROSE2 upstream encodings failed cheap structure precheck")
    return str(rows[0]["name"]), rows


def normalize_rose2_masks(
    *,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    free = np.asarray(observed_free, dtype=bool).copy()
    occupied = np.asarray(observed_occupied, dtype=bool).copy()
    unknown_arr = np.asarray(unknown, dtype=bool).copy()
    if free.shape != occupied.shape or free.shape != unknown_arr.shape:
        raise ValueError("ROSE2 upstream masks must have the same HxW shape")
    overlaps = {
        "free_occupied_overlap": int(np.count_nonzero(free & occupied)),
        "free_unknown_overlap": int(np.count_nonzero(free & unknown_arr)),
        "occupied_unknown_overlap": int(np.count_nonzero(occupied & unknown_arr)),
    }
    # Strict priority for ambiguous cells: occupied > free > unknown.
    free_before = int(np.count_nonzero(free))
    unknown_before = int(np.count_nonzero(unknown_arr))
    free &= ~occupied
    unknown_arr &= ~(occupied | free)
    overlaps["corrected_free_cells"] = int(max(0, free_before - int(np.count_nonzero(free))))
    overlaps["corrected_unknown_cells"] = int(max(0, unknown_before - int(np.count_nonzero(unknown_arr))))
    return free, occupied, unknown_arr, overlaps


def rose2_input_cache_key(
    *,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    encoding: str,
    parameter_overrides: Mapping[str, object] | None = None,
) -> str:
    free, occupied, unknown_arr, _ = normalize_rose2_masks(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
    )
    h = hashlib.sha256()
    for arr in (free, occupied, unknown_arr):
        h.update(np.ascontiguousarray(arr.astype(np.uint8)).tobytes())
    h.update(str(encoding or "auto").encode("utf-8"))
    h.update(json.dumps(dict(parameter_overrides or {}), sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def load_rose2_upstream_input_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(Path(path))
    free = np.asarray(data.get("observed_free", data.get("vertical_free_room_domain", data.get("free"))), dtype=bool)
    occupied = np.asarray(data.get("observed_occupied", data.get("repaired_roomseg_occupied", data.get("occupied", np.zeros_like(free)))), dtype=bool)
    unknown = np.asarray(data.get("unknown", data.get("repaired_roomseg_unknown", np.zeros_like(free))), dtype=bool)
    vertical_observed = np.asarray(data.get("vertical_observed", free | occupied), dtype=bool)
    vertical_free = np.asarray(data.get("vertical_free", data.get("vertical_free_room_domain", free)), dtype=bool)
    wall_conf = np.asarray(data.get("wall_confidence_map", occupied.astype(np.float32)), dtype=np.float32)
    resolution = float(np.asarray(data.get("resolution_m", np.asarray(0.05, dtype=np.float32))).reshape(()))
    return occupied, free, unknown, vertical_observed, vertical_free, wall_conf, resolution


def _metric_array_for_encoding(*, free: np.ndarray, occupied: np.ndarray, unknown: np.ndarray, encoding: str) -> np.ndarray:
    name = str(encoding)
    if name == "source_black_wall_white_free":
        image = np.full(free.shape, 127, dtype=np.uint8)
        image[free] = 255
        image[unknown] = 127
        image[occupied] = 0
        return image
    if name == "source_white_wall_black_free":
        image = np.full(free.shape, 127, dtype=np.uint8)
        image[free] = 0
        image[unknown] = 127
        image[occupied] = 255
        return image
    if name == "source_metric_binary_with_unknown_border":
        image = np.full(free.shape, 127, dtype=np.uint8)
        image[free] = 255
        image[unknown] = 127
        image[occupied] = 0
        if np.any(unknown):
            image[0, :] = 127
            image[-1, :] = 127
            image[:, 0] = 127
            image[:, -1] = 127
        return image
    raise ValueError("unsupported ROSE2 upstream encoding: %s" % name)


def _encoding_precheck(*, image: np.ndarray, occupied: np.ndarray, free: np.ndarray) -> dict:
    occ_count = int(np.count_nonzero(occupied))
    free_count = int(np.count_nonzero(free))
    total = max(1, int(image.size))
    occ_ratio = float(occ_count) / float(total)
    free_ratio = float(free_count) / float(total)
    line_count = _hough_line_count(image)
    component_count = _component_count(occupied)
    sane_ratio = 0.0005 <= occ_ratio <= 0.80 and 0.0005 <= free_ratio <= 0.95
    sane_lines = 0 < line_count < max(10000, total // 2)
    score = 0.0
    if sane_ratio:
        score += 0.35
    if component_count > 0:
        score += 0.15
    if sane_lines:
        score += 0.45
    score += min(0.05, float(component_count) / 2000.0)
    return {
        "line_count": int(line_count),
        "occupied_component_count": int(component_count),
        "occupied_ratio": float(occ_ratio),
        "free_ratio": float(free_ratio),
        "score": float(score),
    }


def _hough_line_count(image: np.ndarray) -> int:
    try:
        import cv2  # type: ignore

        edges = cv2.Canny(np.asarray(image, dtype=np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=12, minLineLength=8, maxLineGap=4)
        return 0 if lines is None else int(len(lines))
    except Exception:
        grad = np.zeros_like(image, dtype=bool)
        grad[1:, :] |= image[1:, :] != image[:-1, :]
        grad[:, 1:] |= image[:, 1:] != image[:, :-1]
        return int(np.count_nonzero(grad) // 8)


def _component_count(mask: np.ndarray) -> int:
    from isaac_bench.mapping.structure_extraction import connected_components

    return int(len(connected_components(np.asarray(mask, dtype=bool))))


def _bool_image(mask: np.ndarray, true_color: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((*np.asarray(mask).shape, 3), dtype=np.uint8)
    arr[np.asarray(mask, dtype=bool)] = np.asarray(true_color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _input_overlay(*, free: np.ndarray, occupied: np.ndarray, unknown: np.ndarray) -> Image.Image:
    arr = np.zeros((*free.shape, 3), dtype=np.uint8)
    arr[:, :] = (32, 34, 38)
    arr[unknown] = (88, 88, 88)
    arr[free] = (235, 235, 235)
    arr[occupied] = (0, 0, 0)
    return Image.fromarray(arr, mode="RGB")
