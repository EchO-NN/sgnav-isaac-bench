from __future__ import annotations

from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image

from isaac_bench.mapping.structure_extraction import connected_components


def parse_rose2_room_output(
    *,
    output_png: str | Path,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    min_room_area_m2: float,
    resolution_m: float,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    if free.shape != unknown_arr.shape:
        raise ValueError("observed_free and unknown must have the same HxW shape")
    raw_labels, raw_debug = _raw_labels_from_output(Path(output_png))
    labels, transform_debug = _align_labels_to_input(raw_labels, free)
    before_outside = int(np.count_nonzero((labels > 0) & ~free))
    before_unknown = int(np.count_nonzero((labels > 0) & unknown_arr))
    labels = labels.astype(np.int32, copy=True)
    labels[~free] = 0
    labels[unknown_arr] = 0
    expansion_debug: dict[str, int | str] = {"grayscale_seed_expansion": "not_used"}
    if str(raw_debug.get("foreground_mode", "")) == "grayscale_cells":
        labels, expansion_debug = _expand_grayscale_cell_seeds(labels, free & ~unknown_arr)
    before_area = _label_count(labels)
    labels, removed = _remove_small_and_relabel(labels, min_cells=max(1, int(round(float(min_room_area_m2) / max(float(resolution_m) ** 2, 1e-9)))))
    debug = {
        "output_png": str(output_png),
        **raw_debug,
        **transform_debug,
        **expansion_debug,
        "room_count_before_area_filter": int(before_area),
        "room_count_after_area_filter": int(_label_count(labels)),
        "small_room_components_removed": int(removed),
        "labels_removed_outside_free": int(before_outside),
        "labels_removed_in_unknown": int(before_unknown),
    }
    return labels.astype(np.int32), debug


def labels_only_on_free(labels: np.ndarray, observed_free: np.ndarray, unknown: np.ndarray | None = None) -> bool:
    arr = np.asarray(labels, dtype=np.int32)
    free = np.asarray(observed_free, dtype=bool)
    if np.any((arr > 0) & ~free):
        return False
    if unknown is not None and np.any((arr > 0) & np.asarray(unknown, dtype=bool)):
        return False
    return True


def _raw_labels_from_output(path: Path) -> tuple[np.ndarray, dict]:
    image = Image.open(path)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    quant = ((rgb.astype(np.uint16) + 8) // 16 * 16).clip(0, 255).astype(np.uint8)
    foreground, foreground_mode = _foreground_from_rose2_output(rgb=rgb, alpha=alpha)
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    next_label = 1
    foreground_components = 0
    for color in sorted({tuple(int(v) for v in row) for row in quant[foreground].reshape((-1, 3))}):
        color_mask = foreground & np.all(quant == np.asarray(color, dtype=np.uint8), axis=2)
        for comp in connected_components(color_mask):
            if not comp:
                continue
            foreground_components += 1
            for r, c in comp:
                labels[int(r), int(c)] = int(next_label)
            next_label += 1
    debug = {
        "raw_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        "raw_unique_color_count": int(len({tuple(int(v) for v in row) for row in rgb.reshape((-1, 3))})),
        "foreground_component_count": int(foreground_components),
        "raw_foreground_pixels": int(np.count_nonzero(foreground)),
        "foreground_mode": str(foreground_mode),
    }
    return labels, debug


def _foreground_from_rose2_output(*, rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(rgb, dtype=np.int16)
    visible = np.asarray(alpha, dtype=np.uint8) > 16
    spread = np.max(arr, axis=2) - np.min(arr, axis=2)
    grayscale_ratio = float(np.count_nonzero(spread <= 10)) / float(max(1, spread.size))
    mean = np.mean(arr, axis=2)
    mid_gray = (spread <= 10) & (mean > 24) & (mean < 232)
    mid_gray_ratio = float(np.count_nonzero(mid_gray & visible)) / float(max(1, spread.size))
    if grayscale_ratio >= 0.98 and mid_gray_ratio >= 0.001:
        return visible & mid_gray, "grayscale_cells"
    return visible & ~_background_color(rgb), "colored_rooms"


def _background_color(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int16)
    near_black = np.max(arr, axis=2) <= 24
    near_white = np.min(arr, axis=2) >= 232
    spread = np.max(arr, axis=2) - np.min(arr, axis=2)
    mean = np.mean(arr, axis=2)
    neutral_gray = (spread <= 10) & (mean >= 70) & (mean <= 190)
    return near_black | near_white | neutral_gray


def _align_labels_to_input(labels: np.ndarray, observed_free: np.ndarray) -> tuple[np.ndarray, dict]:
    target_shape = tuple(int(v) for v in observed_free.shape)
    candidates: list[tuple[str, np.ndarray]] = []
    base = np.asarray(labels, dtype=np.int32)
    variants = [
        ("identity", base),
        ("flipud", np.flipud(base)),
        ("fliplr", np.fliplr(base)),
        ("transpose", base.T),
        ("flipud_transpose", np.flipud(base.T)),
        ("fliplr_transpose", np.fliplr(base.T)),
    ]
    for name, arr in variants:
        candidates.append((name if arr.shape == target_shape else "%s_resize" % name, _resize_nearest(arr, target_shape)))
    free = np.asarray(observed_free, dtype=bool)
    scored = []
    for name, arr in candidates:
        mask = arr > 0
        inter = int(np.count_nonzero(mask & free))
        union = int(np.count_nonzero(mask | free))
        iou = float(inter) / float(max(1, union))
        scored.append((iou, name, arr))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_name, best = scored[0]
    return best.astype(np.int32), {
        "shape_transform": str(best_name),
        "foreground_observed_free_iou": float(best_iou),
        "candidate_shape_scores": [{"transform": name, "iou": float(iou)} for iou, name, _ in scored],
    }


def _resize_nearest(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.shape == shape:
        return arr.copy()
    h, w = shape
    src_h, src_w = arr.shape
    if src_h <= 0 or src_w <= 0:
        return np.zeros(shape, dtype=np.int32)
    rr = np.floor(np.arange(h, dtype=np.float64) * float(src_h) / float(h)).astype(np.int64).clip(0, src_h - 1)
    cc = np.floor(np.arange(w, dtype=np.float64) * float(src_w) / float(w)).astype(np.int64).clip(0, src_w - 1)
    return arr[np.ix_(rr, cc)].astype(np.int32)


def _expand_grayscale_cell_seeds(labels: np.ndarray, free: np.ndarray) -> tuple[np.ndarray, dict[str, int | str]]:
    seeds = np.asarray(labels, dtype=np.int32)
    free_arr = np.asarray(free, dtype=bool)
    out = np.zeros_like(seeds, dtype=np.int32)
    components_with_seed = 0
    components_without_seed = 0
    filled_cells = 0
    seed_cells = int(np.count_nonzero((seeds > 0) & free_arr))
    for comp in connected_components(free_arr):
        if not comp:
            continue
        seed_items = [(int(r), int(c), int(seeds[int(r), int(c)])) for r, c in comp if int(seeds[int(r), int(c)]) > 0]
        if not seed_items:
            components_without_seed += 1
            continue
        components_with_seed += 1
        labels_in_comp = sorted({label for _r, _c, label in seed_items})
        if len(labels_in_comp) == 1:
            label = int(labels_in_comp[0])
            for r, c in comp:
                out[int(r), int(c)] = label
            filled_cells += len(comp)
            continue
        comp_mask = np.zeros_like(free_arr, dtype=bool)
        for r, c in comp:
            comp_mask[int(r), int(c)] = True
        queue: deque[tuple[int, int]] = deque()
        for r, c, label in seed_items:
            if out[r, c] == 0:
                out[r, c] = int(label)
                queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            label = int(out[r, c])
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nr < 0 or nc < 0 or nr >= out.shape[0] or nc >= out.shape[1]:
                    continue
                if not comp_mask[nr, nc] or out[nr, nc] != 0:
                    continue
                out[nr, nc] = label
                queue.append((nr, nc))
        filled_cells += int(np.count_nonzero(out & comp_mask))
    return out, {
        "grayscale_seed_expansion": "used",
        "grayscale_seed_cells": int(seed_cells),
        "grayscale_components_with_seed": int(components_with_seed),
        "grayscale_components_without_seed": int(components_without_seed),
        "grayscale_expanded_cells": int(filled_cells),
    }


def _remove_small_and_relabel(labels: np.ndarray, min_cells: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    next_label = 1
    removed = 0
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        for comp in connected_components(arr == label_id):
            if len(comp) < int(min_cells):
                removed += 1
                continue
            for r, c in comp:
                out[int(r), int(c)] = int(next_label)
            next_label += 1
    return out, removed


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))
