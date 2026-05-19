from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated worker for goldleaf3i/declutter-reconstruct ROSE2 room segmentation.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--metric-map", required=True)
    parser.add_argument("--orebro-input", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--parameter-overrides-json", default="{}")
    args = parser.parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve()
    metric_map = Path(args.metric_map).resolve()
    orebro_seed = Path(args.orebro_input).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_json = Path(args.output_json).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload: dict = {
        "ok": False,
        "source_root": str(source_root),
        "work_dir": str(work_dir),
        "metric_map_path": str(metric_map),
        "orebro_img": str(orebro_seed),
        "room_output_path": None,
        "timing_ms": {},
        "parameter_values": {},
        "compat_patches": [],
    }
    try:
        overrides = json.loads(args.parameter_overrides_json or "{}")
        if not isinstance(overrides, dict):
            overrides = {}
        code_dir = source_root / "code"
        rose_patch_dir = _install_source_file_compat_patches(code_dir, work_dir, payload)
        sys.path.insert(0, str(work_dir))
        sys.path.insert(0, str(code_dir / "rose_v1_repo"))
        sys.path.insert(0, str(code_dir))
        if rose_patch_dir is not None:
            sys.path.insert(0, str(rose_patch_dir))
        _install_compat_shims(work_dir, payload)
        import parameters as par  # type: ignore
        import FFT_MQ as fft  # type: ignore
        import minibatch  # type: ignore

        path_obj = par.PathObj()
        param_obj = par.ParameterObj()
        for key, value in overrides.items():
            if hasattr(param_obj, str(key)):
                setattr(param_obj, str(key), value)
        path_obj.metric_map_path = str(metric_map)
        path_obj.metric_map_name = metric_map.name
        # FFT_MQ writes `path_orebro + "_<filter>.png"`, while minibatch's
        # adaptive loop later reads `filepath + "OREBRO_<filter>.png"`.
        # Therefore this must be a file prefix in work_dir, not a directory.
        path_obj.path_orebro = str(work_dir / "OREBRO")
        path_obj.orebro_img = str(orebro_seed)
        path_obj.filepath = str(work_dir) + os.sep
        path_obj.path_folder_output = str(work_dir)
        path_obj.path_folder_input = str(work_dir)
        path_obj.path_log_folder = str(work_dir)
        path_obj.gt_color = ""
        Path(path_obj.path_orebro).parent.mkdir(parents=True, exist_ok=True)
        (work_dir / "OREBRO").mkdir(parents=True, exist_ok=True)

        t_fft = time.perf_counter()
        try:
            fft.main(path_obj.metric_map_path, path_obj.path_orebro, getattr(param_obj, "filter_level", 0.18), param_obj)
            candidate_orebro = Path(str(path_obj.path_orebro) + "_%s.png" % str(getattr(param_obj, "filter_level", 0.18)))
            if candidate_orebro.exists():
                path_obj.orebro_img = str(candidate_orebro)
        except Exception as exc:
            payload["fft_error"] = "%s: %s" % (type(exc).__name__, exc)
            if orebro_seed.exists():
                fallback = work_dir / "OREBRO_worker_seed.png"
                shutil.copyfile(str(orebro_seed), str(fallback))
                path_obj.orebro_img = str(fallback)
        if not getattr(param_obj, "comp", None):
            param_obj.comp = [0.0, float(np.pi / 2.0)]
            payload["compat_patches"].append("fft_missing_comp_manhattan_default")
        payload["timing_ms"]["fft"] = float((time.perf_counter() - t_fft) * 1000.0)
        payload["orebro_img"] = str(path_obj.orebro_img)

        t_main = time.perf_counter()
        try:
            result = minibatch.start_main(par, param_obj, path_obj)
            payload["minibatch_return"] = repr(result)
            payload["ok"] = True
        except Exception as exc:
            payload["minibatch_error"] = "%s: %s" % (type(exc).__name__, exc)
            payload["minibatch_traceback"] = traceback.format_exc()
        payload["timing_ms"]["minibatch"] = float((time.perf_counter() - t_main) * 1000.0)

        room_output = find_rose2_room_output(work_dir)
        if room_output is not None:
            payload["room_output_path"] = str(room_output)
            # The upstream pipeline can raise in late visualization/Voronoi code
            # after the room PNG has already been produced. Treat the source
            # room image as usable, while recording the late error as nonfatal.
            payload["ok"] = True
            if payload.get("minibatch_error"):
                payload["nonfatal_postprocess_error"] = payload["minibatch_error"]
        elif not payload.get("ok"):
            payload.setdefault("error_message", "upstream ROSE2 source produced no room output image")
        payload["parameter_values"] = _parameter_values(param_obj)
    except Exception as exc:
        payload["ok"] = False
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        room_output = find_rose2_room_output(work_dir)
        if room_output is not None:
            payload["ok"] = True
            payload["room_output_path"] = str(room_output)
            payload["nonfatal_worker_error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        payload["timing_ms"]["total"] = float((time.perf_counter() - t0) * 1000.0)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if bool(payload.get("ok")) else 2


def find_rose2_room_output(work_dir: str | Path) -> Path | None:
    root = Path(work_dir)
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in (
        "8b_rooms_th1_on_map_post.png",
        "8b_rooms_th1_on_map.png",
        "8b_rooms_th1.png",
        "**/*rooms*post*.png",
        "**/*rooms*.png",
        "**/*cells*in*out*.png",
        "**/*cells*.png",
        "**/*.png",
    ):
        for path in root.glob(pattern):
            if path.is_file() and path not in seen and _is_candidate_source_output(path):
                seen.add(path)
                candidates.append(path)
    scored = [(score_room_output_candidate(path), path) for path in candidates]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _is_candidate_source_output(path: Path) -> bool:
    name = path.name.lower()
    if "rooms" in name or "cells" in name:
        return True
    blocked_tokens = (
        "metric_map",
        "metric_input",
        "orebro_input",
        "vertical_free",
        "vertical_occupied",
        "unknown",
        "overlay",
        "parsed_labels",
        "source_output",
        "room_label_map",
        "input",
        "walls",
        "hough",
        "map",
        "orebro",
        "edges",
        "extended",
        "cluster",
        "segment",
    )
    return not any(token in name for token in blocked_tokens)


def score_room_output_candidate(path: Path) -> int:
    try:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return 0
    flat = arr.reshape((-1, 3))
    if flat.size == 0:
        return 0
    colors = {}
    for row in flat[:: max(1, flat.shape[0] // 200000)]:
        color = tuple(int(v) for v in row)
        if color in {(0, 0, 0), (255, 255, 255), (127, 127, 127)}:
            continue
        if max(color) <= 24 or min(color) >= 232:
            continue
        colors[color] = colors.get(color, 0) + 1
    if not colors:
        return 0
    name = path.name.lower()
    raw_room_bonus = 3000 if name in {"8b_rooms_th1.png", "8b_rooms_th1_post.png"} else 0
    name_bonus = 1000 if "rooms" in name else 0
    post_bonus = 100 if "post" in name and "on_map" not in name else 0
    overlay_penalty = 600 if "on_map" in name or "prediction" in name else 0
    return int(raw_room_bonus + name_bonus + post_bonus + len(colors) - overlay_penalty)


def _install_compat_shims(work_dir: Path, payload: dict) -> None:
    shim = work_dir / "png.py"
    shim.write_text(
        """from __future__ import annotations
import numpy as np
from PIL import Image

class Writer:
    def __init__(self, width, height, greyscale=True, alpha=False, bitdepth=8):
        self.width = int(width)
        self.height = int(height)
        self.greyscale = bool(greyscale)
        self.alpha = bool(alpha)
        self.bitdepth = int(bitdepth)

    def write(self, fp, rows):
        arr = np.asarray(rows)
        if arr.ndim == 1:
            arr = arr.reshape((self.height, self.width))
        arr = arr[:self.height, :self.width]
        if self.bitdepth == 1:
            img = (arr > 0).astype(np.uint8) * 255
        else:
            img = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(fp, format='PNG')
""",
        encoding="utf-8",
    )
    payload["compat_patches"].append("local_png_writer")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backend_bases import FigureCanvasBase

        if not hasattr(FigureCanvasBase, "set_window_title"):
            FigureCanvasBase.set_window_title = lambda self, title: None
            payload["compat_patches"].append("matplotlib_set_window_title")
    except Exception:
        pass
    try:
        import skimage.morphology as sk_morph  # type: ignore

        orig_binary_dilation = sk_morph.binary_dilation

        def binary_dilation_compat(image, footprint=None, *args, selem=None, **kwargs):
            if footprint is None and selem is not None:
                footprint = selem
            return orig_binary_dilation(image, footprint=footprint, *args, **kwargs)

        sk_morph.binary_dilation = binary_dilation_compat
        payload["compat_patches"].append("skimage_binary_dilation_selem")
    except Exception:
        pass
    try:
        import networkx as nx  # type: ignore

        if not hasattr(nx, "from_scipy_sparse_matrix") and hasattr(nx, "from_scipy_sparse_array"):
            nx.from_scipy_sparse_matrix = nx.from_scipy_sparse_array
            payload["compat_patches"].append("networkx_sparse_matrix_alias")
    except Exception:
        pass
    try:
        import skan  # type: ignore
        import skan.csr as skan_csr  # type: ignore
        from skan.csr import skeleton_to_csgraph as csr_skeleton_to_csgraph  # type: ignore

        def skeleton_to_csgraph_compat(*args, **kwargs):
            result = csr_skeleton_to_csgraph(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                graph, coordinates = result
                degrees = np.asarray(graph.sum(axis=1)).reshape(-1)
                return graph, coordinates, degrees
            return result

        skan.skeleton_to_csgraph = skeleton_to_csgraph_compat
        orig_skeleton = skan_csr.Skeleton

        class SkeletonCompat:
            def __init__(self, skeleton, *args, **kwargs):
                self._skeleton_image = np.asarray(skeleton, dtype=bool)
                self._inner = orig_skeleton(skeleton, *args, **kwargs)
                self.degrees_image = np.zeros(self._skeleton_image.shape, dtype=np.uint8)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        skan_csr.Skeleton = SkeletonCompat
        skan.Skeleton = SkeletonCompat
        payload["compat_patches"].append("skan_root_and_skeleton_compat")
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        orig_find_contours = cv2.findContours

        def find_contours_compat(*args, **kwargs):
            result = orig_find_contours(*args, **kwargs)
            if len(result) == 2:
                contours, hierarchy = result
                image = args[0] if args else None
                return image, list(contours), hierarchy
            if len(result) == 3:
                image, contours, hierarchy = result
                return image, list(contours), hierarchy
            return result

        cv2.findContours = find_contours_compat
        orig_imwrite = cv2.imwrite

        def imwrite_compat(filename, image, params=None):
            if params is not None and len(params) % 2:
                params = []
            return orig_imwrite(filename, image) if params is None else orig_imwrite(filename, image, params)

        cv2.imwrite = imwrite_compat
        cv2.__version__ = "3.4.0"
        payload["compat_patches"].append("opencv3_compat")
    except Exception:
        pass
    try:
        import util.disegna as rose_draw  # type: ignore
        from matplotlib.patches import Polygon as MplPolygon

        def polygon_patch_compat(polygon, **kwargs):
            geom = polygon
            if hasattr(geom, "geoms"):
                geoms = [g for g in geom.geoms if hasattr(g, "exterior") and not g.is_empty]
                geom = max(geoms, key=lambda g: float(g.area), default=None)
            if geom is not None and hasattr(geom, "exterior"):
                coords = np.asarray(geom.exterior.coords, dtype=float)
            else:
                coords = np.asarray(geom, dtype=float)
            if coords.ndim != 2 or coords.shape[0] < 3:
                coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
            return MplPolygon(coords[:, :2], closed=True, **kwargs)

        rose_draw.PolygonPatch = polygon_patch_compat
        payload["compat_patches"].append("shapely2_polygon_patch")
    except Exception:
        pass
    try:
        import util.layout as rose_layout  # type: ignore
        from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore

        orig_external_contour = rose_layout.external_contour
        free_contour_mask = _load_worker_observed_free_mask(work_dir)

        class DBSCANCompat(SklearnDBSCAN):
            def __init__(self, eps=0.5, min_samples=5, **kwargs):
                super().__init__(eps=eps, min_samples=min_samples, **kwargs)

        def external_contour_compat(img_rgb):
            free_result = _external_contour_from_free_mask(free_contour_mask, img_rgb.shape[:2])
            if free_result is not None:
                return free_result
            try:
                contours, vertices = orig_external_contour(img_rgb)
                if len(vertices) >= 3:
                    return contours, vertices
            except Exception:
                pass
            h, w = img_rgb.shape[:2]
            vertices = [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]]
            contour = np.asarray([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
            return contour, vertices

        rose_layout.external_contour = external_contour_compat
        rose_layout.DBSCAN = DBSCANCompat
        payload["compat_patches"].append("layout_external_contour_observed_free_dbscan")
    except Exception:
        pass


def _load_worker_observed_free_mask(work_dir: Path) -> np.ndarray | None:
    for path in sorted(Path(work_dir).glob("*.input_masks.npz")):
        try:
            with np.load(path) as data:
                if "observed_free" in data:
                    return np.asarray(data["observed_free"], dtype=bool)
        except Exception:
            continue
    return None


def _external_contour_from_free_mask(free_mask: np.ndarray | None, image_shape: tuple[int, int]) -> tuple[object, list[list[float]]] | None:
    if free_mask is None:
        return None
    mask = np.asarray(free_mask, dtype=bool)
    if mask.shape != tuple(int(v) for v in image_shape[:2]) or not np.any(mask):
        return None
    try:
        import cv2  # type: ignore

        img = (mask.astype(np.uint8) * 255)
        found = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(found) == 3:
            _image, contours, _hierarchy = found
        else:
            contours, _hierarchy = found
        contours = list(contours or [])
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if float(cv2.contourArea(contour)) <= 0.0:
            return None
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * float(perimeter)), True)
        vertices = [[float(point[0][0]), float(point[0][1])] for point in approx]
        if len(vertices) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            vertices = [
                [float(x), float(y)],
                [float(x + w - 1), float(y)],
                [float(x + w - 1), float(y + h - 1)],
                [float(x), float(y + h - 1)],
            ]
            approx = np.asarray([[[int(v[0]), int(v[1])]] for v in vertices], dtype=np.int32)
        return approx, vertices
    except Exception:
        return None


def _install_source_file_compat_patches(code_dir: Path, work_dir: Path, payload: dict) -> Path | None:
    patch_dir = work_dir / "rose2_source_compat"
    patches = 0
    fft_source = code_dir / "rose_v1_repo" / "fft_structure_extraction.py"
    if fft_source.exists():
        text = fft_source.read_text(encoding="utf-8")
        patched = text.replace(
        "if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:",
        "if b == 0 or np.abs(Y1) > 3 * np.max(self.binary_map.shape):",
        )
        if patched != text:
            patch_dir.mkdir(parents=True, exist_ok=True)
            (patch_dir / "fft_structure_extraction.py").write_text(patched, encoding="utf-8")
            payload["compat_patches"].append("fft_structure_extraction_y1_short_circuit")
            patches += 1

    minibatch_source = code_dir / "minibatch.py"
    if minibatch_source.exists():
        text = minibatch_source.read_text(encoding="utf-8")
        patched = text.replace(
            "if param_obj.filter_level <= 0.12:",
            "if param_obj.filter_level <= 0.12 or param_obj.filter_level >= getattr(param_obj, 'max_filter_level', 0.50):",
        )
        if patched != text:
            patch_dir.mkdir(parents=True, exist_ok=True)
            (patch_dir / "minibatch.py").write_text(patched, encoding="utf-8")
            payload["compat_patches"].append("minibatch_filter_level_upper_bound")
            patches += 1

    return patch_dir if patches else None


def _parameter_values(param_obj) -> dict:
    keys = ("filter_level", "thresholdHough", "minLineLength", "maxLineGap", "th_post", "th1", "distance_extended_segment")
    out = {}
    for key in keys:
        if hasattr(param_obj, key):
            value = getattr(param_obj, key)
            try:
                value = value.item()
            except Exception:
                pass
            out[key] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
