from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np
from PIL import Image

from isaac_bench.mapping.rose2_source_form import (
    ROSE2SourceResult,
    SOURCE_EXTERNAL_RUNNER_BACKEND,
    export_rose2_source_input,
    parse_rose2_source_label_image,
)


REQUIRED_EXTERNAL_FILES = (
    "code/FFT_MQ.py",
    "code/minibatch.py",
    "code/parameters.py",
    "code/runMe.py",
    "code/util/layout.py",
    "code/util/postprocessing.py",
)


def run_rose2_source_external_runner(
    *,
    source_root: str | Path | None,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    work_dir: str | Path,
    timeout_s: float = 60.0,
) -> ROSE2SourceResult:
    out = Path(work_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    export_paths = export_rose2_source_input(
        out_dir=out,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown_arr,
        stem="external_source_input",
    )
    orebro = out / "input_orebro.png"
    Image.open(export_paths["metric_map_png"]).save(orebro)
    root = Path(source_root).expanduser() if source_root else None
    failure_reason = ""
    missing = []
    if root is None or not root.exists():
        failure_reason = "missing_rose2_source_root"
    else:
        missing = [name for name in REQUIRED_EXTERNAL_FILES if not (root / name).exists()]
        if missing:
            failure_reason = "missing_required_source_files"

    wrapper = out / "run_external_rose2.py"
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    label_map = np.zeros_like(occupied, dtype=np.int32)
    command: list[str] = []
    return_code = None
    output_room_image = None
    if not failure_reason:
        wrapper.write_text(_wrapper_source(root, Path(export_paths["metric_map_png"]), orebro, out), encoding="utf-8")
        command = [sys.executable, str(wrapper)]
        try:
            proc = subprocess.run(
                command,
                cwd=str(root / "code"),
                text=True,
                capture_output=True,
                timeout=float(timeout_s),
                check=False,
            )
            return_code = int(proc.returncode)
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            room_images = sorted(out.glob("**/*rooms*.png"))
            if room_images:
                output_room_image = room_images[0]
                label_map = parse_rose2_source_label_image(output_room_image)
                if proc.returncode != 0:
                    failure_reason = "external_runner_partial_success_after_room_image"
                else:
                    failure_reason = ""
            else:
                failure_reason = "external_runner_failed_or_no_room_image"
        except Exception as exc:
            failure_reason = "external_runner_exception:%s" % type(exc).__name__
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
    else:
        wrapper.write_text("# external runner was not launched: %s\n" % failure_reason, encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("missing=%s\n" % missing, encoding="utf-8")

    summary = {
        "backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_root": str(root) if root else None,
        "required_files_present": not bool(missing),
        "missing_required_files": missing,
        "command": command,
        "return_code": return_code,
        "failure_reason": failure_reason,
        "export_paths": export_paths,
        "input_orebro_png": str(orebro),
        "wrapper": str(wrapper),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output_room_image": str(output_room_image) if output_room_image else None,
        "source_room_count": _label_count(label_map),
    }
    (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _label_image(label_map).save(out / "external_room_label_map.png")
    debug = {
        "source_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_exact_used": bool(output_room_image and not failure_reason),
        "external_runner_summary": summary,
        "external_failure_reason": failure_reason,
        "source_room_count": int(_label_count(label_map)),
        "proposal_room_count": int(_label_count(label_map)),
        "legacy_connected_component_rooms_used": False,
        "failure_mode": failure_reason,
    }
    return ROSE2SourceResult(
        backend=SOURCE_EXTERNAL_RUNNER_BACKEND,
        room_label_map=label_map.astype(np.int32),
        source_room_label_map=label_map.astype(np.int32),
        clean_structure_map=occupied.astype(bool),
        structural_score=occupied.astype(np.float32),
        boundary_map=occupied.astype(bool),
        dominant_directions_rad=[],
        hough_segments=[],
        wall_clusters=[],
        representative_lines=[],
        extended_lines=[],
        faces=[],
        face_adjacency_edges=[],
        cell_edges=[],
        cell_polygons=[],
        timing_ms={},
        debug=debug,
    )


def _wrapper_source(source_root: Path, metric_map: Path, orebro: Path, out: Path) -> str:
    return f"""from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path
import numpy as np

source_root = Path({str(source_root)!r})
metric_map = Path({str(metric_map)!r})
orebro_seed = Path({str(orebro)!r})
out = Path({str(out)!r})
shim = out / 'png.py'
shim.write_text('''from __future__ import annotations
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
''', encoding='utf-8')
sys.path.insert(0, str(out))
sys.path.insert(0, str(source_root / 'code' / 'rose_v1_repo'))
sys.path.insert(0, str(source_root / 'code'))
try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib.backend_bases import FigureCanvasBase
    if not hasattr(FigureCanvasBase, 'set_window_title'):
        FigureCanvasBase.set_window_title = lambda self, title: None
except Exception:
    pass
try:
    import skimage.morphology as _sk_morph
    _orig_binary_dilation = _sk_morph.binary_dilation

    def _binary_dilation_compat(image, footprint=None, *args, selem=None, **kwargs):
        if footprint is None and selem is not None:
            footprint = selem
        return _orig_binary_dilation(image, footprint=footprint, *args, **kwargs)

    _sk_morph.binary_dilation = _binary_dilation_compat
except Exception:
    pass
try:
    import skan
    import skan.csr as _skan_csr
    from skan.csr import skeleton_to_csgraph as _skan_skeleton_to_csgraph

    def _skeleton_to_csgraph_compat(*args, **kwargs):
        result = _skan_skeleton_to_csgraph(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            graph, coordinates = result
            degrees = np.asarray(graph.sum(axis=1)).reshape(-1)
            return graph, coordinates, degrees
        return result

    skan.skeleton_to_csgraph = _skeleton_to_csgraph_compat
    _OrigSkeleton = _skan_csr.Skeleton

    class _SkeletonCompat:
        def __init__(self, skeleton, *args, **kwargs):
            self._skeleton_image = np.asarray(skeleton, dtype=bool)
            self._inner = _OrigSkeleton(skeleton, *args, **kwargs)
            self.degrees_image = np.zeros(self._skeleton_image.shape, dtype=np.uint8)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    _skan_csr.Skeleton = _SkeletonCompat
    skan.Skeleton = _SkeletonCompat
except Exception:
    pass
try:
    import networkx as _nx
    if not hasattr(_nx, 'from_scipy_sparse_matrix') and hasattr(_nx, 'from_scipy_sparse_array'):
        _nx.from_scipy_sparse_matrix = _nx.from_scipy_sparse_array
except Exception:
    pass
try:
    import cv2
    _orig_find_contours = cv2.findContours

    def _find_contours_compat(*args, **kwargs):
        result = _orig_find_contours(*args, **kwargs)
        if len(result) == 2:
            contours, hierarchy = result
            contours = list(contours)
            image = args[0] if args else None
            return image, contours, hierarchy
        if len(result) == 3:
            image, contours, hierarchy = result
            return image, list(contours), hierarchy
        return result

    cv2.findContours = _find_contours_compat
    _orig_imwrite = cv2.imwrite

    def _imwrite_compat(filename, image, params=None):
        if params is not None and len(params) % 2:
            params = []
        if params is None:
            return _orig_imwrite(filename, image)
        return _orig_imwrite(filename, image, params)

    cv2.imwrite = _imwrite_compat
    cv2.__version__ = '3.4.0'
except Exception:
    pass
import parameters as par
import FFT_MQ as fft
import minibatch
try:
    import util.disegna as _rose_draw
    from matplotlib.patches import Polygon as _MplPolygon

    def _polygon_patch_compat(polygon, **kwargs):
        geom = polygon
        if hasattr(geom, 'geoms'):
            geoms = [g for g in geom.geoms if hasattr(g, 'exterior') and not g.is_empty]
            geom = max(geoms, key=lambda g: float(g.area), default=None)
        if geom is not None and hasattr(geom, 'exterior'):
            coords = np.asarray(geom.exterior.coords, dtype=float)
        else:
            coords = np.asarray(geom, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3:
            coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
        return _MplPolygon(coords[:, :2], closed=True, **kwargs)

    _rose_draw.PolygonPatch = _polygon_patch_compat
except Exception:
    pass
try:
    import util.layout as _rose_layout
    from sklearn.cluster import DBSCAN as _SklearnDBSCAN
    _orig_external_contour = _rose_layout.external_contour

    class _DBSCANCompat(_SklearnDBSCAN):
        def __init__(self, eps=0.5, min_samples=5, **kwargs):
            super().__init__(eps=eps, min_samples=min_samples, **kwargs)

    def _external_contour_compat(img_rgb):
        try:
            contours, vertices = _orig_external_contour(img_rgb)
            if len(vertices) >= 3:
                return contours, vertices
        except Exception:
            pass
        h, w = img_rgb.shape[:2]
        vertices = [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]]
        contour = np.asarray([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
        return contour, vertices

    _rose_layout.external_contour = _external_contour_compat
    _rose_layout.DBSCAN = _DBSCANCompat
except Exception:
    pass

params = par.ParameterObj()
paths = par.PathObj()
paths.metric_map_name = 'codex_metric_map'
paths.metric_map_path = str(metric_map)
paths.path_folder_output = str(out)
paths.path_log_folder = str(out)
paths.filepath = str(out) + os.sep
paths.path_orebro = str(out / 'OREBRO')
paths.orebro_img = str(out / ('OREBRO_' + str(params.filter_level) + '.png'))
paths.gt_color = ''
Path(paths.path_orebro).mkdir(parents=True, exist_ok=True)
fft.main(paths.metric_map_path, paths.path_orebro, params.filter_level, params)
if not Path(paths.orebro_img).exists():
    shutil.copy(str(orebro_seed), paths.orebro_img)
minibatch.start_main(par, params, paths)
print('external_rose2_done', paths.filepath)
"""


def _label_image(labels: np.ndarray) -> Image.Image:
    arr = np.asarray(labels, dtype=np.int32)
    canvas = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        canvas[arr == label_id] = _palette(label_id)
    return Image.fromarray(canvas)


def _palette(label_id: int) -> tuple[int, int, int]:
    palette = [(126, 174, 255), (255, 156, 102), (130, 222, 150), (214, 148, 255), (250, 216, 95)]
    return palette[(int(label_id) - 1) % len(palette)]


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _load_input_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    occupied = np.asarray(data.get("observed_occupied", data.get("occupied")), dtype=bool)
    free = np.asarray(data.get("observed_free", data.get("free")), dtype=bool)
    unknown = np.asarray(data.get("unknown", np.zeros_like(free)), dtype=bool)
    return occupied, free, unknown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    occupied, free, unknown = _load_input_npz(Path(args.input_npz))
    result = run_rose2_source_external_runner(
        source_root=args.source_root,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        work_dir=args.work_dir,
        timeout_s=float(args.timeout_s),
    )
    print(json.dumps(result.debug.get("external_runner_summary", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
