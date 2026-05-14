from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from isaac_bench.dataset.interioragent_index import discover_scenes


def _usd_lib_paths(isaac_sim_root: str) -> Optional[tuple[str, str]]:
    root = Path(isaac_sim_root).expanduser().resolve()
    libs = sorted(root.glob("extscache/omni.usd.libs-*"))
    semantics = sorted(root.glob("extscache/omni.usd.schema.semantics-*"))
    if not libs:
        return None
    py_paths = [str(libs[0])]
    if semantics:
        py_paths.append(str(semantics[0]))
    return os.pathsep.join(py_paths), str(libs[0] / "bin")


def ensure_pxr_or_reexec(isaac_sim_root: str) -> None:
    try:
        from pxr import Usd  # noqa: F401
        return
    except Exception:
        pass
    if os.environ.get("ISAAC_BENCH_REEXEC_USD") == "1":
        raise RuntimeError("pxr.Usd is still unavailable after Isaac Kit Python re-exec")
    paths = _usd_lib_paths(isaac_sim_root)
    if paths is None:
        raise RuntimeError("Cannot locate omni.usd.libs under %s" % isaac_sim_root)
    py_path, ld_path = paths
    root = Path(isaac_sim_root).expanduser().resolve()
    kit_python = root / "python.sh"
    if not kit_python.exists():
        kit_python = root / "kit" / "python" / "bin" / "python3"
    env = os.environ.copy()
    env["ISAAC_BENCH_REEXEC_USD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), py_path, env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([ld_path, env.get("LD_LIBRARY_PATH", "")])
    cmd = [str(kit_python)] + sys.argv
    raise SystemExit(subprocess.call(cmd, env=env, cwd=str(Path.cwd())))


def preprocess_scene(scene, out_root: Path, args) -> dict:
    from isaac_bench.dataset.occupancy_builder import save_preprocessed_scene
    from isaac_bench.dataset.usd_semantics import STRUCTURAL_CATEGORIES, extract_objects_from_usd, write_objects_json
    from isaac_bench.mapping.room_map_from_rooms_json import load_rooms

    scene_out = out_root / scene.scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    rooms = load_rooms(scene.rooms_json_path)
    objects_all = extract_objects_from_usd(
        scene.usd_path,
        rooms=rooms,
        min_bbox_diag_m=args.min_object_bbox_diag_m,
        include_structural=True,
    )
    excluded = set(args.exclude_category or []) | set(STRUCTURAL_CATEGORIES)
    objects = [obj for obj in objects_all if obj.category not in excluded]
    write_objects_json(objects_all, str(scene_out / "objects_all.json"))
    write_objects_json(objects, str(scene_out / "objects.json"))
    with open(scene_out / "rooms.json", "w", encoding="utf-8") as handle:
        json.dump([room.to_dict() for room in rooms], handle, ensure_ascii=False, indent=2)
    shutil.copyfile(scene.rooms_json_path, scene_out / "rooms_raw.json")
    map_info = save_preprocessed_scene(
        str(scene_out),
        [obj.to_dict() for obj in objects_all],
        rooms,
        args.resolution,
        args.robot_radius_m,
        args.inflation_radius_m,
        args.obstacle_min_height_m,
        args.obstacle_max_height_m,
    )
    metadata = {
        "scene_id": scene.scene_id,
        "scene_dir": scene.scene_dir,
        "usd_path": scene.usd_path,
        "rooms_json_path": scene.rooms_json_path,
        "num_objects_all": len(objects_all),
        "num_objects": len(objects),
        "map_info": map_info.to_dict(),
    }
    with open(scene_out / "scene_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/echo/InteriorAgent")
    parser.add_argument("--out", default="data/interioragent_preprocessed")
    parser.add_argument("--isaac-sim-root", default="/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64")
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--scene-glob", default="kujiale_*")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--robot-radius-m", type=float, default=0.14)
    parser.add_argument(
        "--inflation-radius-m",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; navigability inflation is footprint-only.",
    )
    parser.add_argument("--min-object-bbox-diag-m", type=float, default=0.05)
    parser.add_argument("--obstacle-min-height-m", type=float, default=0.05)
    parser.add_argument("--obstacle-max-height-m", type=float, default=1.50)
    parser.add_argument("--exclude-category", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_pxr_or_reexec(args.isaac_sim_root)
    scenes = discover_scenes(args.dataset_root, args.scene_glob, args.scene_id)
    if args.limit is not None:
        scenes = scenes[: args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes:
        print("[preprocess] %s" % scene.scene_id, flush=True)
        rows.append(preprocess_scene(scene, out_root, args))
    with open(out_root / "index.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print("[preprocess] wrote %d scenes to %s" % (len(rows), out_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
