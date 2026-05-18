from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from isaac_bench.config import get_nested, load_config
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import (
    DEFAULT_SOURCE_ENV,
    REQUIRED_SOURCE_FILES,
    validate_upstream_rose2_source_root,
)


def _path_from_env_or_config(env_names: Union[str, Sequence[str]], cfg: dict, config_key: str, default: str) -> str:
    names = [env_names] if isinstance(env_names, str) else list(env_names)
    for env_name in names:
        value = os.environ.get(env_name)
        if value:
            return value
    return str(get_nested(cfg, config_key, default) or default)


def _check_path(kind: str, path: str, required: bool, allow_config_reference: bool = False) -> dict:
    exists = bool(path) and Path(path).expanduser().exists()
    if not exists and allow_config_reference and bool(path):
        # SAM2 accepts package/Hydra-style config references such as
        # configs/sam2.1/sam2.1_hiera_s.yaml; they are validated on model load.
        exists = True
    return {
        "kind": kind,
        "path": path,
        "required": bool(required),
        "exists": bool(exists),
        "status": "ok" if exists else ("missing" if required else "optional_missing"),
    }


def _check_rose2_source(path: str, env_name: str, required: bool) -> dict:
    try:
        resolved = validate_upstream_rose2_source_root(path, env_name=env_name, fail=bool(required))
        exists = resolved is not None
        message = ""
    except FileNotFoundError as exc:
        resolved = None
        exists = False
        message = str(exc)
    return {
        "kind": "rose2_source_root",
        "path": str(resolved or path or os.environ.get(env_name, "")),
        "required": bool(required),
        "exists": bool(exists),
        "status": "ok" if exists else ("missing" if required else "optional_missing"),
        "env": env_name,
        "required_files": list(REQUIRED_SOURCE_FILES),
        "message": message,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check external SG-Nav Isaac benchmark assets.")
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--require-yolo-world", action="store_true")
    parser.add_argument("--require-grounding-dino", action="store_true")
    parser.add_argument("--require-sam2", action="store_true")
    parser.add_argument("--require-interioragent", action="store_true")
    parser.add_argument("--require-isaac", action="store_true")
    parser.add_argument("--require-rose2-source", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    checks = [
        _check_path(
            "isaac_sim_root",
            _path_from_env_or_config(
                ["ISAAC_SIM_ROOT", "ISAAC_ROOT"],
                cfg,
                "paths.isaac_sim_root",
                "/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64",
            ),
            args.require_isaac,
        ),
        _check_path(
            "interioragent_root",
            _path_from_env_or_config("INTERIORAGENT_ROOT", cfg, "paths.interioragent_root", "/home/echo/InteriorAgent"),
            args.require_interioragent,
        ),
        _check_path(
            "yolo_world_model",
            _path_from_env_or_config("YOLO_WORLD_MODEL", cfg, "paths.yolo_world_model", "data/models/yolov8l-worldv2.pt"),
            args.require_yolo_world,
        ),
        _check_path(
            "grounding_dino_checkpoint",
            _path_from_env_or_config(
                "GROUNDING_DINO_CHECKPOINT",
                cfg,
                "paths.grounding_dino_checkpoint",
                "data/models/groundingdino_swinb_cogcoor.pth",
            ),
            args.require_grounding_dino,
        ),
        _check_path(
            "grounding_dino_config",
            _path_from_env_or_config(
                "GROUNDING_DINO_CONFIG",
                cfg,
                "paths.grounding_dino_config",
                "/home/echo/SG-Nav/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py",
            ),
            args.require_grounding_dino,
        ),
        _check_path(
            "sam2_checkpoint",
            _path_from_env_or_config("SAM2_CHECKPOINT", cfg, "perception.sam2_checkpoint", "data/models/sam2.1_hiera_small.pt"),
            args.require_sam2,
        ),
        _check_path(
            "sam2_model_cfg",
            _path_from_env_or_config("SAM2_MODEL_CFG", cfg, "perception.sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_s.yaml"),
            args.require_sam2,
            allow_config_reference=True,
        ),
        _check_rose2_source(
            _path_from_env_or_config(
                str(get_nested(cfg, "mapping.room_segmentation.upstream_repo_env", DEFAULT_SOURCE_ENV) or DEFAULT_SOURCE_ENV),
                cfg,
                "mapping.room_segmentation.source_root",
                "",
            ),
            str(get_nested(cfg, "mapping.room_segmentation.upstream_repo_env", DEFAULT_SOURCE_ENV) or DEFAULT_SOURCE_ENV),
            args.require_rose2_source,
        ),
    ]
    missing_required = [item for item in checks if item["required"] and not item["exists"]]
    payload = {"ok": not missing_required, "checks": checks}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            prefix = "OK" if item["exists"] else ("MISSING" if item["required"] else "OPTIONAL_MISSING")
            print("%s %s: %s" % (prefix, item["kind"], item["path"]))
            if item.get("message"):
                print("  %s" % item["message"])
    return 0 if not missing_required else 2


if __name__ == "__main__":
    raise SystemExit(main())
