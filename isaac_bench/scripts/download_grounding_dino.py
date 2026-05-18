from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from isaac_bench.perception.grounding_dino_detector import DEFAULT_GROUNDING_DINO_REPO_ID


DEFAULT_CHECKPOINT_FILE = "groundingdino_swinb_cogcoor.pth"
DEFAULT_CONFIG_FILE = "GroundingDINO_SwinB.cfg.py"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download GroundingDINO-B/Swin-B config and checkpoint.")
    parser.add_argument("--repo-id", default=DEFAULT_GROUNDING_DINO_REPO_ID)
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--out-dir", default="data/models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit("huggingface_hub is required to download GroundingDINO-B/Swin-B assets: %s" % exc) from exc

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads = []
    for filename in (args.config_file, args.checkpoint_file):
        target = out_dir / filename
        if target.exists() and not args.force:
            downloads.append(target)
            continue
        cached = Path(
            hf_hub_download(
                repo_id=str(args.repo_id),
                filename=str(filename),
                local_dir=str(out_dir),
                force_download=bool(args.force),
            )
        )
        if cached.resolve() != target.resolve():
            shutil.copy2(cached, target)
        downloads.append(target)
    print("GroundingDINO-B/Swin-B assets ready:")
    print("  config: %s" % downloads[0])
    print("  checkpoint: %s" % downloads[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
