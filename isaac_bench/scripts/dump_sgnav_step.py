from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from isaac_bench.metrics.episode_logger import make_jsonable
from isaac_bench.metrics.result_schema import empty_sgnav_step_dump


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an SG-Nav decision dump JSON artifact with the contract-required keys."
    )
    parser.add_argument("--output", required=True, help="Path to the JSON artifact to write.")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--goal-category", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    metadata = {
        key: value
        for key, value in {
            "episode_id": args.episode_id,
            "scene_id": args.scene_id,
            "goal_category": args.goal_category,
            "schema_only": True,
        }.items()
        if value is not None
    }
    payload = empty_sgnav_step_dump(metadata)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2 if args.pretty else None)
    out.write_text(text + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
