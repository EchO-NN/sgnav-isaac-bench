from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect accepted/rejected online_roomseg separator candidates.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-id", type=int, default=None)
    parser.add_argument("--kind", default=None)
    args = parser.parse_args(argv)

    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    candidates = list(data.get("candidates", []) or [])
    if args.candidate_id is not None:
        candidates = [item for item in candidates if int(item.get("candidate_id", -1)) == int(args.candidate_id)]
    if args.kind:
        candidates = [item for item in candidates if str(item.get("kind", "")) == str(args.kind)]
    print(json.dumps({"count": len(candidates), "candidates": candidates}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
