from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import List, Optional


def summarize_rows(rows: List[dict], group_keys: Optional[List[str]] = None) -> List[dict]:
    group_keys = group_keys or ["planner", "detector"]
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups[key].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        n = len(vals)
        item = {k: v for k, v in zip(group_keys, key)}
        item.update(
            {
                "episodes": n,
                "sr": sum(1.0 for r in vals if r.get("success")) / max(n, 1),
                "spl": sum(float(r.get("spl", 0.0)) for r in vals) / max(n, 1),
                "softspl": sum(float(r.get("softspl", 0.0)) for r in vals) / max(n, 1),
                "mean_dtg": sum(float(r.get("distance_to_goal", 0.0)) for r in vals) / max(n, 1),
                "mean_path_length": sum(float(r.get("path_length", 0.0)) for r in vals) / max(n, 1),
            }
        )
        out.append(item)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--by", nargs="*", default=["planner", "detector"])
    args = parser.parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in summarize_rows(rows, args.by):
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

