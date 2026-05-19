from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from isaac_bench.mapping.rose2_upstream_io import load_rose2_upstream_input_npz
from isaac_bench.mapping.rose2_upstream_runner import run_rose2_external_runner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one saved vertical-free ROSE2 input through the upstream source runner.")
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--backend", default="rose2_source_external_runner", choices=["rose2_source_external_runner"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoding", default="auto")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--min-room-area-m2", type=float, default=1.5)
    args = parser.parse_args(argv)
    occupied, free, unknown, vertical_observed, vertical_free, wall_conf, resolution = load_rose2_upstream_input_npz(Path(args.input_npz))
    result = run_rose2_external_runner(
        source_root=args.source_root,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=vertical_free,
        wall_confidence_map=wall_conf,
        resolution_m=float(resolution),
        min_room_area_m2=float(args.min_room_area_m2),
        work_dir=args.out_dir,
        timeout_s=float(args.timeout_s),
        encoding=str(args.encoding),
    )
    summary = dict(result.debug.get("external_runner_summary", {}))
    print("backend: %s" % result.backend)
    print("room_count: %d" % int(summary.get("room_count", summary.get("source_room_count", 0)) or 0))
    print("source_output_path: %s" % str(summary.get("source_output_path", "")))
    print("encoding: %s" % str(summary.get("chosen_encoding", args.encoding)))
    print("source_form_used: %s" % str(bool(summary.get("source_form_used_for_final", False))).lower())
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
