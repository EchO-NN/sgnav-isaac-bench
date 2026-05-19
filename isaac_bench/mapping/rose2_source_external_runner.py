from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from isaac_bench.mapping.rose2_upstream_io import load_rose2_upstream_input_npz
from isaac_bench.mapping.rose2_upstream_runner import (
    REQUIRED_UPSTREAM_SOURCE_FILES as REQUIRED_EXTERNAL_FILES,
    run_rose2_source_external_runner,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--encoding", default="auto")
    args = parser.parse_args(argv)
    occupied, free, unknown, vertical_observed, vertical_free, wall_conf, resolution = load_rose2_upstream_input_npz(Path(args.input_npz))
    result = run_rose2_source_external_runner(
        source_root=args.source_root,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=vertical_free,
        wall_confidence_map=wall_conf,
        resolution_m=float(resolution),
        work_dir=args.work_dir,
        timeout_s=float(args.timeout_s),
        encoding=str(args.encoding),
    )
    print(json.dumps(result.debug.get("external_runner_summary", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
