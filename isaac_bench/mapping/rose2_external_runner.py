from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from isaac_bench.mapping.rose2_source_external_runner import run_rose2_source_external_runner


def run_rose2_external_runner(
    *,
    source_root: str | Path | None,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    work_dir: str | Path,
    timeout_s: float = 60.0,
    unknown_mode: str = "gray",
):
    """Run the upstream ROSE2 wrapper without silent fallback.

    `unknown_mode` is recorded for A/B diagnostics. The current wrapper exports
    the same NPZ masks either way; metric-map rendering remains centralized in
    `rose2_source_external_runner`.
    """

    result = run_rose2_source_external_runner(
        source_root=source_root,
        observed_occupied=observed_occupied,
        observed_free=observed_free,
        unknown=unknown,
        work_dir=work_dir,
        timeout_s=float(timeout_s),
    )
    result.debug["external_unknown_mode"] = str(unknown_mode)
    summary_path = Path(work_dir) / "external_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["unknown_mode"] = str(unknown_mode)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    free = np.asarray(data.get("vertical_free_room_domain", data.get("observed_free", data.get("free"))), dtype=bool)
    occupied = np.asarray(data.get("repaired_roomseg_occupied", data.get("observed_occupied", data.get("occupied", np.zeros_like(free)))), dtype=bool)
    unknown = np.asarray(data.get("repaired_roomseg_unknown", data.get("unknown", np.zeros_like(free))), dtype=bool)
    return occupied, free, unknown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--input", "--input-npz", dest="input_npz", required=True)
    parser.add_argument("--out-dir", "--work-dir", dest="work_dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--unknown-mode", choices=["gray", "free"], default="gray")
    args = parser.parse_args(argv)
    occupied, free, unknown = _load_npz(Path(args.input_npz))
    if args.unknown_mode == "free":
        free = free | unknown
        unknown = np.zeros_like(free, dtype=bool)
    result = run_rose2_external_runner(
        source_root=args.source_root,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        work_dir=args.work_dir,
        timeout_s=float(args.timeout_s),
        unknown_mode=str(args.unknown_mode),
    )
    print(json.dumps(result.debug.get("external_runner_summary", result.debug), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
