#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_last_jsonl(path: Path) -> dict:
    last = {}
    if not path.exists():
        return last
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()

    result = _read_last_jsonl(Path(args.result))
    trace_rows = list(_iter_jsonl(Path(args.trace)) or [])
    failures: list[str] = []
    warnings: list[str] = []

    runtime_radius = _finite_float(result.get("runtime_robot_radius_m", result.get("robot_radius_m")))
    configured_radius = _finite_float(result.get("configured_robot_radius_m"))
    if runtime_radius is not None and configured_radius is not None and runtime_radius + 1e-6 < configured_radius:
        failures.append("runtime robot radius %.3f < configured %.3f" % (runtime_radius, configured_radius))

    for row in trace_rows:
        configured = _finite_float(row.get("lookahead_min_clearance_m_configured"))
        effective = _finite_float(row.get("lookahead_min_clearance_m_effective"))
        if configured is not None and effective is not None and effective + 1e-6 < configured:
            failures.append(
                "step %s lookahead effective %.3f < configured %.3f"
                % (row.get("step"), effective, configured)
            )
            break

    mismatch_rows = [row for row in trace_rows if bool(row.get("commit_center_target_mismatch"))]
    if mismatch_rows:
        failures.append("commitment center/target mismatch in %d trace rows" % len(mismatch_rows))

    target_failures = [
        row
        for row in trace_rows
        if _finite_float(row.get("target_clearance_m")) is not None
        and _finite_float(row.get("target_clearance_m")) < _finite_float(row.get("guard_min_clearance_m_effective") or 0.0)
    ]
    if target_failures:
        warnings.append("target clearance below guard clearance in %d trace rows" % len(target_failures))

    guard_failures = [
        row
        for row in trace_rows
        if _finite_float(row.get("guard_min_swept_clearance_m")) is not None
        and _finite_float(row.get("guard_min_clearance_m")) is not None
        and _finite_float(row.get("guard_min_swept_clearance_m")) + 1e-6 < _finite_float(row.get("guard_min_clearance_m"))
        and not bool(row.get("blocked_by_online_guard"))
    ]
    if guard_failures:
        failures.append("unblocked low-clearance swept guard rows: %d" % len(guard_failures))

    if result.get("stop_reason") == "agent_off_static_metric_map" or result.get("failure_reason") == "agent_off_static_metric_map":
        failures.append("agent_off_static_metric_map used as stop/failure reason")
    elif bool(result.get("agent_off_static_metric_map_online_diagnostic", False)):
        warnings.append("static metric invalid is diagnostic only; online policy did not use static map")

    for item in failures:
        print("[FAIL] " + item)
    for item in warnings:
        print("[WARN] " + item)
    if not failures and not warnings:
        print("[OK] online navigation safety audit passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
