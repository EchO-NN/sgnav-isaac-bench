from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay voxel snapshots in parallel with task-level CPU parallelism.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--backend", default="cpu_numba")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--glob", default="*.npz")
    args = parser.parse_args(argv)

    snapshot_dir = Path(args.snapshot_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    snapshots = sorted(snapshot_dir.rglob(str(args.glob)))
    if not snapshots:
        raise SystemExit("no snapshots found under %s with glob %s" % (str(snapshot_dir), str(args.glob)))
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(snapshot: Path) -> dict[str, object]:
        rel = snapshot.relative_to(snapshot_dir)
        profile_path = out_dir / rel.with_suffix(".profile.json")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["NUMBA_NUM_THREADS"] = str(int(args.threads_per_job))
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        cmd = [
            sys.executable,
            "isaac_bench/scripts/benchmark_voxel_grid_update.py",
            "--snapshot",
            str(snapshot),
            "--backend",
            str(args.backend),
            "--threads",
            str(int(args.threads_per_job)),
            "--threads-mode",
            "manual",
            "--repeat",
            str(int(args.repeat)),
            "--warmup",
            str(int(args.warmup)),
            "--json-out",
            str(profile_path),
            "--assert-backend",
            str(args.backend),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)
        return {
            "snapshot": str(snapshot),
            "profile": str(profile_path),
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        for result in pool.map(run_one, snapshots):
            results.append(result)
            status = "ok" if int(result["returncode"]) == 0 else "failed"
            print("[batch-replay] %s %s" % (status, result["snapshot"]), flush=True)

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in results if int(item["returncode"]) != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
