#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ROOT="${RUN_ROOT:-data/isaac_bench_runs/radius005_voxel_v35_all_scenes}"
exec "$SCRIPT_DIR/run_radius005_all_scenes_voxel_v33.sh" "$@"
