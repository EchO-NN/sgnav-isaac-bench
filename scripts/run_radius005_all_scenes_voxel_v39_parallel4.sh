#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-7}"
export RUN_NUMBA_NUM_THREADS="${RUN_NUMBA_NUM_THREADS:-$NUMBA_NUM_THREADS}"
export RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/result/radius005_robot005_voxel_v39_parallel4_all_scenes}"

exec "$SCRIPT_DIR/run_radius005_all_scenes_voxel_v38.sh" "$@"
