#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/result/radius005_voxel_v36_all_scenes}"
exec "$SCRIPT_DIR/run_radius005_all_scenes_voxel_v33.sh" "$@"
