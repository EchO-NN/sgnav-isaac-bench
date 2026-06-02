#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/echo/miniforge3/envs/sgnav-isaac/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/echo/InteriorAgent}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/interioragent_preprocessed_radius005}"
EPISODE_DIR="${EPISODE_DIR:-data/interioragent_episodes/radius005_all_scenes}"
RUN_ROOT="${RUN_ROOT:-data/isaac_bench_runs/radius005_voxel_v33_all_scenes}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*}"
ROOMSEG_SNAPSHOT_MAX_SAVES="${ROOMSEG_SNAPSHOT_MAX_SAVES:-100000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
FORCE_EPISODES="${FORCE_EPISODES:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
ROBOT_RADIUS_M="${ROBOT_RADIUS_M:-0.05}"
RUNTIME_PLANNING_CLEARANCE_M="${RUNTIME_PLANNING_CLEARANCE_M:-0.01}"
ASTAR_GOAL_MIN_CLEARANCE_M="${ASTAR_GOAL_MIN_CLEARANCE_M:-0.08}"
LOOKAHEAD_MIN_CLEARANCE_M="${LOOKAHEAD_MIN_CLEARANCE_M:-0.06}"
GUARD_MIN_CLEARANCE_M="${GUARD_MIN_CLEARANCE_M:-0.06}"
ASTAR_CLEARANCE_DESIRED_M="${ASTAR_CLEARANCE_DESIRED_M:-0.12}"

export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-28}"
RUN_NUMBA_NUM_THREADS="${RUN_NUMBA_NUM_THREADS:-$NUMBA_NUM_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! [[ "$PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "[all-scenes] PARALLEL_JOBS must be a positive integer, got '$PARALLEL_JOBS'" >&2
  exit 1
fi
if ! [[ "$RUN_NUMBA_NUM_THREADS" =~ ^[0-9]+$ ]] || [[ "$RUN_NUMBA_NUM_THREADS" -lt 1 ]]; then
  echo "[all-scenes] RUN_NUMBA_NUM_THREADS must be a positive integer, got '$RUN_NUMBA_NUM_THREADS'" >&2
  exit 1
fi

mkdir -p "$EPISODE_DIR" "$RUN_ROOT"

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$PREPROCESSED_DIR/index.json" ]]; then
  echo "[all-scenes] preprocessing InteriorAgent scenes at radius 0.05m -> $PREPROCESSED_DIR"
  ./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/preprocess_interioragent.py \
    --dataset-root "$DATASET_ROOT" \
    --scene-glob "$SCENE_GLOB" \
    --out "$PREPROCESSED_DIR" \
    --resolution 0.05 \
    --robot-radius-m "$ROBOT_RADIUS_M" \
    --inflation-radius-m 0.0
else
  echo "[all-scenes] reusing preprocessed scenes in $PREPROCESSED_DIR"
fi

if [[ "$FORCE_EPISODES" == "1" || ! -s "$EPISODE_DIR/episode_files.txt" ]]; then
  echo "[all-scenes] generating one longest-instance A* episode per scene"
  "$PYTHON_BIN" isaac_bench/scripts/generate_all_scene_episodes_radius005.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --out-dir "$EPISODE_DIR" \
    --combined-out "$EPISODE_DIR/all_scenes_radius005.jsonl" \
    --report-out "$EPISODE_DIR/all_scenes_radius005_report.json" \
    --file-list-out "$EPISODE_DIR/episode_files.txt" \
    --debug-map-dir "$EPISODE_DIR/debug_maps" \
    --min-planning-clearance-m "$RUNTIME_PLANNING_CLEARANCE_M" \
    --robot-spawn-height-m 0.05
else
  episode_count="$(wc -l < "$EPISODE_DIR/episode_files.txt")"
  echo "[all-scenes] reusing $episode_count generated episode files in $EPISODE_DIR"
fi

if [[ "$GENERATE_ONLY" == "1" ]]; then
  echo "[all-scenes] GENERATE_ONLY=1, not launching Isaac runs"
  exit 0
fi

if [[ ! -s "$EPISODE_DIR/episode_files.txt" ]]; then
  echo "[all-scenes] no episode files generated" >&2
  exit 1
fi

acquire_run_lock() {
  local lock_dir="$RUN_ROOT/.run_lock"
  local lock_pid=""
  if mkdir "$lock_dir" 2>/dev/null; then
    echo "$$" > "$lock_dir/pid"
    trap 'rm -rf "$RUN_ROOT/.run_lock"' EXIT
    return 0
  fi
  if [[ -f "$lock_dir/pid" ]]; then
    lock_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
  fi
  if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
    echo "[all-scenes] RUN_ROOT is already being scheduled by pid $lock_pid: $RUN_ROOT" >&2
    echo "[all-scenes] use a different RUN_ROOT or stop that run first" >&2
    exit 1
  fi
  echo "[all-scenes] removing stale run lock: $lock_dir"
  rm -rf "$lock_dir"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[all-scenes] failed to acquire RUN_ROOT lock: $lock_dir" >&2
    exit 1
  fi
  echo "$$" > "$lock_dir/pid"
  trap 'rm -rf "$RUN_ROOT/.run_lock"' EXIT
}

run_scene() {
  local episode_file="$1"
  local run_index="$2"
  [[ -z "$episode_file" ]] && return 0
  local scene_id
  local scene_run_dir
  local snapshot_dir
  scene_id="$(basename "$episode_file" .jsonl)"
  scene_run_dir="$RUN_ROOT/$scene_id"
  snapshot_dir="$scene_run_dir/roomseg_snapshots"
  mkdir -p "$snapshot_dir"
  rm -f "$scene_run_dir/results.jsonl"
  echo "[all-scenes] ($run_index) running $scene_id threads=$RUN_NUMBA_NUM_THREADS"

  cat > "$scene_run_dir/command.txt" <<EOF
NUMBA_NUM_THREADS=$RUN_NUMBA_NUM_THREADS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/run_one_episode.py --config isaac_bench/configs/isaac_bench.yaml --episode-file "$episode_file" --episode-index 0 --sim-backend isaac --planner astar --detector none --segmenter none --no-llm-enabled --no-vllm-frontier-scoring --no-vllm-image-scoring --frontier-selection-mode random --frontier-source voxel_vertical_free --room-map-mode voxel_occupancy_door_wall_v33_vlm --roomseg-backend voxel_occupancy_door_wall_v33 --no-strict-benchmark --allow-debug-fallbacks --explore-until-no-frontiers --max-control-steps 5000 --robot-radius-m "$ROBOT_RADIUS_M" --runtime-planning-clearance-m "$RUNTIME_PLANNING_CLEARANCE_M" --astar-clearance-desired-m "$ASTAR_CLEARANCE_DESIRED_M" --astar-goal-min-clearance-m "$ASTAR_GOAL_MIN_CLEARANCE_M" --lookahead-min-clearance-m "$LOOKAHEAD_MIN_CLEARANCE_M" --guard-min-clearance-m "$GUARD_MIN_CLEARANCE_M" --headless --sgnav-viz --sgnav-viz-every-steps 1 --save-roomseg-snapshots --roomseg-snapshot-dir "$snapshot_dir" --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" --save-roomseg-voxel-evidence --output "$scene_run_dir/results.jsonl"
EOF

  local run_status=0
  if NUMBA_NUM_THREADS="$RUN_NUMBA_NUM_THREADS" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/run_one_episode.py \
    --config isaac_bench/configs/isaac_bench.yaml \
    --episode-file "$episode_file" \
    --episode-index 0 \
    --sim-backend isaac \
    --planner astar \
    --detector none \
    --segmenter none \
    --no-llm-enabled \
    --no-vllm-frontier-scoring \
    --no-vllm-image-scoring \
    --frontier-selection-mode random \
    --frontier-source voxel_vertical_free \
    --room-map-mode voxel_occupancy_door_wall_v33_vlm \
    --roomseg-backend voxel_occupancy_door_wall_v33 \
    --no-strict-benchmark \
    --allow-debug-fallbacks \
    --explore-until-no-frontiers \
    --max-control-steps 5000 \
    --robot-radius-m "$ROBOT_RADIUS_M" \
    --runtime-planning-clearance-m "$RUNTIME_PLANNING_CLEARANCE_M" \
    --astar-clearance-desired-m "$ASTAR_CLEARANCE_DESIRED_M" \
    --astar-goal-min-clearance-m "$ASTAR_GOAL_MIN_CLEARANCE_M" \
    --lookahead-min-clearance-m "$LOOKAHEAD_MIN_CLEARANCE_M" \
    --guard-min-clearance-m "$GUARD_MIN_CLEARANCE_M" \
    --headless \
    --sgnav-viz \
    --sgnav-viz-every-steps 1 \
    --save-roomseg-snapshots \
    --roomseg-snapshot-dir "$snapshot_dir" \
    --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" \
    --save-roomseg-voxel-evidence \
    --output "$scene_run_dir/results.jsonl" \
    > "$scene_run_dir/run.log" 2>&1; then
    run_status=0
  else
    run_status=$?
  fi

  if [[ "$run_status" -eq 0 && ! -s "$scene_run_dir/results.jsonl" ]]; then
    run_status=1
    echo "[all-scenes] $scene_id failed: missing or empty $scene_run_dir/results.jsonl" | tee -a "$scene_run_dir/run.log"
  fi

  if [[ "$run_status" -eq 0 ]]; then
    echo "[all-scenes] $scene_id finished"
  else
    echo "[all-scenes] $scene_id failed with status $run_status" | tee -a "$scene_run_dir/run.log"
  fi
  return "$run_status"
}

run_index=0
active_jobs=0
failure_count=0
declare -A scheduled_scene_ids=()
acquire_run_lock
echo "[all-scenes] launching runs under $RUN_ROOT jobs=$PARALLEL_JOBS threads_per_job=$RUN_NUMBA_NUM_THREADS"
while IFS= read -r episode_file; do
  [[ -z "$episode_file" ]] && continue
  scene_id="$(basename "$episode_file" .jsonl)"
  if [[ -n "${scheduled_scene_ids[$scene_id]+x}" ]]; then
    echo "[all-scenes] skipping duplicate scene in episode list: $scene_id"
    continue
  fi
  scheduled_scene_ids[$scene_id]=1
  run_index=$((run_index + 1))
  run_scene "$episode_file" "$run_index" &
  active_jobs=$((active_jobs + 1))
  if [[ "$active_jobs" -ge "$PARALLEL_JOBS" ]]; then
    if ! wait -n; then
      failure_count=$((failure_count + 1))
    fi
    active_jobs=$((active_jobs - 1))
    if [[ "$CONTINUE_ON_ERROR" != "1" && "$failure_count" -gt 0 ]]; then
      wait
      exit 1
    fi
  fi
done < "$EPISODE_DIR/episode_files.txt"

while [[ "$active_jobs" -gt 0 ]]; do
  if ! wait -n; then
    failure_count=$((failure_count + 1))
  fi
  active_jobs=$((active_jobs - 1))
done

if [[ "$failure_count" -gt 0 ]]; then
  echo "[all-scenes] done with $failure_count failed scene(s). Runs are under $RUN_ROOT"
  if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
  exit 0
fi

echo "[all-scenes] done. Runs are under $RUN_ROOT"
