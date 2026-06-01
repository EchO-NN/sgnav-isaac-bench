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

export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-28}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$EPISODE_DIR" "$RUN_ROOT"

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$PREPROCESSED_DIR/index.json" ]]; then
  echo "[all-scenes] preprocessing InteriorAgent scenes at radius 0.05m -> $PREPROCESSED_DIR"
  ./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/preprocess_interioragent.py \
    --dataset-root "$DATASET_ROOT" \
    --scene-glob "$SCENE_GLOB" \
    --out "$PREPROCESSED_DIR" \
    --resolution 0.05 \
    --robot-radius-m 0.05 \
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
    --min-planning-clearance-m 0.0 \
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

run_index=0
while IFS= read -r episode_file; do
  [[ -z "$episode_file" ]] && continue
  scene_id="$(basename "$episode_file" .jsonl)"
  scene_run_dir="$RUN_ROOT/$scene_id"
  snapshot_dir="$scene_run_dir/roomseg_snapshots"
  mkdir -p "$snapshot_dir"
  rm -f "$scene_run_dir/results.jsonl"
  run_index=$((run_index + 1))
  echo "[all-scenes] ($run_index) running $scene_id"

  cat > "$scene_run_dir/command.txt" <<EOF
./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/run_one_episode.py --config isaac_bench/configs/isaac_bench.yaml --episode-file "$episode_file" --episode-index 0 --sim-backend isaac --planner astar --detector none --segmenter none --no-llm-enabled --no-vllm-frontier-scoring --no-vllm-image-scoring --frontier-selection-mode random --frontier-source voxel_vertical_free --room-map-mode voxel_occupancy_door_wall_v33_vlm --roomseg-backend voxel_occupancy_door_wall_v33 --no-strict-benchmark --allow-debug-fallbacks --explore-until-no-frontiers --max-control-steps 5000 --robot-radius-m 0.05 --headless --sgnav-viz --sgnav-viz-every-steps 1 --save-roomseg-snapshots --roomseg-snapshot-dir "$snapshot_dir" --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" --save-roomseg-voxel-evidence --output "$scene_run_dir/results.jsonl"
EOF

  run_status=0
  if ./scripts/run_sgnav_isaac_env.sh isaac_bench/scripts/run_one_episode.py \
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
    --robot-radius-m 0.05 \
    --headless \
    --sgnav-viz \
    --sgnav-viz-every-steps 1 \
    --save-roomseg-snapshots \
    --roomseg-snapshot-dir "$snapshot_dir" \
    --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" \
    --save-roomseg-voxel-evidence \
    --output "$scene_run_dir/results.jsonl" \
    2>&1 | tee "$scene_run_dir/run.log"; then
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
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$run_status"
    fi
  fi
done < "$EPISODE_DIR/episode_files.txt"

echo "[all-scenes] done. Runs are under $RUN_ROOT"
