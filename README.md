# SG-Nav Isaac / InteriorAgent Benchmark

This repository wraps SG-Nav ObjectNav evaluation on Isaac Sim and
InteriorAgent scenes. The metric path is intentionally strict: Isaac RGB-D,
YOLO-World detections, SAM2 masks, mask/depth object memory, online mapping,
scene-graph subgraph reasoning, frontier interpolation, candidate
re-perception, STOP confirmation, and deterministic A*/FMM-equivalent local
planning.

See also:

- `docs/sgnav_paper_parity_contract.md`
- `docs/benchmark_metric_validity_contract.md`
- `docs/sgnav_decision_dump_contract.md`
- `docs/external_assets.md`

## Setup

External assets are not vendored. By default the project expects:

- Isaac Sim: `/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64`
- InteriorAgent: `/home/echo/InteriorAgent`
- YOLO-World: `data/models/yolov8l-worldv2.pt`
- SAM2: `data/models/sam2.1_hiera_small.pt`
- Optional OpenAI-compatible LLM endpoint: `http://127.0.0.1:8000/v1`

Override paths with `ISAAC_SIM_ROOT`, `INTERIORAGENT_ROOT`,
`YOLO_WORLD_MODEL`, `SAM2_CHECKPOINT`, `SAM2_MODEL_CFG`, and `LLM_BASE_URL`.
`ISAAC_ROOT` remains a legacy alias for `ISAAC_SIM_ROOT`.

```bash
./scripts/setup_sgnav_isaac_env.sh
source ./scripts/activate_sgnav_isaac_env.sh
```

Check required assets:

```bash
python -m isaac_bench.scripts.check_assets \
  --require-yolo-world \
  --require-sam2 \
  --require-interioragent \
  --require-isaac
```

## Preprocess And Episodes

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.preprocess_interioragent \
  --dataset-root ${INTERIORAGENT_ROOT:-/home/echo/InteriorAgent} \
  --out data/interioragent_preprocessed \
  --resolution 0.05 \
  --scene-id kujiale_0031
```

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.generate_episodes \
  --preprocessed-dir data/interioragent_preprocessed \
  --scene-id kujiale_0031 \
  --episodes-per-scene 10 \
  --out data/interioragent_episodes/debug.jsonl
```

## Smoke Run

This command is for smoke/debug only. It must produce `metric_valid=false`.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --detector dry_run \
  --sim-backend map \
  --allow-debug-fallbacks \
  --output data/isaac_bench_runs/dry_run_smoke/results.jsonl
```

## Strict Isaac SG-Nav

Strict SG-Nav requires YOLO-World and SAM2. A row is metric-valid only when no
debug fallback is used. Local deterministic LLM scoring is non-metric unless a
named ablation is explicitly declared; configure a real OpenAI-compatible LLM
for metric SG-Nav scoring.

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector yolo_world \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --llm-enabled true \
  --output data/isaac_bench_runs/final_strict_smoke/results.jsonl \
  --debug-map debug/final_strict_smoke.png
```

For a short integration check on machines with Isaac/model assets:

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector yolo_world \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --max-control-steps 1 \
  --panorama-steps 0 \
  --output data/isaac_bench_runs/short_isaac_smoke/results.jsonl
```

## Batch And Summary

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_benchmark \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --planner astar \
  --policy sgnav_original \
  --detector yolo_world \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --output-dir data/isaac_bench_runs/sgnav_yolo_sam2_debug
```

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.metrics.summarize \
  --input data/isaac_bench_runs/sgnav_yolo_sam2_debug/results.jsonl \
  --out data/isaac_bench_runs/sgnav_yolo_sam2_debug/summary.json
```

## Debug Artifacts

Enable saved panels and graph dumps:

```bash
./scripts/run_sgnav_isaac_env.sh \
  -m isaac_bench.scripts.run_one_episode \
  --config isaac_bench/configs/isaac_bench.yaml \
  --episode-file data/interioragent_episodes/debug.jsonl \
  --episode-index 0 \
  --planner astar \
  --policy sgnav_original \
  --detector yolo_world \
  --segmenter sam2 \
  --sim-backend isaac \
  --headless true \
  --strict-benchmark true \
  --sgnav-viz-save-dir debug/sgnav_panels \
  --debug-graph-dump \
  --debug-graph-dump-dir debug/graphs \
  --output data/isaac_bench_runs/debug_panels/results.jsonl
```

Convert one graph step into the SG-Nav decision dump contract:

```bash
python -m isaac_bench.scripts.dump_sgnav_step \
  --output debug/sgnav_step.json \
  --graph-debug-dump debug/graphs/graph_step_000000.json \
  --result-row data/isaac_bench_runs/debug_panels/results.jsonl \
  --pretty
```

## Tests

After activating the environment:

```bash
pytest -q
```

From the repo-local lightweight environment:

```bash
./run_isaac_bench.sh -m pytest -q
```

Generated data, logs, maps, videos, and benchmark outputs belong under ignored
`data/` or `debug/` paths.
