# External Assets

This repository does not vendor Isaac Sim, InteriorAgent scenes, YOLO-World
weights, SAM2 checkpoints, or LLM model/server assets.

## Default Asset Paths

The default config uses:

- Isaac Sim root: `/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64`
- InteriorAgent root: `/home/echo/InteriorAgent`
- YOLO-World model: `data/models/yolov8l-worldv2.pt`
- SAM2 checkpoint: `data/models/sam2.1_hiera_small.pt`
- SAM2 model config: `configs/sam2.1/sam2.1_hiera_s.yaml` (accepted by SAM2 as
  a package/Hydra config reference; it does not need to exist as a repo-local
  file when the installed SAM2 package can resolve it)
- OpenAI-compatible LLM endpoint: `http://127.0.0.1:8000/v1`

## Environment Overrides

Use these environment variables or config fields to point at local assets:

- `ISAAC_SIM_ROOT` or `paths.isaac_sim_root`
- `ISAAC_ROOT` is accepted as a legacy alias for `ISAAC_SIM_ROOT` by the setup
  and activation scripts.
- `INTERIORAGENT_ROOT` or `paths.interioragent_root`
- `YOLO_WORLD_MODEL` or `paths.yolo_world_model`
- `SAM2_CHECKPOINT` or `perception.sam2_checkpoint`
- `SAM2_MODEL_CFG` or `perception.sam2_model_cfg`
- `LLM_BASE_URL` or `llm.base_url`

## Asset Check Command

Run:

```bash
python -m isaac_bench.scripts.check_assets \
  --require-yolo-world \
  --require-sam2 \
  --require-interioragent \
  --require-isaac
```

The command exits with status `0` when all requested assets are present and
prints a clear missing-path report with nonzero status otherwise.

## Benchmark Rule

Strict benchmark runs must fail clearly or skip clearly when required external
assets are missing. They must never silently switch to dry-run detection,
ground-truth object memory, missing-model stubs, static-map SG-Nav shortcuts, or
local deterministic LLM scoring while claiming `metric_valid=true`.
