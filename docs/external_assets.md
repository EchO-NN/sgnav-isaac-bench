# External Assets

This repository does not vendor Isaac Sim, InteriorAgent scenes, GroundingDINO-B/Swin-B
weights, SAM2 checkpoints, optional upstream ROSE2/declutter-reconstruct source
checkout for debug/ablation, or LLM model/server assets.

## Default Asset Paths

The default config uses:

- Isaac Sim root: `/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64`
- InteriorAgent root: `/home/echo/InteriorAgent`
- GroundingDINO-B/Swin-B checkpoint: `data/models/groundingdino_swinb_cogcoor.pth`
- GroundingDINO-B/Swin-B config: `data/models/GroundingDINO_SwinB.cfg.py`
- SAM2 checkpoint: `data/models/sam2.1_hiera_small.pt`
- SAM2 model config: `configs/sam2.1/sam2.1_hiera_s.yaml` (accepted by SAM2 as
  a package/Hydra config reference; it does not need to exist as a repo-local
  file when the installed SAM2 package can resolve it)
- OpenAI-compatible LLM endpoint: `http://127.0.0.1:8000/v1`
- Optional upstream no-ROS ROSE2 source root for debug/ablation:
  `${ROSE2_SOURCE_ROOT}` pointing at a
  `goldleaf3i/declutter-reconstruct` checkout that contains
  `code/FFT_MQ.py`, `code/minibatch.py`, and `code/parameters.py`

## Environment Overrides

Use these environment variables or config fields to point at local assets:

- `ISAAC_SIM_ROOT` or `paths.isaac_sim_root`
- `ISAAC_ROOT` is accepted as a legacy alias for `ISAAC_SIM_ROOT` by the setup
  and activation scripts.
- `INTERIORAGENT_ROOT` or `paths.interioragent_root`
- `GROUNDING_DINO_CHECKPOINT` or `paths.grounding_dino_checkpoint`
- `GROUNDING_DINO_CONFIG` or `paths.grounding_dino_config`
- `GROUNDING_DINO_ROOT` for an installed GroundingDINO checkout
- `SAM2_CHECKPOINT` or `perception.sam2_checkpoint`
- `SAM2_MODEL_CFG` or `perception.sam2_model_cfg`
- `LLM_BASE_URL` or `llm.base_url`
- `ROSE2_SOURCE_ROOT` or `mapping.room_segmentation.source_root`

## Asset Check Command

Run:

```bash
./run_isaac_bench.sh -m isaac_bench.scripts.download_grounding_dino

python -m isaac_bench.scripts.check_assets \
  --require-grounding-dino \
  --require-sam2 \
  --require-interioragent \
  --require-isaac
```

The command exits with status `0` when all requested strict-default assets are
present and prints a clear missing-path report with nonzero status otherwise.
Add `--require-rose2-source` only when intentionally running an upstream ROSE2
debug/ablation backend.

## Benchmark Rule

Strict benchmark runs must fail clearly or skip clearly when required external
assets are missing. They must never silently switch to dry-run detection,
ground-truth object memory, missing-model stubs, static-map SG-Nav shortcuts, or
local deterministic LLM scoring while claiming `metric_valid=true`.
