#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${SGNAV_ISAAC_ENV_NAME:-sgnav-isaac}"
MAMBA="${MAMBA:-/home/echo/miniforge3/bin/mamba}"
CONDA_SH="${CONDA_SH:-/home/echo/miniforge3/etc/profile.d/conda.sh}"
ISAAC_ROOT="${ISAAC_ROOT:-/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64}"
SAM2_SPEC="${SAM2_SPEC:-git+https://github.com/facebookresearch/sam2.git}"
CLIP_SPEC="${CLIP_SPEC:-git+https://github.com/ultralytics/CLIP.git}"
INSTALL_SAM2="${INSTALL_SAM2:-1}"

if [[ ! -x "$MAMBA" ]]; then
  echo "mamba not found or not executable: $MAMBA" >&2
  exit 1
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "conda shell hook not found: $CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$ISAAC_ROOT/setup_conda_env.sh" ]]; then
  echo "Isaac setup_conda_env.sh not found under: $ISAAC_ROOT" >&2
  exit 1
fi

if "$MAMBA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$MAMBA" env update -n "$ENV_NAME" -f "$REPO_ROOT/envs/sgnav-isaac-py311.yml"
else
  "$MAMBA" env create -n "$ENV_NAME" -f "$REPO_ROOT/envs/sgnav-isaac-py311.yml"
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Isaac Sim 5.1 requires Python 3.11, got %s" % sys.version)
print("python", sys.version)
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$REPO_ROOT"
python -m pip install \
  "numpy>=1.26,<1.27" \
  "opencv-python<4.12" \
  psutil \
  pycocotools \
  supervision \
  huggingface_hub \
  safetensors \
  einops \
  iopath \
  fvcore \
  hydra-core \
  ultralytics-thop \
  ftfy \
  regex \
  wcwidth

# Install packages that depend on torch without letting pip pull a second torch
# wheel or mutate Isaac's prebundled torch/numpy packages.
python -m pip install --no-deps "$CLIP_SPEC"
python -m pip install --no-deps ultralytics

if [[ "$INSTALL_SAM2" == "1" || "$INSTALL_SAM2" == "true" || "$INSTALL_SAM2" == "yes" ]]; then
  python -m pip install --no-deps "$SAM2_SPEC"
fi

# Isaac Sim 5.1 exposes Kit/Omni/Isaac Python packages and native libraries
# through this script. Source it after pip installs so pip never uninstalls or
# upgrades Isaac's prebundled packages while resolving dependencies.
# Isaac's setup script reads optional shell variables such as ZSH_VERSION. Keep
# our script strict, but do not let nounset break third-party setup code.
set +u
# shellcheck disable=SC1090
source "$ISAAC_ROOT/setup_conda_env.sh"
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import numpy
import torch
print("numpy      ", numpy.__version__, numpy.__file__)
print("torch       ", torch.__version__, torch.version.cuda)
for name in ("isaacsim", "omni", "torch", "ultralytics", "cv2"):
    spec = importlib.util.find_spec(name)
    print("%-12s %s" % (name, "ok" if spec else "missing"))
sam2 = importlib.util.find_spec("sam2")
print("%-12s %s" % ("sam2", "ok" if sam2 else "missing"))
PY

echo
echo "Done. Activate with:"
echo "  source $REPO_ROOT/scripts/activate_sgnav_isaac_env.sh"
echo
echo "Run with:"
echo "  $REPO_ROOT/scripts/run_sgnav_isaac_env.sh -m isaac_bench.scripts.run_one_episode ..."
