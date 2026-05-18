#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SGNAV_ISAAC_ENV_NAME:-sgnav-isaac}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ISAAC_ROOT:-}" ]]; then
  ISAAC_ROOT="${ISAAC_SIM_ROOT:-}"
fi
if [[ -z "$ISAAC_ROOT" ]]; then
  for candidate in \
    "$HOME/isaac-sim-standalone-5.1.0-linux-x86_64" \
    "/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64" \
    "/home/joey/isaac-sim-standalone-5.1.0-linux-x86_64"; do
    if [[ -f "$candidate/setup_conda_env.sh" ]]; then
      ISAAC_ROOT="$candidate"
      break
    fi
  done
fi
ISAAC_ROOT="${ISAAC_ROOT:-/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64}"
export ISAAC_SIM_ROOT="$ISAAC_ROOT"

if [[ -z "${CONDA_SH:-}" ]]; then
  for candidate in \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/home/echo/miniforge3/etc/profile.d/conda.sh" \
    "/home/joey/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      CONDA_SH="$candidate"
      break
    fi
  done
fi
CONDA_SH="${CONDA_SH:-/home/echo/miniforge3/etc/profile.d/conda.sh}"

if [[ -z "${ROSE2_SOURCE_ROOT:-}" ]]; then
  for candidate in \
    "$HOME/declutter-reconstruct" \
    "$REPO_ROOT/../declutter-reconstruct" \
    "/home/echo/declutter-reconstruct" \
    "/home/joey/declutter-reconstruct"; do
    if [[ -f "$candidate/code/FFT_MQ.py" && -f "$candidate/code/minibatch.py" && -f "$candidate/code/parameters.py" ]]; then
      export ROSE2_SOURCE_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -z "${INTERIORAGENT_ROOT:-}" ]]; then
  for candidate in \
    "$HOME/InteriorAgent" \
    "$REPO_ROOT/../InteriorAgent" \
    "/home/echo/InteriorAgent" \
    "/home/joey/InteriorAgent"; do
    if [[ -d "$candidate" ]]; then
      export INTERIORAGENT_ROOT="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "conda shell hook not found: $CONDA_SH" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$ISAAC_ROOT/setup_conda_env.sh" ]]; then
  echo "Isaac setup_conda_env.sh not found under: $ISAAC_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

# Isaac Sim 5.1 exposes Kit/Omni/Isaac Python packages and native libraries
# through this script. It requires the active Python to be 3.11.
# Isaac's setup script reads optional shell variables such as ZSH_VERSION. Keep
# our script strict, but do not let nounset break third-party setup code.
set +u
# shellcheck disable=SC1090
source "$ISAAC_ROOT/setup_conda_env.sh"
set -u

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
