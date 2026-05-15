#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SGNAV_ISAAC_ENV_NAME:-sgnav-isaac}"
ISAAC_ROOT="${ISAAC_ROOT:-${ISAAC_SIM_ROOT:-/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64}}"
export ISAAC_SIM_ROOT="$ISAAC_ROOT"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/echo/miniforge3/etc/profile.d/conda.sh}"

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
