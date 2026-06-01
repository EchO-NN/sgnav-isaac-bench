#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_sgnav_isaac_env.sh"

export NUMBA_NUM_THREADS="${SGNAV_NUMBA_THREADS:-28}"
export OMP_NUM_THREADS="${SGNAV_OMP_THREADS:-1}"
export MKL_NUM_THREADS="${SGNAV_MKL_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${SGNAV_OPENBLAS_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${SGNAV_NUMEXPR_THREADS:-1}"

exec python "$@"
