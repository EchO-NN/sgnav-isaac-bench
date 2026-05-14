#!/usr/bin/env bash
set -euo pipefail

SGNAV_ROOT="${SGNAV_ROOT:-/home/echo/SG-Nav}"
VLLM_ENV="${VLLM_ENV:-$SGNAV_ROOT/.mamba/envs/sg-nav-vllm}"
RUN_VLLM="$SGNAV_ROOT/run_vllm.sh"

if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "vLLM executable not found: $VLLM_ENV/bin/vllm" >&2
  echo "Create the vLLM env from /home/echo/SG-Nav first, or set VLLM_ENV." >&2
  exit 1
fi

if [[ ! -x "$RUN_VLLM" ]]; then
  echo "SG-Nav vLLM launcher not found: $RUN_VLLM" >&2
  exit 1
fi

export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_MODEL="${VLLM_MODEL:-qwen3-vl-8b-instruct}"

exec "$RUN_VLLM" "$@"
