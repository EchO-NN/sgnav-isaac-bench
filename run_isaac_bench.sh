#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SG_NAV_ENV="${SG_NAV_ENV:-/home/echo/SG-Nav/.mamba/envs/sg-nav}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -z "${GROUNDING_DINO_ROOT:-}" && -d "/home/echo/SG-Nav/GroundingDINO" ]]; then
  export GROUNDING_DINO_ROOT="/home/echo/SG-Nav/GroundingDINO"
fi
if [[ -z "${GROUNDING_DINO_CHECKPOINT:-}" && -f "$ROOT_DIR/data/models/groundingdino_swinb_cogcoor.pth" ]]; then
  export GROUNDING_DINO_CHECKPOINT="$ROOT_DIR/data/models/groundingdino_swinb_cogcoor.pth"
fi
if [[ -z "${GROUNDING_DINO_CONFIG:-}" && -f "$ROOT_DIR/data/models/GroundingDINO_SwinB.cfg.py" ]]; then
  export GROUNDING_DINO_CONFIG="$ROOT_DIR/data/models/GroundingDINO_SwinB.cfg.py"
fi

exec "$SG_NAV_ENV/bin/python" "$@"
