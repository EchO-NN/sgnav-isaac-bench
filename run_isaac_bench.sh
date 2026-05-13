#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SG_NAV_ENV="${SG_NAV_ENV:-/home/echo/SG-Nav/.mamba/envs/sg-nav}"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

exec "$SG_NAV_ENV/bin/python" "$@"

