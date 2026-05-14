#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-python}"
CODEX_CMD="${CODEX_CMD:-codex}"
MAX_HOURS="${MAX_HOURS:-5}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
ROUND_TIMEOUT_MINUTES="${ROUND_TIMEOUT_MINUTES:-45}"
REMOTE_HOST="${REMOTE_HOST:-sheng-xiang@100.64.0.4}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Tgprediction}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python}"
MODEL="${MODEL:-gpt-5.5}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"

exec "$PYTHON_BIN" -u scripts/codex_universal_tg_agent_loop.py \
  --max-hours "$MAX_HOURS" \
  --max-rounds "$MAX_ROUNDS" \
  --round-timeout-minutes "$ROUND_TIMEOUT_MINUTES" \
  --codex-cmd "$CODEX_CMD" \
  --codex-sandbox "${CODEX_SANDBOX:-danger-full-access}" \
  --codex-approval "${CODEX_APPROVAL:-never}" \
  --model "$MODEL" \
  --reasoning-effort "$REASONING_EFFORT" \
  --remote-host "$REMOTE_HOST" \
  --remote-project-dir "$REMOTE_PROJECT_DIR" \
  --remote-python "$REMOTE_PYTHON" \
  "$@"
