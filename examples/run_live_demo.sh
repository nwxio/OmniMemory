#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi

MODE="${1:-extended}"
if [[ "$MODE" != "short" && "$MODE" != "extended" ]]; then
  echo "Usage: ./examples/run_live_demo.sh [short|extended]"
  exit 1
fi

echo "Running Memory-MCP live demo session ($MODE)..."
echo "Tip: start your screen recording now."

PYTHONPATH=. python3 examples/live_memory_session_demo.py --mode "$MODE"
