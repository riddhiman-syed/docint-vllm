#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== 1/3: Starting Qdrant ==="
"$REPO_ROOT/scripts/start_qdrant.sh"

echo "=== 2/3: Starting vLLM ==="
"$REPO_ROOT/scripts/start_vllm.sh"

echo "=== 3/3: Starting API ==="
"$REPO_ROOT/scripts/start_api.sh"

echo ""
echo "All services up:"
echo "  Qdrant: http://localhost:6333"
echo "  vLLM:   http://localhost:${VLLM_PORT:-8000}"
echo "  API:    http://localhost:${API_PORT:-8080}"
echo ""
echo "Logs: $REPO_ROOT/logs/{qdrant,vllm,api}.log"
echo "Stop everything with: ./scripts/stop_all.sh"
