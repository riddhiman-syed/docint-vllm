#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Override any of these as needed, e.g.:
#   VLLM_SERVE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ VLLM_MAX_MODEL_LEN=4096 ./scripts/start_vllm.sh
VLLM_SERVE_MODEL="${VLLM_SERVE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct-AWQ}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"  # add --enforce-eager here if CUDA graphs misbehave

if is_running vllm; then
    echo "vLLM already running (pid $(cat "$RUN_DIR/vllm.pid"))"
    exit 0
fi

if already_up "http://localhost:$VLLM_PORT/health"; then
    echo "vLLM is already up at localhost:$VLLM_PORT (not started by this script — leaving it alone)."
    echo "stop_all.sh won't manage it either, since there's no PID on record for it."
    exit 0
fi

# Belt-and-suspenders: these are no-ops if flashinfer-python isn't
# installed at all (recommended), but harmless to keep set either way.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1

start_bg vllm "$REPO_ROOT/logs/vllm.log" \
    vllm serve "$VLLM_SERVE_MODEL" \
    --quantization awq \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
    --attention-backend TRITON_ATTN \
    --port "$VLLM_PORT" \
    $VLLM_EXTRA_ARGS

echo "First boot on sm_120 GPUs compiles Triton kernels from scratch — this can take 10-20+ minutes."
wait_for_http "http://localhost:$VLLM_PORT/health" "vLLM" 1800
