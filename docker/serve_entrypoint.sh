#!/usr/bin/env bash
# Entrypoint for the MoE-Infinity serving image (docker/Dockerfile.serve).
# Translates MOE_* environment variables into api_server_v2 CLI flags so the
# model/served config can be changed via `docker run -e` / compose env,
# without rebuilding the image.
set -euo pipefail

args=(
  --model "${MOE_MODEL}"
  --offload-dir "${MOE_OFFLOAD_DIR}"
  --host "${MOE_HOST}"
  --port "${MOE_PORT}"
  --device-memory-ratio "${MOE_DEVICE_MEMORY_RATIO}"
  --kv-cache-ratio "${MOE_KV_CACHE_RATIO}"
  --max-batch-size "${MOE_MAX_BATCH_SIZE}"
)

if [ -n "${MOE_API_KEYS:-}" ]; then
  args+=(--api-key "${MOE_API_KEYS}")
fi

if [ "${MOE_ENABLE_PREFIX_CACHING:-0}" = "1" ]; then
  args+=(--enable-prefix-caching)
fi

if [ "${MOE_ENABLE_CONTEXTPILOT:-0}" = "1" ]; then
  args+=(--enable-contextpilot)
fi

if [ -n "${MOE_LOG_LEVEL:-}" ]; then
  args+=(--log-level "$(echo "${MOE_LOG_LEVEL}" | tr '[:upper:]' '[:lower:]')")
fi

exec python -m moe_infinity.entrypoints.openai.api_server_v2 "${args[@]}" "$@"
