#!/usr/bin/env bash
# Starts the MoE-Infinity OpenAI-compatible server + Open WebUI chat frontend.
# Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
# Usage: ./start-webui.sh
#        ./start-webui.sh openai/gpt-oss-20b
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  export MOE_MODEL="$1"
fi

docker compose -f docker-compose.webui.yml up -d --build

echo ""
echo "MoE-Infinity API server : http://localhost:8000/v1"
echo "Open WebUI chat         : http://localhost:3000"
echo ""
echo "The model can take a while to load on first start (weight download + offload build)."
echo "Follow progress with: docker compose -f docker-compose.webui.yml logs -f moe-infinity"
