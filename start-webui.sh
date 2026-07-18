#!/usr/bin/env bash
# Starts the MoE-Infinity OpenAI-compatible server + Open WebUI chat frontend.
# Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
# Usage: ./start-webui.sh
#        ./start-webui.sh openai/gpt-oss-20b
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  export MOE_MODEL="$1"
fi

# Best-effort: lets the model-status dashboard show true host RAM usage,
# not just what's visible inside Docker Desktop's WSL2 VM. Works on native
# Linux (free) and falls back to powershell.exe when running under Git Bash
# on Windows; non-fatal if neither is available.
if command -v free >/dev/null 2>&1; then
  export HOST_RAM_TOTAL_GB="$(free -g | awk '/^Mem:/{print $2}')"
elif command -v powershell.exe >/dev/null 2>&1; then
  export HOST_RAM_TOTAL_GB="$(powershell.exe -NoProfile -Command \
    '[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB, 1)' \
    2>/dev/null | tr -d '\r')"
fi

docker compose -f docker-compose.webui.yml up -d --build

echo ""
echo "MoE-Infinity API server : http://localhost:8000/v1"
echo "Open WebUI chat         : http://localhost:3000"
echo "Status dashboard        : http://localhost:8600"
echo ""
echo "The model can take a while to load on first start (weight download + offload build)."
echo "Follow progress with: docker compose -f docker-compose.webui.yml logs -f moe-infinity"
