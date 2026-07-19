#!/usr/bin/env bash
# Checks the status of the MoE-Infinity + Open WebUI stack, including model
# download/load progress for the moe-infinity service.
# Usage: ./check-webui-status.sh
set -uo pipefail
export MSYS_NO_PATHCONV=1

echo "== Containers =="
docker ps -a --filter "name=moe-infinity" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "== HuggingFace cache size (model download progress) =="
docker exec moe-infinity-server du -sh /root/.cache/huggingface 2>/dev/null \
  || echo "moe-infinity-server not reachable"

echo
echo "== Offload dir size (starts filling after download completes) =="
docker exec moe-infinity-server du -sh /offload 2>/dev/null \
  || echo "moe-infinity-server not reachable"

echo
echo "== /health endpoint =="
curl -s -m 5 http://localhost:8000/health || echo "not responding yet (still starting/loading)"
echo

echo
echo "== Last 15 log lines (moe-infinity-server) =="
docker logs moe-infinity-server --tail 15 2>&1
