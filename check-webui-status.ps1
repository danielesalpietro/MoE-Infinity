# Checks the status of the MoE-Infinity + Open WebUI stack, including model
# download/load progress for the moe-infinity service.
# Usage: .\check-webui-status.ps1
$ErrorActionPreference = "Continue"

Write-Host "== Containers ==" -ForegroundColor Cyan
docker ps -a --filter "name=moe-infinity" --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"

Write-Host ""
Write-Host "== HuggingFace cache size (model download progress) ==" -ForegroundColor Cyan
docker exec moe-infinity-server du -sh /root/.cache/huggingface

Write-Host ""
Write-Host "== Offload dir size (starts filling after download completes) ==" -ForegroundColor Cyan
docker exec moe-infinity-server du -sh /offload

Write-Host ""
Write-Host "== /health endpoint ==" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri http://localhost:8000/health -TimeoutSec 5
    $health | ConvertTo-Json -Compress
} catch {
    Write-Host "not responding yet (still starting/loading)"
}

Write-Host ""
Write-Host "== Last 15 log lines (moe-infinity-server) ==" -ForegroundColor Cyan
docker logs moe-infinity-server --tail 15
