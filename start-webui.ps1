# Starts the MoE-Infinity OpenAI-compatible server + Open WebUI chat frontend.
# Requires an NVIDIA GPU with the NVIDIA Container Toolkit on the host.
# Usage: .\start-webui.ps1
#        .\start-webui.ps1 -Model openai/gpt-oss-20b
param(
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"

if ($Model -ne "") {
    $env:MOE_MODEL = $Model
}

docker compose -f docker-compose.webui.yml up -d --build

Write-Host ""
Write-Host "MoE-Infinity API server : http://localhost:8000/v1"
Write-Host "Open WebUI chat         : http://localhost:3000"
Write-Host ""
Write-Host "The model can take a while to load on first start (weight download + offload build)."
Write-Host "Follow progress with: docker compose -f docker-compose.webui.yml logs -f moe-infinity"
