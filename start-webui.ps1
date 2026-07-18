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

# Best-effort: lets the model-status dashboard show true host RAM usage,
# not just what's visible inside Docker Desktop's WSL2 VM.
try {
    $env:HOST_RAM_TOTAL_GB = [math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB, 1)
} catch {
    # Non-fatal -- the dashboard just omits the host-RAM gauge.
}

docker compose -f docker-compose.webui.yml up -d --build

Write-Host ""
Write-Host "MoE-Infinity API server : http://localhost:8000/v1"
Write-Host "Open WebUI chat         : http://localhost:3000"
Write-Host "Status dashboard        : http://localhost:8600"
Write-Host ""
Write-Host "The model can take a while to load on first start (weight download + offload build)."
Write-Host "Follow progress with: docker compose -f docker-compose.webui.yml logs -f moe-infinity"
