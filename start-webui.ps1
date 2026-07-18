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

# Best-effort: "shared GPU memory" (pinned system RAM mapped for the GPU,
# what Task Manager's GPU tab shows) is a Windows/WDDM concept with no
# nvidia-smi/Linux equivalent, so it can't be read from inside the
# container -- read it here instead. There's no "Shared Limit" perf
# counter on this system, so the total is estimated as half of host RAM
# (Windows' default shared-GPU-memory pool policy), clearly labeled as an
# estimate in the dashboard.
try {
    $sharedSamples = (Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples
    $maxShared = ($sharedSamples | Sort-Object CookedValue -Descending | Select-Object -First 1).CookedValue
    $env:HOST_GPU_SHARED_USED_GB = [math]::Round($maxShared / 1GB, 2)
    if ($env:HOST_RAM_TOTAL_GB) {
        $env:HOST_GPU_SHARED_TOTAL_GB = [math]::Round([double]$env:HOST_RAM_TOTAL_GB * 0.5, 1)
    }
} catch {
    # Non-fatal -- the dashboard just omits the shared-GPU-memory gauge.
}

docker compose -f docker-compose.webui.yml up -d --build

Write-Host ""
Write-Host "MoE-Infinity API server : http://localhost:8000/v1"
Write-Host "Open WebUI chat         : http://localhost:3000"
Write-Host "Status dashboard        : http://localhost:8600"
Write-Host ""
Write-Host "The model can take a while to load on first start (weight download + offload build)."
Write-Host "Follow progress with: docker compose -f docker-compose.webui.yml logs -f moe-infinity"
