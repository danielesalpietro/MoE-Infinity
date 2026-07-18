# Pre-flight check: verifies the host meets the requirements to build and
# run the MoE-Infinity + Open WebUI stack (docker-compose.webui.yml) before
# you spend time on a build that will run out of RAM/VRAM/disk partway
# through. Run this before .\start-webui.ps1.
#
# Usage: .\check-system-requirements.ps1
$ErrorActionPreference = "Continue"

# Thresholds (GB unless noted). MIN = will very likely fail or thrash;
# RECOMMENDED = comfortable for a small MoE model like
# deepseek-ai/DeepSeek-V2-Lite-Chat (~30GB download).
$MinCpuCores = 4
$RecommendedCpuCores = 8
$MinRamGB = 16
$RecommendedRamGB = 32
$MinVramGB = 8
$RecommendedVramGB = 16
$MinDiskFreeGB = 40
$RecommendedDiskFreeGB = 80

$passCount = 0
$warnCount = 0
$failCount = 0

function Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green; $script:passCount++ }
function Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow; $script:warnCount++ }
function Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red; $script:failCount++ }

Write-Host "== Docker ==" -ForegroundColor Cyan
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCmd) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        Pass "Docker is installed and the daemon is reachable"
    } else {
        Fail "Docker is installed but the daemon is not reachable (is Docker Desktop running?)"
    }
    $composeVersion = docker compose version --short 2>$null
    if ($LASTEXITCODE -eq 0) {
        Pass "Docker Compose v2 is available ($composeVersion)"
    } else {
        Fail "Docker Compose v2 plugin not found (docker compose ...)"
    }
} else {
    Fail "Docker is not installed or not on PATH"
}

Write-Host ""
Write-Host "== CPU ==" -ForegroundColor Cyan
$cpuCores = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if ($cpuCores -ge $RecommendedCpuCores) {
    Pass "$cpuCores logical CPU cores (recommended: ${RecommendedCpuCores}+)"
} elseif ($cpuCores -ge $MinCpuCores) {
    Warn "$cpuCores logical CPU cores (minimum: $MinCpuCores, recommended: ${RecommendedCpuCores}+)"
} else {
    Fail "$cpuCores logical CPU cores (below minimum: $MinCpuCores)"
}

Write-Host ""
Write-Host "== RAM ==" -ForegroundColor Cyan
$os = Get-CimInstance Win32_OperatingSystem
$ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
if ($ramGB -ge $RecommendedRamGB) {
    Pass "${ramGB}GB total RAM (recommended: ${RecommendedRamGB}GB+)"
} elseif ($ramGB -ge $MinRamGB) {
    Warn "${ramGB}GB total RAM (minimum: ${MinRamGB}GB, recommended: ${RecommendedRamGB}GB+). MoE-Infinity offloads experts to host RAM as an intermediate tier -- tight RAM here causes heavy swapping."
} else {
    Fail "${ramGB}GB total RAM (below minimum: ${MinRamGB}GB)"
}

$wslConfigPath = "$env:UserProfile\.wslconfig"
if (Test-Path $wslConfigPath) {
    $wslMemLine = Select-String -Path $wslConfigPath -Pattern "^\s*memory\s*=\s*(\d+)\s*GB" -ErrorAction SilentlyContinue
    if ($wslMemLine) {
        $wslMemGB = [int]$wslMemLine.Matches[0].Groups[1].Value
        if ($wslMemGB -lt $MinRamGB) {
            Warn ".wslconfig limits WSL2 to ${wslMemGB}GB RAM (below minimum: ${MinRamGB}GB) -- Docker Desktop containers run inside this limit regardless of total host RAM"
        } else {
            Pass ".wslconfig allows WSL2 up to ${wslMemGB}GB RAM"
        }
    }
}

Write-Host ""
Write-Host "== GPU / VRAM ==" -ForegroundColor Cyan
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuLine = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($gpuLine) {
        $parts = $gpuLine -split ","
        $gpuName = $parts[0].Trim()
        $vramGB = [math]::Round([int]($parts[1].Trim()) / 1024, 1)
        if ($vramGB -ge $RecommendedVramGB) {
            Pass "$gpuName, ${vramGB}GB VRAM (recommended: ${RecommendedVramGB}GB+)"
        } elseif ($vramGB -ge $MinVramGB) {
            Warn "$gpuName, ${vramGB}GB VRAM (minimum: ${MinVramGB}GB). Lower MOE_DEVICE_MEMORY_RATIO if you hit OOM."
        } else {
            Fail "$gpuName, ${vramGB}GB VRAM (below minimum: ${MinVramGB}GB)"
        }
    } else {
        Fail "nvidia-smi found but returned no GPU -- driver or GPU access problem"
    }
} else {
    Fail "nvidia-smi not found -- MoE-Infinity requires an NVIDIA GPU with a working driver, reachable from Docker Desktop (NVIDIA Container Toolkit / WSL2 GPU passthrough)"
}

Write-Host ""
Write-Host "== Disk space (C:) ==" -ForegroundColor Cyan
$diskFreeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
if ($diskFreeGB -ge $RecommendedDiskFreeGB) {
    Pass "${diskFreeGB}GB free on C: (recommended: ${RecommendedDiskFreeGB}GB+)"
} elseif ($diskFreeGB -ge $MinDiskFreeGB) {
    Warn "${diskFreeGB}GB free on C: (minimum: ${MinDiskFreeGB}GB). The model download (~30GB) plus the built image (~15-30GB) and offload storage can get tight."
} else {
    Fail "${diskFreeGB}GB free on C: (below minimum: ${MinDiskFreeGB}GB)"
}

Write-Host ""
Write-Host "== Summary ==" -ForegroundColor Cyan
Write-Host "  $passCount passed, $warnCount warning(s), $failCount failure(s)"
if ($failCount -gt 0) {
    Write-Host "  Result: NOT READY -- resolve the [FAIL] items above before running .\start-webui.ps1" -ForegroundColor Red
    exit 2
} elseif ($warnCount -gt 0) {
    Write-Host "  Result: READY WITH WARNINGS -- .\start-webui.ps1 should work but may be slow or need lower MOE_* ratios" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "  Result: READY" -ForegroundColor Green
    exit 0
}
